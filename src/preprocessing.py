"""Engenharia de atributos e pré-processadores para os modelos de severidade.

Seguem a seção 3.3 do artigo: variáveis temporais, meteorológicas, viárias e
operacionais explicando a severidade (SEM_VITIMAS, COM_FERIDOS, FATAL).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ALVO = "alvo_severidade"
CLASSES = ["SEM_VITIMAS", "COM_FERIDOS", "FATAL"]

# Colunas que já "contêm" a resposta (só são conhecidas depois de contar as
# vítimas). Usá-las como entrada seria vazamento de dados (data leakage).
COLUNAS_VAZAMENTO = [
    "classificacao_acidente",
    "mortos",
    "feridos_leves",
    "feridos_graves",
    "ilesos",
    "pessoas",
    "indice_severidade",
]

# Componentes possíveis de "tracado_via" (a coluna traz combinações, ex.: "CURVA;DECLIVE")
COMPONENTES_TRACADO = [
    "RETA", "CURVA", "ACLIVE", "DECLIVE", "INTERSEÇÃO DE VIAS", "ROTATÓRIA",
    "RETORNO REGULAMENTADO", "EM OBRAS", "VIADUTO", "PONTE",
    "DESVIO TEMPORÁRIO", "TÚNEL",
]

CATEGORICAS_BASE = [
    "dia_semana", "fase_dia", "condicao_metereologica", "tipo_pista",
    "uso_solo", "sentido_via", "uf", "br",
]
NUMERICAS_BASE = ["hora", "hora_sin", "hora_cos", "mes", "latitude", "longitude"]
# Extras opcionais (descrevem a dinâmica do sinistro; não fazem parte do artigo)
CATEGORICAS_EXTRA = ["causa_acidente", "tipo_acidente"]


def criar_features(df: pd.DataFrame, incluir_causa_tipo: bool = False):
    """Devolve (X, y, colunas_categoricas, colunas_numericas)."""
    d = df.copy()

    # Hora do dia é circular (23h está perto de 0h): usamos seno/cosseno também
    d["hora_sin"] = np.sin(2 * np.pi * d["hora"] / 24)
    d["hora_cos"] = np.cos(2 * np.pi * d["hora"] / 24)

    # Um indicador 0/1 para cada componente do traçado da via
    flags = []
    for comp in COMPONENTES_TRACADO:
        nome = "tracado_" + comp.lower().replace(" ", "_").replace("ç", "c").replace("ã", "a") \
            .replace("ó", "o").replace("ú", "u").replace("é", "e")
        d[nome] = d["tracado_via"].fillna("").str.contains(comp, regex=False).astype(int)
        flags.append(nome)

    d["condicao_metereologica"] = d["condicao_metereologica"].fillna("NÃO INFORMADO")
    d["br"] = d["br"].astype(int).astype(str)  # número da BR é um rótulo, não uma quantidade

    categoricas = list(CATEGORICAS_BASE) + (CATEGORICAS_EXTRA if incluir_causa_tipo else [])
    numericas = list(NUMERICAS_BASE) + flags

    X = d[categoricas + numericas].copy()
    X[categoricas] = X[categoricas].astype(str)
    y = d[ALVO]
    return X, y, categoricas, numericas


def grupos_de_variaveis(colunas):
    """Agrupa variáveis que carregam a mesma informação (ex.: hora, hora_sin, hora_cos, fase_dia).

    Necessário para a importância por permutação: se embaralharmos só uma delas, as
    parecidas compensam e a importância sai subestimada (ou até negativa).
    """
    grupos = {
        "Horário (hora, fase do dia)": ["hora", "hora_sin", "hora_cos", "fase_dia"],
        "Calendário (dia da semana, mês)": ["dia_semana", "mes"],
        "Localização (lat/lon, UF, BR)": ["latitude", "longitude", "uf", "br"],
        "Tipo de pista": ["tipo_pista"],
        "Traçado da via": [c for c in colunas if c.startswith("tracado_")],
        "Uso do solo": ["uso_solo"],
        "Sentido da via": ["sentido_via"],
        "Condição meteorológica": ["condicao_metereologica"],
        "Causa do acidente": ["causa_acidente"],
        "Tipo de acidente": ["tipo_acidente"],
    }
    return {nome: [c for c in cols if c in colunas] for nome, cols in grupos.items()
            if any(c in colunas for c in cols)}


def dividir_treino_teste(X, y, teste=0.2, semente=42):
    """Divisão 80/20 estratificada: mantém a proporção das classes nos dois lados."""
    return train_test_split(X, y, test_size=teste, stratify=y, random_state=semente)


def preprocessador_arvores(categoricas, numericas):
    """Para Random Forest: categorias viram números inteiros (árvores aceitam bem)."""
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categoricas),
            ("num", "passthrough", numericas),
        ],
        verbose_feature_names_out=False,
    )


def preprocessador_linear(categoricas, numericas):
    """Para Regressão Logística: one-hot nas categorias e escala nas numéricas."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), categoricas),
            ("num", StandardScaler(), numericas),
        ]
    )
