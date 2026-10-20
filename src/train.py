"""Treina e avalia os modelos descritos na metodologia do artigo.

O que este script faz, em ordem:
    1. Carrega a base tratada da PRF e monta as variáveis de entrada.
    2. Divide os dados em treino (80%) e teste (20%).
    3. Treina três modelos e compara o desempenho de cada um no teste.
       Em seguida, repete a comparação com validação cruzada (só nos dados de treino).
    4. Mede quais variáveis mais pesam na previsão (importância das variáveis).
    5. Salva métricas, tabelas, figuras e o modelo treinado.

Uso (a partir da raiz do projeto):
    python -m src.train                       # configuração do artigo
    python -m src.train --incluir-causa-tipo  # experimento extra com causa e tipo do acidente
    python -m src.train --n-arvores 100       # mais rápido, para testar
    python -m src.train --folds 0             # pula a validação cruzada (mais rápido)

Experimentos complementares (validação temporal e espacial, ajuste de hiperparâmetros e
da regra de decisão) ficam em src/experimentos.py e não alteram os resultados deste script.

Modelos comparados (mesmos dados, mesma divisão 80/20):
    0. Baseline: sempre prevê a classe mais comum (referência mínima)
    1. Random Forest Classifier (class_weight="balanced")
    2. Regressão Logística Multinomial (class_weight="balanced")
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# Funções dos outros arquivos da pasta src/
from src.data_loader import RAIZ_PROJETO, carregar_base
from src.preprocessing import (CLASSES, criar_features, dividir_treino_teste,
                               grupos_de_variaveis, preprocessador_arvores, preprocessador_linear)
from src.visualization import PASTA_FIGURAS, plotar_importancia, plotar_matrizes_confusao

# "Semente" do sorteio aleatório. Fixar o valor faz o resultado ser o mesmo
# toda vez que o script roda, o que permite reproduzir os números do artigo.
SEMENTE = 42
# Hiperparâmetros do Random Forest do artigo, fixados a priori (sem ajuste). Ficam numa constante para
# que src/experimentos.py use exatamente os mesmos valores como ponto de partida.
#   max_depth        = profundidade máxima de cada árvore (evita decorar os dados)
#   min_samples_leaf = mínimo de acidentes em cada folha (também evita decorar)
#   class_weight     = "balanced" dá mais peso às classes raras (ex.: FATAL)
PARAMS_RF = dict(max_depth=20, min_samples_leaf=5, class_weight="balanced")
PASTA_RESULTADOS = RAIZ_PROJETO / "results"
PASTA_SAIDAS = RAIZ_PROJETO / "data" / "outputs"  # ignorada pelo git (arquivos grandes)


def avaliar(nome, modelo, X_teste, y_teste):
    """Calcula as métricas da seção 3.5 do artigo para um modelo já treinado.

    Devolve: (dicionário de métricas, matriz de confusão, relatório em texto).
    """
    # Classe prevista para cada acidente do teste (ex.: "FATAL")
    previsto = modelo.predict(X_teste)
    # Probabilidade que o modelo dá a cada classe (usada no cálculo do AUC)
    proba = modelo.predict_proba(X_teste)
    ordem = list(modelo.classes_)  # ordem alfabética, a mesma das colunas de proba
    # Relatório por classe (precisão, recall, F1), em formato de dicionário
    rel_dict = classification_report(y_teste, previsto, labels=CLASSES, output_dict=True, zero_division=0)
    metricas = {
        "modelo": nome,
        # Acurácia: % de acidentes classificados corretamente (enganosa com classes desbalanceadas)
        "acuracia": accuracy_score(y_teste, previsto),
        # F1 macro: média do F1 das 3 classes, todas com o mesmo peso (métrica principal do artigo)
        "f1_macro": f1_score(y_teste, previsto, average="macro"),
        # F1 weighted: média do F1 ponderada pelo tamanho de cada classe
        "f1_weighted": f1_score(y_teste, previsto, average="weighted"),
        # AUC: quão bem o modelo separa uma classe das demais (0,5 = acaso, 1,0 = perfeito)
        "roc_auc_ovr_macro": roc_auc_score(y_teste, proba, multi_class="ovr",
                                           average="macro", labels=ordem),
        "f1_por_classe": dict(zip(CLASSES, f1_score(y_teste, previsto, labels=CLASSES, average=None))),
        # Recall: de todos os acidentes reais da classe, quantos o modelo encontrou
        "recall_por_classe": {c: rel_dict[c]["recall"] for c in CLASSES},
    }
    # Matriz de confusão: linhas = classe real, colunas = classe prevista
    cm = confusion_matrix(y_teste, previsto, labels=CLASSES)
    relatorio = classification_report(y_teste, previsto, labels=CLASSES, digits=3, zero_division=0)
    return metricas, cm, relatorio


def validar_cruzado(modelos, X_treino, y_treino, folds):
    """Validação cruzada estratificada: mede o quanto o resultado varia entre divisões diferentes.

    Os dados de TREINO são divididos em `folds` partes (mantendo a proporção das classes).
    Cada parte serve uma vez de "teste" enquanto o modelo é treinado nas demais. Assim vemos a
    média e o desvio padrão de cada métrica, e não só o valor de uma única divisão.
    O conjunto de teste (20%) não é usado aqui: continua reservado para a avaliação final.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEMENTE)
    # nome que usamos -> nome da métrica no scikit-learn
    metricas = {"f1_macro": "f1_macro", "acuracia": "accuracy", "roc_auc_ovr_macro": "roc_auc_ovr"}
    linhas = []
    for nome, modelo in modelos.items():
        t0 = time.time()
        # cross_validate treina uma cópia nova do modelo em cada divisão
        res = cross_validate(modelo, X_treino, y_treino, cv=cv, scoring=metricas, n_jobs=1)
        linha = {"modelo": nome, "folds": folds}
        for chave in metricas:
            valores = res["test_" + chave]
            linha[f"{chave}_media"] = float(valores.mean())
            linha[f"{chave}_desvio"] = float(valores.std(ddof=1))  # desvio padrão amostral
        # F1 macro de cada parte, para ver se a ordem entre os modelos se mantém em todas elas
        linha["f1_macro_por_parte"] = ";".join(f"{v:.4f}" for v in res["test_f1_macro"])
        linha["segundos"] = round(time.time() - t0, 1)
        linhas.append(linha)
        print(f"  {nome:34s} F1 macro = {linha['f1_macro_media']:.3f} ± {linha['f1_macro_desvio']:.3f}  "
              f"| AUC = {linha['roc_auc_ovr_macro_media']:.3f} ± {linha['roc_auc_ovr_macro_desvio']:.3f}")
    return pd.DataFrame(linhas)


def main():
    # --- opções de linha de comando (todas opcionais) ---
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incluir-causa-tipo", action="store_true",
                    help="inclui causa_acidente e tipo_acidente como variáveis (fora do artigo)")
    ap.add_argument("--n-arvores", type=int, default=200, help="árvores do Random Forest")
    ap.add_argument("--amostra-permutacao", type=int, default=20000,
                    help="linhas do teste usadas na importância por permutação")
    ap.add_argument("--folds", type=int, default=5,
                    help="partes da validação cruzada nos dados de treino (0 = não executa)")
    args = ap.parse_args()

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)
    # No experimento extra, os arquivos ganham um sufixo para não sobrescrever os do artigo
    sufixo = "_com_causa_tipo" if args.incluir_causa_tipo else ""

    # --- 1. dados ---
    print("Carregando a base...")
    df = carregar_base()
    # X = variáveis de entrada, y = gravidade (o que queremos prever)
    X, y, cat, num = criar_features(df, incluir_causa_tipo=args.incluir_causa_tipo)
    # Treino (80%): o modelo aprende. Teste (20%): dados que ele nunca viu, para avaliar de verdade.
    X_tr, X_te, y_tr, y_te = dividir_treino_teste(X, y, teste=0.2, semente=SEMENTE)
    print(f"Variáveis: {X.shape[1]} | treino: {len(X_tr):,} | teste: {len(X_te):,}".replace(",", "."))

    # --- 2. definição dos modelos ---
    # Cada modelo é um "Pipeline": primeiro o pré-processamento (converter textos em números),
    # depois o classificador. Assim o mesmo tratamento vale no treino e no teste.
    modelos = {
        # Referência mínima: ignora as variáveis e sempre chuta a classe mais comum.
        "Baseline (classe mais comum)": Pipeline([
            ("pre", preprocessador_arvores(cat, num)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        # Modelo principal: conjunto de árvores de decisão que votam juntas.
        #   n_estimators     = número de árvores
        #   PARAMS_RF        = profundidade máxima, folha mínima e ponderação de classes (ver acima)
        #   n_jobs=-1        = usa todos os núcleos do processador
        "Random Forest": Pipeline([
            ("pre", preprocessador_arvores(cat, num)),
            ("clf", RandomForestClassifier(
                n_estimators=args.n_arvores, n_jobs=-1, random_state=SEMENTE, **PARAMS_RF)),
        ]),
        # Modelo estatístico clássico, usado como comparação com o Random Forest.
        #   max_iter = número máximo de iterações até o ajuste convergir
        "Regressão Logística Multinomial": Pipeline([
            ("pre", preprocessador_linear(cat, num)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]),
    }

    # --- 3. treino e avaliação de cada modelo ---
    todas, matrizes, relatorios = [], {}, []
    for nome, modelo in modelos.items():
        t0 = time.time()
        modelo.fit(X_tr, y_tr)  # treino: o modelo aprende com os dados de treino
        met, cm, rel = avaliar(nome, modelo, X_te, y_te)  # avaliação nos dados de teste
        met["segundos_treino_e_avaliacao"] = round(time.time() - t0, 1)
        todas.append(met)
        relatorios.append(f"=== {nome} ===\n{rel}")
        # O baseline não entra na figura da matriz de confusão (só preveria uma classe)
        if nome != "Baseline (classe mais comum)":
            matrizes[nome] = cm
        print(f"{nome:34s} acurácia={met['acuracia']:.3f}  F1 macro={met['f1_macro']:.3f}  "
              f"F1 weighted={met['f1_weighted']:.3f}  AUC={met['roc_auc_ovr_macro']:.3f}")

    # --- 3b. validação cruzada (robustez da comparação entre os modelos) ---
    # Roda depois da avaliação no teste e só usa o treino, então não altera os números acima.
    cv_tabela = None
    if args.folds >= 2:
        print(f"Validação cruzada estratificada ({args.folds} partes) nos dados de treino...")
        cv_tabela = validar_cruzado(modelos, X_tr, y_tr, args.folds)

    # --- 4. importância das variáveis (Random Forest) ---
    rf = modelos["Random Forest"]

    # (a) Feature Importance clássica do Random Forest (por impureza), uma linha por variável.
    #     Atenção: tende a favorecer variáveis contínuas ou com muitas categorias (lat, lon, mês).
    nomes = list(rf.named_steps["pre"].get_feature_names_out())
    tabela = pd.DataFrame(
        {"importancia_impureza": rf.named_steps["clf"].feature_importances_}, index=nomes
    ).sort_values("importancia_impureza", ascending=False)

    # (b) Importância por permutação, por GRUPO de variáveis: quanto o F1 macro cai no teste
    #     quando embaralhamos o grupo inteiro (mais confiável quando há variáveis redundantes).
    #     Ideia: se embaralhar uma variável faz o desempenho cair muito, ela era importante.
    print("Calculando importância por permutação de grupos...")
    grupos = grupos_de_variaveis(list(X.columns))
    # Usamos só uma amostra do teste para não demorar muito
    amostra = X_te.sample(min(args.amostra_permutacao, len(X_te)), random_state=SEMENTE)
    y_amostra = y_te.loc[amostra.index]
    # F1 macro original, sem embaralhar nada (ponto de comparação)
    base = f1_score(y_amostra, rf.predict(amostra), average="macro")
    rng = np.random.default_rng(SEMENTE)
    quedas = {}
    for nome_grupo, cols in grupos.items():
        repeticoes = []
        for _ in range(3):  # repete 3 vezes e tira a média, pois o embaralhamento é aleatório
            embaralhada = amostra.copy()
            ordem = rng.permutation(len(amostra))
            embaralhada[cols] = amostra[cols].to_numpy()[ordem]  # o grupo todo é embaralhado junto
            # Queda do F1 = F1 original - F1 com o grupo embaralhado
            repeticoes.append(base - f1_score(y_amostra, rf.predict(embaralhada), average="macro"))
        quedas[nome_grupo] = float(np.mean(repeticoes))
    tabela_grupos = pd.DataFrame({"queda_f1_macro": quedas}).sort_values("queda_f1_macro", ascending=False)
    print(tabela_grupos.to_string(float_format=lambda v: f"{v:.4f}"))

    # --- 5. salvar tudo ---
    # Métricas e relatório em texto, para consultar e citar no artigo
    (PASTA_RESULTADOS / f"metricas{sufixo}.json").write_text(
        json.dumps(todas, indent=2, ensure_ascii=False), encoding="utf-8")
    (PASTA_RESULTADOS / f"relatorio_classificacao{sufixo}.txt").write_text(
        "\n\n".join(relatorios), encoding="utf-8")
    # Resultado da validação cruzada (média e desvio padrão por modelo)
    if cv_tabela is not None:
        cv_tabela.to_csv(PASTA_RESULTADOS / f"validacao_cruzada{sufixo}.csv", index=False, encoding="utf-8")
    # Tabelas de importância das variáveis
    tabela.to_csv(PASTA_RESULTADOS / f"importancia_variaveis{sufixo}.csv", encoding="utf-8")
    tabela_grupos.to_csv(PASTA_RESULTADOS / f"importancia_grupos{sufixo}.csv", encoding="utf-8")
    # Figuras do artigo (06 = matrizes de confusão, 07 e 08 = importâncias)
    plotar_matrizes_confusao(matrizes, CLASSES, PASTA_FIGURAS / f"06_matrizes_confusao{sufixo}.png")
    plotar_importancia(tabela_grupos, "queda_f1_macro",
                       "Importância por grupo (queda do F1 macro ao embaralhar o grupo)",
                       PASTA_FIGURAS / f"07_importancia_grupos{sufixo}.png")
    plotar_importancia(tabela, "importancia_impureza",
                       "Importância por impureza (Feature Importance do Random Forest)",
                       PASTA_FIGURAS / f"08_importancia_impureza{sufixo}.png")
    # Modelo treinado, para reutilizar depois sem treinar de novo (não vai para o GitHub)
    joblib.dump(rf, PASTA_SAIDAS / f"random_forest{sufixo}.joblib", compress=3)
    print("Pronto. Resultados em results/ e figures/.")


# Só roda main() quando o arquivo é executado direto (python -m src.train),
# e não quando é apenas importado por outro arquivo.
if __name__ == "__main__":
    main()
