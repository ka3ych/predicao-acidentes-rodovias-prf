"""Experimentos complementares ao treinamento principal (src/train.py).

Respondem às limitações apontadas na discussão do artigo, sem alterar os números principais:

    1. temporal : treina com 2023-2024 e testa em 2025 (dados "do futuro"), em vez de misturar os anos.
    2. espacial : em cada rodada, estados inteiros ficam de fora do treino (validação "por grupos de UF").
    3. ajuste   : busca de hiperparâmetros (Random Forest e Regressão Logística) com validação
                  cruzada SOMENTE nos dados de treino; o teste só avalia o modelo final.
    4. limiar   : ajusta a regra de decisão do Random Forest com pesos por classe aplicados às
                  probabilidades, escolhidos com previsões "fora da amostra" do treino.

O conjunto de teste (20%) nunca é usado para escolher nada: só mede o resultado final de cada etapa.

Uso (a partir da raiz do projeto):
    python -m src.experimentos                          # roda as quatro etapas (cerca de 20 min)
    python -m src.experimentos --etapas temporal ajuste # só as etapas escolhidas
    python -m src.experimentos --n-candidatos 6         # busca menor, para testar mais rápido

Saídas em results/: validacao_temporal.csv, validacao_espacial.csv, ajuste_hiperparametros.csv,
ajuste_hiperparametros.json, modelos_ajustados.csv, ajuste_limiar.json e ajuste_limiar_curva.csv.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import (GroupKFold, ParameterSampler, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline

from src.data_loader import carregar_base
from src.preprocessing import (CLASSES, criar_features, dividir_treino_teste,
                               preprocessador_arvores, preprocessador_linear)
from src.train import PARAMS_RF, PASTA_RESULTADOS, SEMENTE

N_ARVORES = 200  # o mesmo número de árvores do treinamento principal

# Valores testados na busca de hiperparâmetros do Random Forest. Evitamos árvores sem limite de
# profundidade e folhas de 1 acidente, que gerariam modelos enormes (memória) e tenderiam a decorar.
GRADE_RF = {
    "max_depth": [10, 15, 20, 30],
    "min_samples_leaf": [2, 5, 10, 20],
    "max_features": ["sqrt", 0.3, 0.5],  # quantas variáveis cada árvore examina em cada divisão
    "class_weight": ["balanced", "balanced_subsample"],
}
# Intensidade da regularização da Regressão Logística (C menor = modelo mais "contido")
GRADE_C = [0.01, 0.1, 1.0, 10.0]
# Pesos testados para o ajuste da regra de decisão (1,0 = não mexe na probabilidade da classe)
GRADE_PESOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


# ----------------------------------------------------------------------------------------------
# Funções de apoio
# ----------------------------------------------------------------------------------------------
def montar_modelo(tipo, cat, num, **params):
    """Monta um Pipeline (pré-processamento + classificador) igual ao de src/train.py.

    tipo: "baseline", "rf" ou "lr". Os `params` sobrescrevem os valores padrão do artigo.
    """
    if tipo == "baseline":
        return Pipeline([("pre", preprocessador_arvores(cat, num)),
                         ("clf", DummyClassifier(strategy="most_frequent"))])
    if tipo == "rf":
        p = {**PARAMS_RF, **params}  # parte dos valores do artigo e troca só o que foi pedido
        return Pipeline([("pre", preprocessador_arvores(cat, num)),
                         ("clf", RandomForestClassifier(n_estimators=N_ARVORES, n_jobs=-1,
                                                        random_state=SEMENTE, **p))])
    if tipo == "lr":
        p = {"max_iter": 500, "class_weight": "balanced", **params}
        return Pipeline([("pre", preprocessador_linear(cat, num)),
                         ("clf", LogisticRegression(**p))])
    raise ValueError(tipo)


def resumir(y_real, previsto, proba, ordem):
    """Métricas de um conjunto de previsões (as mesmas do artigo, mais precisão e revocação de FATAL)."""
    rel = classification_report(y_real, previsto, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "acuracia": accuracy_score(y_real, previsto),
        "f1_macro": f1_score(y_real, previsto, average="macro"),
        "f1_weighted": f1_score(y_real, previsto, average="weighted"),
        "roc_auc_ovr_macro": roc_auc_score(y_real, proba, multi_class="ovr", average="macro", labels=ordem),
        "recall_fatal": rel["FATAL"]["recall"],      # dos fatais reais, quantos foram encontrados
        "precisao_fatal": rel["FATAL"]["precision"],  # dos alertas "fatal", quantos eram mesmo fatais
    }


def avaliar_modelo(modelo, X_te, y_te):
    """Previsões do modelo já treinado, resumidas nas métricas acima."""
    return resumir(y_te, modelo.predict(X_te), modelo.predict_proba(X_te), list(modelo.classes_))


def salvar_csv(linhas, nome):
    pd.DataFrame(linhas).to_csv(PASTA_RESULTADOS / nome, index=False, encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Etapa 1: validação temporal
# ----------------------------------------------------------------------------------------------
def etapa_temporal(X, y, anos, cat, num):
    """Treina com 2023-2024 e testa em 2025: os sinistros do teste acontecem DEPOIS dos do treino.

    Na divisão aleatória do artigo, sinistros do mesmo trecho e da mesma época podem estar no treino
    e no teste. Aqui isso não acontece, então o resultado é uma estimativa mais rigorosa.
    """
    print("\n[temporal] treino 2023-2024 -> teste 2025")
    tr, te = (anos <= 2024).to_numpy(), (anos == 2025).to_numpy()
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    print(f"  treino: {len(X_tr):,} | teste: {len(X_te):,} | % fatais treino: "
          f"{(y_tr == 'FATAL').mean():.2%} | % fatais teste: {(y_te == 'FATAL').mean():.2%}".replace(",", "."))
    linhas = []
    for nome, tipo in [("Baseline (classe mais comum)", "baseline"), ("Regressão Logística Multinomial", "lr"),
                       ("Random Forest", "rf")]:
        t0 = time.time()
        modelo = montar_modelo(tipo, cat, num).fit(X_tr, y_tr)
        met = avaliar_modelo(modelo, X_te, y_te)
        linhas.append({"modelo": nome, "n_treino": len(X_tr), "n_teste": len(X_te),
                       "pct_fatal_teste": float((y_te == "FATAL").mean()), **met,
                       "segundos": round(time.time() - t0, 1)})
        print(f"  {nome:34s} acurácia={met['acuracia']:.3f}  F1 macro={met['f1_macro']:.3f}  "
              f"AUC={met['roc_auc_ovr_macro']:.3f}  revocação fatal={met['recall_fatal']:.3f}")
    salvar_csv(linhas, "validacao_temporal.csv")


# ----------------------------------------------------------------------------------------------
# Etapa 2: validação espacial
# ----------------------------------------------------------------------------------------------
def etapa_espacial(X, y, uf, cat, num, partes=5):
    """Validação por grupos de UF: em cada rodada, alguns estados inteiros ficam fora do treino.

    O modelo precisa prever sinistros de estados que nunca viu (a UF e a BR dessas linhas são
    categorias desconhecidas e só a latitude e a longitude ainda dão pistas do local).
    """
    print(f"\n[espacial] {partes} rodadas, estados inteiros fora do treino")
    cv = GroupKFold(n_splits=partes)
    acumulado = {"Baseline (classe mais comum)": [], "Regressão Logística Multinomial": [], "Random Forest": []}
    tipos = {"Baseline (classe mais comum)": "baseline", "Regressão Logística Multinomial": "lr",
             "Random Forest": "rf"}
    ufs_fora = []
    for k, (tr, te) in enumerate(cv.split(X, y, groups=uf), 1):
        ufs_fora.append("/".join(sorted(uf.iloc[te].unique())))
        for nome, tipo in tipos.items():
            modelo = montar_modelo(tipo, cat, num).fit(X.iloc[tr], y.iloc[tr])
            acumulado[nome].append(avaliar_modelo(modelo, X.iloc[te], y.iloc[te]))
        print(f"  rodada {k}: {len(te):,} sinistros fora do treino ".replace(",", ".") +
              " | ".join(f"{n.split()[0]} F1={a[-1]['f1_macro']:.3f}" for n, a in acumulado.items()))
    linhas = []
    for nome, lista in acumulado.items():
        f1s = np.array([m["f1_macro"] for m in lista])
        linhas.append({
            "modelo": nome, "partes": partes,
            "f1_macro_media": f1s.mean(), "f1_macro_desvio": f1s.std(ddof=1),
            "acuracia_media": np.mean([m["acuracia"] for m in lista]),
            "roc_auc_ovr_macro_media": np.mean([m["roc_auc_ovr_macro"] for m in lista]),
            "recall_fatal_media": np.mean([m["recall_fatal"] for m in lista]),
            "f1_macro_por_parte": ";".join(f"{v:.4f}" for v in f1s),
        })
        print(f"  {nome:34s} F1 macro = {f1s.mean():.3f} ± {f1s.std(ddof=1):.3f}")
    salvar_csv(linhas, "validacao_espacial.csv")
    # Quais estados ficaram fora em cada rodada (para consulta)
    (PASTA_RESULTADOS / "validacao_espacial_ufs.txt").write_text(
        "\n".join(f"rodada {i}: {u}" for i, u in enumerate(ufs_fora, 1)), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Etapa 3: ajuste de hiperparâmetros
# ----------------------------------------------------------------------------------------------
def buscar(tipo, candidatos, X_tr, y_tr, cat, num, cv):
    """Avalia cada candidato por validação cruzada no treino e devolve a tabela (uma linha por candidato)."""
    linhas = []
    for i, params in enumerate(candidatos, 1):
        t0 = time.time()
        res = cross_validate(montar_modelo(tipo, cat, num, **params), X_tr, y_tr, cv=cv,
                             scoring="f1_macro", n_jobs=1)
        v = res["test_score"]
        linhas.append({"modelo": "Random Forest" if tipo == "rf" else "Regressão Logística Multinomial",
                       "candidato": i, "parametros": json.dumps(params), "f1_macro_media": float(v.mean()),
                       "f1_macro_desvio": float(v.std(ddof=1)), "segundos": round(time.time() - t0, 1)})
        print(f"  [{tipo}] {i:2d}/{len(candidatos)} F1 macro = {v.mean():.4f} ± {v.std(ddof=1):.4f}  {params}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    return linhas


def etapa_ajuste(X_tr, y_tr, X_te, y_te, cat, num, n_candidatos):
    """Busca aleatória de hiperparâmetros (3 partes, só no treino) e avaliação única no teste.

    O candidato "padrão" (os valores fixados a priori no artigo) entra na busca: assim a comparação
    com ele usa exatamente as mesmas partes da validação cruzada.
    """
    print("\n[ajuste] busca de hiperparâmetros (validação cruzada de 3 partes no treino)")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMENTE)

    padrao_rf = {"max_depth": PARAMS_RF["max_depth"], "min_samples_leaf": PARAMS_RF["min_samples_leaf"],
                 "max_features": "sqrt", "class_weight": PARAMS_RF["class_weight"]}
    sorteados = [p for p in ParameterSampler(GRADE_RF, n_iter=n_candidatos, random_state=SEMENTE)
                 if p != padrao_rf]
    linhas = buscar("rf", [padrao_rf] + sorteados, X_tr, y_tr, cat, num, cv)
    padrao_lr = {"C": 1.0}
    linhas += buscar("lr", [padrao_lr] + [{"C": c} for c in GRADE_C if c != 1.0], X_tr, y_tr, cat, num, cv)
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(PASTA_RESULTADOS / "ajuste_hiperparametros.csv", index=False, encoding="utf-8")

    # Melhor candidato de cada modelo = maior F1 macro médio na validação cruzada
    escolhidos, resultado = {}, []
    for nome, tipo, padrao in [("Random Forest", "rf", padrao_rf), ("Regressão Logística Multinomial", "lr", padrao_lr)]:
        sub = tabela[tabela["modelo"] == nome]
        melhor = sub.loc[sub["f1_macro_media"].idxmax()]
        ref = sub.iloc[0]  # o primeiro candidato é sempre o padrão
        escolhidos[tipo] = json.loads(melhor["parametros"])
        print(f"  melhor {nome}: {escolhidos[tipo]} (F1 macro CV {melhor['f1_macro_media']:.4f}; "
              f"padrão: {ref['f1_macro_media']:.4f})")
        # Avaliação única no teste: o modelo "a priori" e o ajustado, lado a lado
        for rotulo, params, f1_cv in [("a priori", padrao, ref["f1_macro_media"]),
                                      ("ajustado", escolhidos[tipo], melhor["f1_macro_media"])]:
            modelo = montar_modelo(tipo, cat, num, **params).fit(X_tr, y_tr)
            met = avaliar_modelo(modelo, X_te, y_te)
            resultado.append({"modelo": nome, "versao": rotulo, "parametros": json.dumps(params),
                              "f1_macro_cv3": f1_cv, **met})
            print(f"    {rotulo:9s} no teste: F1 macro={met['f1_macro']:.4f}  AUC={met['roc_auc_ovr_macro']:.4f}  "
                  f"revocação fatal={met['recall_fatal']:.3f}  precisão fatal={met['precisao_fatal']:.3f}", flush=True)
    salvar_csv(resultado, "modelos_ajustados.csv")
    (PASTA_RESULTADOS / "ajuste_hiperparametros.json").write_text(
        json.dumps(escolhidos, indent=2), encoding="utf-8")
    return escolhidos


# ----------------------------------------------------------------------------------------------
# Etapa 4: ajuste da regra de decisão (limiar)
# ----------------------------------------------------------------------------------------------
def prever_com_pesos(proba, ordem, pesos):
    """Classe prevista = a de maior (probabilidade x peso). Peso 1,0 em todas = decisão normal."""
    return np.array(ordem)[np.argmax(proba * np.array([pesos[c] for c in ordem]), axis=1)]


def etapa_limiar(X_tr, y_tr, X_te, y_te, cat, num, params_rf):
    """Escolhe pesos por classe que maximizam o F1 macro em previsões fora da amostra do TREINO.

    O Random Forest com class_weight="balanced" já equilibra as classes, mas o ponto de corte
    padrão (classe de maior probabilidade) não é necessariamente o que maximiza o F1 macro. Aqui os
    pesos são escolhidos nas previsões "fora da amostra" do treino (validação cruzada de 5 partes) e
    só depois aplicados uma vez ao teste.
    """
    print("\n[limiar] pesos por classe sobre as probabilidades do Random Forest ajustado")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    modelo = montar_modelo("rf", cat, num, **params_rf)
    # Probabilidades de cada sinistro do treino, previstas por modelos que NÃO o viram no treino
    proba_oof = cross_val_predict(modelo, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)
    ordem = sorted(CLASSES)  # ordem alfabética das colunas de proba (a mesma de classes_)
    f1_padrao = f1_score(y_tr, prever_com_pesos(proba_oof, ordem, {c: 1.0 for c in ordem}), average="macro")
    melhor = (f1_padrao, {c: 1.0 for c in ordem})
    for w_fatal in GRADE_PESOS:
        for w_sem in GRADE_PESOS:
            pesos = {"COM_FERIDOS": 1.0, "FATAL": w_fatal, "SEM_VITIMAS": w_sem}
            f1 = f1_score(y_tr, prever_com_pesos(proba_oof, ordem, pesos), average="macro")
            if f1 > melhor[0]:
                melhor = (f1, pesos)
    f1_oof, pesos = melhor
    print(f"  pesos escolhidos: {pesos}  (F1 macro fora da amostra: {f1_padrao:.4f} -> {f1_oof:.4f})")

    # Modelo final treinado em todo o treino e avaliado uma única vez no teste
    final = modelo.fit(X_tr, y_tr)
    proba_te = final.predict_proba(X_te)
    sem_ajuste = resumir(y_te, final.predict(X_te), proba_te, ordem)
    com_ajuste = resumir(y_te, prever_com_pesos(proba_te, ordem, pesos), proba_te, ordem)
    for rotulo, met in [("decisão padrão", sem_ajuste), ("com pesos", com_ajuste)]:
        print(f"    {rotulo:15s} no teste: F1 macro={met['f1_macro']:.4f}  acurácia={met['acuracia']:.3f}  "
              f"revocação fatal={met['recall_fatal']:.3f}  precisão fatal={met['precisao_fatal']:.3f}")
    # Curva de "ponto de operação" (apenas descritiva, nada é escolhido com ela): quanto mais peso
    # damos à classe FATAL, mais fatais o modelo encontra (revocação), com mais alertas falsos.
    curva = []
    for w_fatal in [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        pesos_c = {"COM_FERIDOS": 1.0, "FATAL": w_fatal, "SEM_VITIMAS": 1.0}
        met = resumir(y_te, prever_com_pesos(proba_te, ordem, pesos_c), proba_te, ordem)
        curva.append({"peso_fatal": w_fatal, **met})
        print(f"    peso FATAL {w_fatal:4.2f}: F1 macro={met['f1_macro']:.4f}  acurácia={met['acuracia']:.3f}  "
              f"revocação fatal={met['recall_fatal']:.3f}  precisão fatal={met['precisao_fatal']:.3f}")
    salvar_csv(curva, "ajuste_limiar_curva.csv")
    (PASTA_RESULTADOS / "ajuste_limiar.json").write_text(json.dumps({
        "parametros_rf": params_rf, "pesos": pesos,
        "f1_macro_fora_da_amostra_padrao": f1_padrao, "f1_macro_fora_da_amostra_com_pesos": f1_oof,
        "teste_decisao_padrao": sem_ajuste, "teste_com_pesos": com_ajuste,
        "segundos": round(time.time() - t0, 1)}, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapas", nargs="+", default=["temporal", "espacial", "ajuste", "limiar"],
                    choices=["temporal", "espacial", "ajuste", "limiar"], help="etapas a executar")
    ap.add_argument("--n-candidatos", type=int, default=9,
                    help="combinações sorteadas na busca do Random Forest (além do padrão)")
    args = ap.parse_args()
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    print("Carregando a base...")
    df = carregar_base()
    X, y, cat, num = criar_features(df)
    anos, uf = df["ano"], df["uf"]  # a base tem o ano de cada sinistro; ele NÃO é variável de entrada
    # A mesma divisão 80/20 (mesma semente) do treinamento principal, para os números serem comparáveis
    X_tr, X_te, y_tr, y_te = dividir_treino_teste(X, y, teste=0.2, semente=SEMENTE)

    if "temporal" in args.etapas:
        etapa_temporal(X, y, anos, cat, num)
    if "espacial" in args.etapas:
        etapa_espacial(X, y, uf, cat, num)
    escolhidos = None
    if "ajuste" in args.etapas:
        escolhidos = etapa_ajuste(X_tr, y_tr, X_te, y_te, cat, num, args.n_candidatos)
    if "limiar" in args.etapas:
        if escolhidos is None:  # etapa rodada sozinha: usa o ajuste salvo antes
            caminho = PASTA_RESULTADOS / "ajuste_hiperparametros.json"
            if not caminho.exists():
                raise SystemExit("Rode antes a etapa 'ajuste' (gera results/ajuste_hiperparametros.json).")
            escolhidos = json.loads(caminho.read_text(encoding="utf-8"))
        etapa_limiar(X_tr, y_tr, X_te, y_te, cat, num, escolhidos["rf"])
    print("\nPronto. Resultados em results/.")


if __name__ == "__main__":
    main()
