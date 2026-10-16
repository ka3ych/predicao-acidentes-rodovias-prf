"""Treina e avalia os modelos descritos na metodologia do artigo.

Uso (a partir da raiz do projeto):
    python -m src.train                       # configuração do artigo
    python -m src.train --incluir-causa-tipo  # experimento extra com causa e tipo do acidente
    python -m src.train --n-arvores 100       # mais rápido, para testar

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
from sklearn.pipeline import Pipeline

from src.data_loader import RAIZ_PROJETO, carregar_base
from src.preprocessing import (CLASSES, criar_features, dividir_treino_teste,
                               grupos_de_variaveis, preprocessador_arvores, preprocessador_linear)
from src.visualization import PASTA_FIGURAS, plotar_importancia, plotar_matrizes_confusao

SEMENTE = 42
PASTA_RESULTADOS = RAIZ_PROJETO / "results"
PASTA_SAIDAS = RAIZ_PROJETO / "data" / "outputs"  # ignorada pelo git (arquivos grandes)


def avaliar(nome, modelo, X_teste, y_teste):
    """Calcula as métricas da seção 3.5 do artigo."""
    previsto = modelo.predict(X_teste)
    proba = modelo.predict_proba(X_teste)
    ordem = list(modelo.classes_)  # ordem alfabética, a mesma das colunas de proba
    rel_dict = classification_report(y_teste, previsto, labels=CLASSES, output_dict=True, zero_division=0)
    metricas = {
        "modelo": nome,
        "acuracia": accuracy_score(y_teste, previsto),
        "f1_macro": f1_score(y_teste, previsto, average="macro"),
        "f1_weighted": f1_score(y_teste, previsto, average="weighted"),
        "roc_auc_ovr_macro": roc_auc_score(y_teste, proba, multi_class="ovr",
                                           average="macro", labels=ordem),
        "f1_por_classe": dict(zip(CLASSES, f1_score(y_teste, previsto, labels=CLASSES, average=None))),
        "recall_por_classe": {c: rel_dict[c]["recall"] for c in CLASSES},
    }
    cm = confusion_matrix(y_teste, previsto, labels=CLASSES)
    relatorio = classification_report(y_teste, previsto, labels=CLASSES, digits=3, zero_division=0)
    return metricas, cm, relatorio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incluir-causa-tipo", action="store_true",
                    help="inclui causa_acidente e tipo_acidente como variáveis (fora do artigo)")
    ap.add_argument("--n-arvores", type=int, default=200, help="árvores do Random Forest")
    ap.add_argument("--amostra-permutacao", type=int, default=20000,
                    help="linhas do teste usadas na importância por permutação")
    args = ap.parse_args()

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)
    sufixo = "_com_causa_tipo" if args.incluir_causa_tipo else ""

    print("Carregando a base...")
    df = carregar_base()
    X, y, cat, num = criar_features(df, incluir_causa_tipo=args.incluir_causa_tipo)
    X_tr, X_te, y_tr, y_te = dividir_treino_teste(X, y, teste=0.2, semente=SEMENTE)
    print(f"Variáveis: {X.shape[1]} | treino: {len(X_tr):,} | teste: {len(X_te):,}".replace(",", "."))

    modelos = {
        "Baseline (classe mais comum)": Pipeline([
            ("pre", preprocessador_arvores(cat, num)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessador_arvores(cat, num)),
            ("clf", RandomForestClassifier(
                n_estimators=args.n_arvores, max_depth=20, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=SEMENTE)),
        ]),
        "Regressão Logística Multinomial": Pipeline([
            ("pre", preprocessador_linear(cat, num)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]),
    }

    todas, matrizes, relatorios = [], {}, []
    for nome, modelo in modelos.items():
        t0 = time.time()
        modelo.fit(X_tr, y_tr)
        met, cm, rel = avaliar(nome, modelo, X_te, y_te)
        met["segundos_treino_e_avaliacao"] = round(time.time() - t0, 1)
        todas.append(met)
        relatorios.append(f"=== {nome} ===\n{rel}")
        if nome != "Baseline (classe mais comum)":
            matrizes[nome] = cm
        print(f"{nome:34s} acurácia={met['acuracia']:.3f}  F1 macro={met['f1_macro']:.3f}  "
              f"F1 weighted={met['f1_weighted']:.3f}  AUC={met['roc_auc_ovr_macro']:.3f}")

    # --- importância das variáveis (Random Forest) ---
    rf = modelos["Random Forest"]

    # (a) Feature Importance clássica do Random Forest (por impureza), uma linha por variável.
    #     Atenção: tende a favorecer variáveis contínuas ou com muitas categorias (lat, lon, mês).
    nomes = list(rf.named_steps["pre"].get_feature_names_out())
    tabela = pd.DataFrame(
        {"importancia_impureza": rf.named_steps["clf"].feature_importances_}, index=nomes
    ).sort_values("importancia_impureza", ascending=False)

    # (b) Importância por permutação, por GRUPO de variáveis: quanto o F1 macro cai no teste
    #     quando embaralhamos o grupo inteiro (mais confiável quando há variáveis redundantes).
    print("Calculando importância por permutação de grupos...")
    grupos = grupos_de_variaveis(list(X.columns))
    amostra = X_te.sample(min(args.amostra_permutacao, len(X_te)), random_state=SEMENTE)
    y_amostra = y_te.loc[amostra.index]
    base = f1_score(y_amostra, rf.predict(amostra), average="macro")
    rng = np.random.default_rng(SEMENTE)
    quedas = {}
    for nome_grupo, cols in grupos.items():
        repeticoes = []
        for _ in range(3):
            embaralhada = amostra.copy()
            ordem = rng.permutation(len(amostra))
            embaralhada[cols] = amostra[cols].to_numpy()[ordem]  # o grupo todo é embaralhado junto
            repeticoes.append(base - f1_score(y_amostra, rf.predict(embaralhada), average="macro"))
        quedas[nome_grupo] = float(np.mean(repeticoes))
    tabela_grupos = pd.DataFrame({"queda_f1_macro": quedas}).sort_values("queda_f1_macro", ascending=False)
    print(tabela_grupos.to_string(float_format=lambda v: f"{v:.4f}"))

    # --- salvar tudo ---
    (PASTA_RESULTADOS / f"metricas{sufixo}.json").write_text(
        json.dumps(todas, indent=2, ensure_ascii=False), encoding="utf-8")
    (PASTA_RESULTADOS / f"relatorio_classificacao{sufixo}.txt").write_text(
        "\n\n".join(relatorios), encoding="utf-8")
    tabela.to_csv(PASTA_RESULTADOS / f"importancia_variaveis{sufixo}.csv", encoding="utf-8")
    tabela_grupos.to_csv(PASTA_RESULTADOS / f"importancia_grupos{sufixo}.csv", encoding="utf-8")
    plotar_matrizes_confusao(matrizes, CLASSES, PASTA_FIGURAS / f"06_matrizes_confusao{sufixo}.png")
    plotar_importancia(tabela_grupos, "queda_f1_macro",
                       "Importância por grupo (queda do F1 macro ao embaralhar o grupo)",
                       PASTA_FIGURAS / f"07_importancia_grupos{sufixo}.png")
    plotar_importancia(tabela, "importancia_impureza",
                       "Importância por impureza (Feature Importance do Random Forest)",
                       PASTA_FIGURAS / f"08_importancia_impureza{sufixo}.png")
    joblib.dump(rf, PASTA_SAIDAS / f"random_forest{sufixo}.joblib", compress=3)
    print("Pronto. Resultados em results/ e figures/.")


if __name__ == "__main__":
    main()
