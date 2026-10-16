"""Figuras dos resultados do treinamento (matriz de confusão e importância)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela
import matplotlib.pyplot as plt
import numpy as np

COR = "#2a5d8f"  # um único tom de azul: as figuras mostram magnitude
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
PASTA_FIGURAS = RAIZ_PROJETO / "figures"


def plotar_matrizes_confusao(matrizes: dict, classes, caminho=None):
    """matrizes: {nome_do_modelo: matriz (linhas = real, colunas = previsto)}.

    Cada linha é normalizada (soma 100%), então a diagonal é a revocação (recall) da classe.
    """
    caminho = Path(caminho or PASTA_FIGURAS / "06_matrizes_confusao.png")
    fig, eixos = plt.subplots(1, len(matrizes), figsize=(6 * len(matrizes), 5), squeeze=False)
    for ax, (nome, cm) in zip(eixos[0], matrizes.items()):
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(classes)):
            for j in range(len(classes)):
                cor = "white" if pct[i, j] > 55 else "#222222"
                ax.text(j, i, f"{pct[i, j]:.1f}%\n({cm[i, j]:,})".replace(",", "."),
                        ha="center", va="center", color=cor, fontsize=9)
        ax.set_xticks(range(len(classes)), classes, rotation=20)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        ax.set_title(nome, fontsize=11, fontweight="bold")
    fig.suptitle("Matriz de confusão no conjunto de teste (% por classe real)", fontsize=12)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def plotar_importancia(tabela, coluna, titulo, caminho, topo=15):
    """Barras horizontais das variáveis mais importantes (tabela: DataFrame indexado pela variável)."""
    dados = tabela[coluna].sort_values(ascending=False).head(topo)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(dados) + 1.5))
    ax.barh(dados.index, dados.values, color=COR, height=0.6)
    ax.set_title(titulo, fontsize=11, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return Path(caminho)
