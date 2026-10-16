"""Carregamento da base tratada da PRF (2023-2025)."""

from pathlib import Path

import pandas as pd

# Raiz do projeto = pasta acima de src/. Usar caminhos relativos a ela faz o código
# funcionar em qualquer computador, não importa onde o projeto foi salvo.
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_BASE_TRATADA = RAIZ_PROJETO / "data" / "processed" / "prf_2023_2025_limpo.csv"


def carregar_base(caminho: Path = CAMINHO_BASE_TRATADA) -> pd.DataFrame:
    """Lê a base já limpa gerada pelo notebook 01 (separador ';')."""
    # Avisa com uma mensagem clara se o CSV não estiver na pasta (ele não vai para o GitHub)
    if not Path(caminho).exists():
        raise FileNotFoundError(
            f"Base não encontrada em {caminho}. "
            "Rode o notebook 01_coleta_e_limpeza.ipynb ou baixe a base tratada."
        )
    # sep=";" porque o CSV usa ponto e vírgula; low_memory=False evita aviso de tipos mistos
    return pd.read_csv(caminho, sep=";", low_memory=False)
