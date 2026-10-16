"""Carregamento da base tratada da PRF (2023-2025)."""

from pathlib import Path

import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CAMINHO_BASE_TRATADA = RAIZ_PROJETO / "data" / "processed" / "prf_2023_2025_limpo.csv"


def carregar_base(caminho: Path = CAMINHO_BASE_TRATADA) -> pd.DataFrame:
    """Lê a base já limpa gerada pelo notebook 01 (separador ';')."""
    if not Path(caminho).exists():
        raise FileNotFoundError(
            f"Base não encontrada em {caminho}. "
            "Rode o notebook 01_coleta_e_limpeza.ipynb ou baixe a base tratada."
        )
    return pd.read_csv(caminho, sep=";", low_memory=False)
