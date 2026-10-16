# 🚦 Identificação de Trechos Críticos e Predição de Severidade de Sinistros Rodoviários

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow.svg)](#)

Repositório dedicado à pesquisa e desenvolvimento de modelos preditivos e análises geoespaciais de sinistros de trânsito em rodovias federais brasileiras. O projeto compara abordagens de aprendizado de máquina (*Random Forest Classifier*) e modelagem estatística clássica (Regressão de Poisson/Logística) para estimar a gravidade de acidentes e identificar trechos prioritários para intervenções de engenharia viária.

---

## 👥 Integrantes do Grupo

*   **André** 
*   **Karina** 
*   **Yago** 

---

## 📌 Objetivos da Pesquisa

- Analisar o perfil e a severidade dos sinistros em rodovias federais da PRF (2023–2025).
- Investigar a influência de características operacionais, de infraestrutura viária e condições meteorológicas no agravamento das vítimas.
- Aplicar o algoritmo **Random Forest** para ranquear a importância das variáveis (*Feature Importance*) associadas a traumas graves e fatais.
- Identificar padrões espaciais e pontos críticos (hotspots) para subsidiar tomadas de decisão em segurança viária.

---
## 📊 Fontes de Dados Utilizadas
**Polícia Rodoviária Federal (PRF):**

- Dados abertos de acidentes agrupados por ocorrência (2023, 2024 e 2025).

- Variáveis: temporalidade, traçado da via, tipo de pista, clima, causa presumida e severidade das vítimas.

**Infosiga-SP (Base Complementar):**

- Registros de acidentes e óbitos de trânsito no Estado de São Paulo.

- Utilizado para validação cruzada de dados de vítimas e calibração espacial em trechos paulistas.

---

## 💾 Acesso aos Dados Brutos (Google Drive)

Devido ao limite de tamanho de arquivos suportado pelo GitHub, os arquivos brutos (`.csv`) da PRF (2023–2025) e as tabelas complementares estão hospedados em uma pasta compartilhada do Google Drive.

*   🔗 **[Acessar Repositório de Dados no Google Drive](https://drive.google.com/drive/folders/1zCGJHtUm_F3HZ3kj2-8XtjhBuqwpa_8A?usp=sharing)**

### Instruções para reproduzir o pipeline:
1. Acesse o link acima e faça o download dos arquivos brutos da PRF.
2. Salve os arquivos `.csv` baixados diretamente no diretório local do projeto:
   ```text
   predicao-acidentes-prf/data/raw/
   ```

---

## 🤖 Treinamento dos modelos

Instale as dependências e rode a partir da **raiz do projeto**:

```bash
pip install -r requirements.txt
python -m src.train
```

O script (`src/train.py`) usa a base tratada em `data/processed/prf_2023_2025_limpo.csv`, divide em 80% treino / 20% teste (estratificado, `random_state=42`) e compara:

| Modelo | Papel |
|---|---|
| Baseline (classe mais comum) | referência mínima |
| Random Forest Classifier (`class_weight="balanced"`) | modelo principal do artigo |
| Regressão Logística Multinomial (`class_weight="balanced"`) | comparação com modelo estatístico clássico |

**Atenção a vazamento de dados:** as colunas `mortos`, `feridos_*`, `ilesos`, `pessoas`, `classificacao_acidente` e `indice_severidade` entregam a resposta e **não** são usadas como entrada.

Saídas geradas:

- `results/metricas.json` — acurácia, F1 (macro e weighted), ROC-AUC e F1/recall por classe
- `results/relatorio_classificacao.txt` — precisão, recall e F1 de cada classe
- `results/importancia_variaveis.csv` e `results/importancia_grupos.csv` — importância das variáveis
- `figures/06_matrizes_confusao.png`, `figures/07_importancia_grupos.png`, `figures/08_importancia_impureza.png`
- `data/outputs/random_forest.joblib` — modelo treinado (grande, não vai para o GitHub)

Opções: `--n-arvores 100` (mais rápido) e `--incluir-causa-tipo` (experimento extra com causa e tipo do acidente, que **não** faz parte do artigo).
