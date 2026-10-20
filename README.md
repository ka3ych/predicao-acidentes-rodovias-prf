# 🚦 Identificação de Trechos Críticos e Predição de Severidade de Sinistros Rodoviários

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow.svg)](#)

Repositório dedicado à pesquisa e desenvolvimento de modelos preditivos e análises geoespaciais de sinistros de trânsito em rodovias federais brasileiras. O projeto compara abordagens de aprendizado de máquina (*Random Forest Classifier*) e modelagem estatística clássica (Regressão Logística Multinomial) para estimar a gravidade de acidentes e apontar os fatores de risco associados, subsidiando a priorização de intervenções de engenharia viária.

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

Rode tudo a partir da **raiz do projeto**. Recomendamos usar um ambiente virtual (`.venv`), que isola as bibliotecas deste projeto das do resto do computador.

**1. Criar e ativar o ambiente virtual** (só na primeira vez o `venv`; a ativação é a cada novo terminal):

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (Prompt de Comando)
python -m venv .venv
.venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Quando o ambiente estiver ativo, o terminal mostra `(.venv)` no início da linha. No VS Code, também é possível usar `Ctrl+Shift+P` → **Python: Create Environment** → **Venv** e marcar o `requirements.txt`.

> No PowerShell, se aparecer erro de execução de scripts, rode `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de novo.

**2. Instalar as dependências e treinar:**

```bash
pip install -r requirements.txt
python -m src.train
```

Se a instalação do `geopandas` falhar no Windows, ele só é necessário para os notebooks de mapas; para o treino bastam `pandas numpy matplotlib scikit-learn`.

O script (`src/train.py`) usa a base tratada em `data/processed/prf_2023_2025_limpo.csv`, divide em 80% treino / 20% teste (estratificado, `random_state=42`) e compara:

| Modelo | Papel |
|---|---|
| Baseline (classe mais comum) | referência mínima |
| Random Forest Classifier (`class_weight="balanced"`) | modelo principal do artigo |
| Regressão Logística Multinomial (`class_weight="balanced"`) | comparação com modelo estatístico clássico |

Depois da avaliação no teste, o script repete a comparação com **validação cruzada estratificada de 5 partes** nos dados de treino, para mostrar o quanto o resultado varia entre divisões diferentes (o teste continua reservado e os números dele não mudam).

**Atenção a vazamento de dados:** as colunas `mortos`, `feridos_*`, `ilesos`, `pessoas`, `classificacao_acidente` e `indice_severidade` entregam a resposta e **não** são usadas como entrada.

Saídas geradas:

- `results/metricas.json` — acurácia, F1 (macro e weighted), ROC-AUC e F1/recall por classe
- `results/relatorio_classificacao.txt` — precisão, recall e F1 de cada classe
- `results/validacao_cruzada.csv` — média e desvio padrão de acurácia, F1 macro e ROC-AUC na validação cruzada (e o F1 macro de cada parte)
- `results/importancia_variaveis.csv` e `results/importancia_grupos.csv` — importância das variáveis
- `figures/06_matrizes_confusao.png`, `figures/07_importancia_grupos.png`, `figures/08_importancia_impureza.png`
- `data/outputs/random_forest.joblib` — modelo treinado (grande, não vai para o GitHub)

Opções: `--n-arvores 100` (mais rápido), `--folds 0` (pula a validação cruzada, que leva alguns minutos) e `--incluir-causa-tipo` (experimento extra com causa e tipo do acidente, que **não** faz parte do artigo).

### Experimentos complementares

O script `src/experimentos.py` responde às limitações discutidas no artigo e **não altera** os resultados principais acima (usa a mesma divisão 80/20 e a mesma semente):

```bash
python -m src.experimentos                           # as quatro etapas (cerca de 20 minutos)
python -m src.experimentos --etapas temporal ajuste  # só algumas etapas
python -m src.experimentos --n-candidatos 6          # busca menor, para testar mais rápido
```

| Etapa | O que faz | Saída em `results/` |
|---|---|---|
| `temporal` | treina com 2023–2024 e testa em 2025 | `validacao_temporal.csv` |
| `espacial` | 5 rodadas deixando estados inteiros fora do treino | `validacao_espacial.csv`, `validacao_espacial_ufs.txt` |
| `ajuste` | busca de hiperparâmetros (Random Forest e Regressão Logística) com validação cruzada de 3 partes **só no treino**; o teste avalia só o resultado final | `ajuste_hiperparametros.csv` e `.json`, `modelos_ajustados.csv` |
| `limiar` | pesos por classe sobre as probabilidades do Random Forest ajustado, escolhidos com previsões fora da amostra do treino, e curva de ponto de operação (revocação × precisão dos fatais) | `ajuste_limiar.json`, `ajuste_limiar_curva.csv` |

Principais números (F1 macro): no teste aleatório, Random Forest 0,410 e Regressão Logística 0,375; treinando em 2023–2024 e testando em 2025, 0,408 e 0,373; com estados inteiros fora do treino, 0,364 e 0,358 (a vantagem do Random Forest praticamente desaparece). O ajuste de hiperparâmetros eleva o Random Forest a 0,428 no teste, mas principalmente por deslocar o ponto de operação (menos fatais detectados e menos alarmes falsos); a ROC-AUC quase não muda (0,658 → 0,660).
