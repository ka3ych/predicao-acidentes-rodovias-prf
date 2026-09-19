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