# **Análise de Sentimentos em Tweets**

Este projeto realiza uma análise de sentimentos em tweets categorizados em **positivo**, **negativo**, e **neutro**. Inclui etapas de limpeza de texto, visualização de dados e, futuramente, um modelo preditivo de aprendizado de máquina.

---

## **🎯 Objetivo**

O objetivo deste projeto é analisar tweets coletados para identificar padrões de sentimentos, bem como explorar características textuais como o tamanho dos tweets e as palavras mais frequentes por categoria.

---

## **📂 Estrutura do Projeto**

- **`codigo.py`**: Script principal com todas as análises e visualizações.
- **`tweets_processados.csv`**: Arquivo gerado com os tweets limpos após o processamento.
- **`twitter_training.csv`**: Dataset original contendo os tweets e rótulos de sentimentos.

---

## **🚀 Funcionalidades Implementadas**

1. **Limpeza de Dados:**
   - Remoção de URLs, caracteres especiais e espaços extras.
   - Conversão do texto para minúsculas.

2. **Visualizações:**
   - Gráfico de barras da distribuição de sentimentos.
   - Gráfico da distribuição do tamanho dos tweets.
   - Nuvens de palavras para cada categoria de sentimento.

3. **Análise Exploratória:**
   - Comparação do número de palavras entre diferentes sentimentos.
   - Identificação de tweets mais longos para cada categoria.

---


## **📈 Futuras Implementações**

- Treinamento de um modelo de aprendizado de máquina para prever sentimentos.
- Dashboard interativo com **Streamlit** para análise em tempo real.
- Análise temporal dos tweets (se datas estiverem disponíveis).
- Implementação de um modelo avançado baseado em **BERT** ou outro transformer.

---

## **📦 Como Rodar o Projeto**

### **1. Pré-requisitos**

Certifique-se de ter o Python instalado. Recomenda-se usar um ambiente virtual.

### **2. Instalar Dependências**
```bash
pip install -r requirements.txt
