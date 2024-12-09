import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
dados = pd.read_csv("/home/ianlee/Área de Trabalho/AnaliseSentimentos/twitter_training.csv")
dados.columns = ['id', 'category', 'sentiment', 'tweet']
dados.dropna(subset=['tweet'], inplace=True)

# Função para limpar o texto
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

dados['tweet_limpo'] = dados['tweet'].apply(limpar_texto)
# Gráfico da distribuição de sentimentos
sns.countplot(x='sentiment', data=dados, hue='sentiment', palette='coolwarm', dodge=False)
plt.title('Distribuição de Sentimentos')
plt.legend([],[], frameon=False)
plt.show()

# Adicionar coluna com o tamanho dos tweets
dados['tamanho_tweet'] = dados['tweet'].apply(len)

# Gráfico da distribuição do tamanho dos tweets
sns.histplot(dados['tamanho_tweet'], kde=True, color='blue', bins=30)
plt.title('Distribuição do Tamanho dos Tweets')
plt.xlabel('Tamanho do Tweet')
plt.ylabel('Frequência')
plt.show()
from wordcloud import WordCloud

# Criar a nuvem de palavras
texto = " ".join(dados['tweet_limpo'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Frequentes nos Tweets')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(dados['tweet_limpo'], dados['sentiment'], test_size=0.2, random_state=42)

# Vetorização
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Modelo
modelo = LogisticRegression()
modelo.fit(X_train_tfidf, y_train)

# Avaliação
y_pred = modelo.predict(X_test_tfidf)

from collections import Counter
from wordcloud import WordCloud

# Criar uma função para gerar frequência de palavras por sentimento
def gerar_nuvem(sentimento):
    tweets = dados[dados['sentiment'] == sentimento]['tweet_limpo']
    palavras = " ".join(tweets)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(palavras)

    # Exibir a nuvem de palavras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Nuvem de Palavras - Sentimento: {sentimento}')
    plt.show()

# Gerar nuvens de palavras para cada sentimento
for sentimento in dados['sentiment'].unique():
    gerar_nuvem(sentimento)

	# Adicionar coluna com o número de palavras em cada tweet
dados['num_palavras'] = dados['tweet_limpo'].apply(lambda x: len(x.split()))

# Boxplot para comparar o número de palavras por sentimento
sns.boxplot(x='sentiment', y='num_palavras', data=dados, palette='coolwarm')
plt.title('Comparação do Número de Palavras por Sentimento')
plt.xlabel('Sentimento')
plt.ylabel('Número de Palavras')
plt.show()






