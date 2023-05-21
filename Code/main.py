# Progetto per il Corso di Semantic Intelligence
# Campopiano Daniele - 174624
# Salvatore Carmine - 174337

# Dato un Dataset (presente nella root del progetto)
# Effettuiamo Sentiment e Content Analysis

import pandas as pd
import warnings
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob

# Librerie per il funzionamento della Regressione Logistica per la Sentiment Analysis
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.feature_extraction.text import CountVectorizer

# Queste righe possono essere commentata/decommentata in base a se le librerie indicate sono installate o meno
# import nltk
# spacy.cli.download("en_core_web_sm")
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
# nltk.download('punkt')

# Non stampa i FutureWarning nella console
warnings.simplefilter(action='ignore', category=FutureWarning)

# Carica il Dataset presente nella root del progetto
df = pd.read_csv('../Dataset/datasetTweetCovid.csv')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Sentiment Analysis: Calcola la Media generale del Sentiment dei tweet,
# il sentiment per ogni tweet, il numero di Tweet con sentiment positivo/neutro/negativo

# Rimuovi gli hashtag, i link e i simboli dal testo del tweet
df['clean_tweet'] = df['tweet'].str.replace('#', '').str.replace('http\S+|www.\S+', '', case=False)

# Tokenizza le parole del testo del tweet
df['tokens'] = df['clean_tweet'].apply(word_tokenize)

# Rimuovi le parole di stop dal testo del tweet
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Analizza il sentimento dei tweet utilizzando l'analisi delle polarità di NLTK
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['clean_tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Calcola la media del sentimento dei tweet
sentiment_mean = df['sentiment'].mean()

# Aggiungi una colonna per il sentimento
df['sentiment'] = None

# Effettua l'analisi del sentimento per ogni frase
for i, row in df.iterrows():
    text = row['tweet']                    # il testo deve essere contenuto in una colonna chiamata 'tweet'
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    df.at[i, 'sentiment'] = sentiment

# Esegue un'analisi di sentiment sui tweet
positive_tweets = 0
negative_tweets = 0
neutral_tweets = 0
sentiment_scores = []
analyzer = SentimentIntensityAnalyzer()

for tweet in df['clean_tweet']:
    sentiment = analyzer.polarity_scores(tweet)
    sentiment_scores.append(sentiment['compound'])
    if sentiment['compound'] > 0:
        positive_tweets += 1
    elif sentiment['compound'] < 0:
        negative_tweets += 1
    else:
        neutral_tweets += 1

# Salva i risultati su un file di testo
with open('../Risultati/Risultati - Sentiment Analysis.txt', 'w', encoding='utf-8') as f:
        f.write(f"La media del Sentiment Analysis dei tweet è: {sentiment_mean:.2f}\n")
        f.write(f"\nDi {positive_tweets + neutral_tweets + negative_tweets} tweets analizzati abbiamo:\n")
        f.write(f"Tweet Positivi: {positive_tweets}\n")
        f.write(f"Tweet Neutrali: {neutral_tweets}\n")
        f.write(f"Tweet Negativi: {negative_tweets}\n")
        f.write(f"\nI sentiment per ogni tweet sono i seguenti:\n")
        for i, row in df.iterrows():
            text = row['tweet']
            sentiment = row['sentiment']
            f.write(f'Testo: {text}\nSentimento: {sentiment}\n')

# Stampa un messaggio di conferma del salvataggio su file
print("I risultati inerenti la Sentiment Analysis sono stati salvati sul file 'Risultati - Sentiment Analysis.txt'")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Content Analysis: Trova le 50 parole più frequenti ed Identifica frasi e entità nomi nei tweet

# Tokenizza i tweet
tokens = [word_tokenize(tweet) for tweet in df['tweet']]

# Rimuove le stop words
stopWords = set(stopwords.words('english'))
filtered_tokens = [[word for word in tweet if word.lower() not in stopWords] for tweet in tokens]

# Calcola la frequenza di ogni parola
word_counts = Counter([word for tweet in filtered_tokens for word in tweet])

# Salva le 50 parole più frequenti
topWords = word_counts.most_common(50)

# Carica il modello di lingua inglese di spacy
nlp = spacy.load('en_core_web_sm')

# Analizza la sintassi dei tweet
df['doc'] = df['tweet'].apply(nlp)

# I set a differenza delle liste ci permettono di non avere duplicati
sentences = set()
entities = set()

# Estrai le frasi e le entità nomi dai tweet
for doc in df['doc']:
    for sent in doc.sents:
        sentences.add(sent.text)
    for ent in doc.ents:
        entities.add(ent.text)

# Salva i risultati su un file di testo
with open('../Risultati/Risultati - Content Analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Risultati Content Analysis: 50 Parole più utilizzate\n\n")
    for word, count in topWords:
        f.write(f"{word}\t{count}\n")
    f.write(f"\nRisultati Content Analysis - Identificazione Frasi:\n")
    for sent in sentences:
        f.write(sent + '\n')
    f.write(f"\nRisultati Content Analysis - Identificazione Entità Nomi:\n")
    for entity in entities:
        f.write(entity + '\n')
    f.write(f"\nRisultati Content Analysis: {len(sentences)} frasi identificate")
    f.write(f"\nRisultati Content Analysis: {len(entities)} entità nomi identificate")

# Stampa un messaggio di conferma del salvataggio su file
print("I risultati inerenti la Content Analysis sono stati salvati su file 'Risultati - Content Analysis.txt'")