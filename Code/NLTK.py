import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Lettura del dataset
df = pd.read_csv('../Dataset/datasetTweetCovid.csv')

# Definizione del SentimentIntensityAnalyzer di NLTK
sia = SentimentIntensityAnalyzer()

# Prepara il file di output
output_file = open('../Risultati/Risultati - NLTK.txt', 'w', encoding="utf-8")

# Chiede all'utente d'inserire una parola da cercare
search_word = input("Inserisci una parola da cercare: ").lower()

# Crea un subset del dataframe contenente solo le righe che contengono la parola cercata
subset_df = df[df['tweet'].str.contains(search_word)]

# Analizza il sentiment di ogni frase che contiene la parola cercata
positive_count = 0
negative_count = 0
neutral_count = 0
sentiments = []
output_file.write(f"Sentiment per le frasi contenenti '{search_word}':\n\n")
for i, row in subset_df.iterrows():
    tweet = row['tweet'].lower()
    sentiment = sia.polarity_scores(tweet)['compound']
    sentiments.append(sentiment)
    output_file.write(f"Sentiment: {sentiment:.2f} - {tweet}\n\n")
    if sentiment > 0.2:
        positive_count += 1
    elif sentiment < -0.2:
        negative_count += 1
    else:
        neutral_count += 1

# Calcola la media dei sentiment delle frasi contenenti la parola cercata
if len(sentiments) > 0:
    avg_sentiment = sum(sentiments) / len(sentiments)
    output_file.write(f"Media del Sentiment per le frasi contenenti '{search_word}': {avg_sentiment:.2f}\n")
    output_file.write(f"Frasi positive: {positive_count}\n")
    output_file.write(f"Frasi negative: {negative_count}\n")
    output_file.write(f"Frasi neutre: {neutral_count}\n")
else:
    output_file.write(f"Nessuna frase trovata contenente la parola '{search_word}'")

# Chiude il file di output
output_file.close()