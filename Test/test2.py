import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Lettura del dataset
df = pd.read_csv('nuovoDataset.csv')

# Definizione del SentimentIntensityAnalyzer di NLTK
sia = SentimentIntensityAnalyzer()

# Prepara il file di output
output_file = open('Carmine.txt', 'w', encoding="utf-8")

# Chiede all'utente di inserire una parola da cercare
search_word = input("Inserisci una parola da cercare: ")

# Crea un subset del dataframe contenente solo le righe che contengono la parola cercata
subset_df = df[df['tweet'].str.contains(search_word)]

# Analizza il sentiment di ogni frase che contiene la parola cercata
sentiments = []
output_file.write(f"Sentiment per le frasi contenenti '{search_word}':\n\n")
for i, row in subset_df.iterrows():
    tweet = row['tweet']
    sentiment = sia.polarity_scores(tweet)['compound']
    sentiments.append(sentiment)
    output_file.write(f"Sentiment: {sentiment:.2f} - {tweet}\n\n")

# Calcola la media dei sentiment delle frasi contenenti la parola cercata
if len(sentiments) > 0:
    avg_sentiment = sum(sentiments) / len(sentiments)
    output_file.write(f"Media del Sentiment per le frasi contenenti '{search_word}': {avg_sentiment:.2f}")
else:
    output_file.write(f"Nessuna frase trovata contenente la parola '{search_word}'")

# Chiude il file di output
output_file.close()
