import pandas as pd
from textblob import TextBlob

# Lettura del dataset
df = pd.read_csv('../Dataset/datasetTweetCovid.csv')

# Prepara il file di output
output_file = open('../Risultati/Risultati - TextBlob.txt', 'w', encoding="utf-8")

# Chiede all'utente d'inserire una parola da cercare
search_word = input("Inserisci una parola da cercare: ").lower()

# Crea un subset del dataframe contenente solo le righe che contengono la parola cercata
subset_df = df[df['tweet'].str.contains(search_word)]

# Analizza il sentiment di ogni frase che contiene la parola cercata
sentiments = []
output_file.write(f"Sentiment per le frasi contenenti '{search_word}':\n\n")
for i, row in subset_df.iterrows():
    tweet = row['tweet'].lower()
    sentiment = TextBlob(tweet).sentiment.polarity
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