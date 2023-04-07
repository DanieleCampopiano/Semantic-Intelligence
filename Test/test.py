import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Lettura del dataset
df = pd.read_csv('nuovoDataset.csv')

# Definizione del SentimentIntensityAnalyzer di NLTK
sia = SentimentIntensityAnalyzer()

# Calcolo del Sentiment di tutti i tweet
df['sentiment'] = df['tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Seleziona i 20 tweet con Sentiment maggiore e minore
top_tweets = df.nlargest(20, 'sentiment')
bottom_tweets = df.nsmallest(20, 'sentiment')

# Prepara il file di output
output_file = open('output2.txt', 'w', encoding="utf-8")

# Analizza i 20 tweet con Sentiment maggiore
output_file.write("TOP TWEETS:\n\n")
for i, row in top_tweets.iterrows():
    tweet = row['tweet']
    sentiment = row['sentiment']
    output_file.write(f"Sentiment: {sentiment:.2f} - {tweet}\n")

    # Rimuove le stop word
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in tweet.split() if word.lower() not in stop_words]

    # Analizza il sentiment di ogni parola
    for word in words:
        word_sentiment = sia.polarity_scores(word)['compound']
        output_file.write(f"{word} ~ Sentiment: {word_sentiment:.2f}\n")

    output_file.write('\n')

# Analizza i 20 tweet con Sentiment minore
output_file.write("BOTTOM TWEETS: - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")
for i, row in bottom_tweets.iterrows():
    tweet = row['tweet']
    sentiment = row['sentiment']
    output_file.write(f"Sentiment: {sentiment:.2f} - {tweet}\n")

    # Rimuove le stop word
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in tweet.split() if word.lower() not in stop_words]

    # Analizza il sentiment di ogni parola
    for word in words:
        word_sentiment = sia.polarity_scores(word)['compound']
        output_file.write(f"{word} ~ Sentiment: {word_sentiment:.2f}\n")

    output_file.write('\n')

# Chiude il file di output
output_file.close()