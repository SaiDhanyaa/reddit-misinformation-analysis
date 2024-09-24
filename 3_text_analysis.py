import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load the cleaned Reddit data from the CSV file where the posts are already cleaned
df = pd.read_csv('cleaned_reddit_posts.csv')

#####################################
# 1. Sentiment Analysis
#####################################

# Sentiment analysis helps to understand the emotional tone behind the text.
# We will use the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool from nltk.
# VADER provides sentiment polarity scores (positive, negative, neutral, and compound score).

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to calculate sentiment scores for the text
def get_sentiment(text):
    # 'compound' score gives an overall sentiment score, ranging from -1 (most negative) to +1 (most positive).
    return sid.polarity_scores(text)['compound']

# Apply the sentiment analysis function to the 'Cleaned_Body' column of the dataframe.
# This column contains the main text from Reddit posts.
df['Sentiment_Score'] = df['Cleaned_Body'].apply(lambda x: get_sentiment(str(x)))

# Assign sentiment labels based on the compound score:
# Positive if the score is above 0.05, Negative if the score is below -0.05, Neutral otherwise.
df['Sentiment_Label'] = df['Sentiment_Score'].apply(lambda score: 'positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral'))

# Show the first few rows of the updated dataframe with sentiment scores and labels.
print(df[['Title', 'Cleaned_Body', 'Sentiment_Score', 'Sentiment_Label']].head())

#####################################
# 2. Topic Modeling Using Latent Dirichlet Allocation (LDA)
#####################################

# Topic modeling helps in discovering abstract topics within a collection of text.
# LDA (Latent Dirichlet Allocation) is a popular unsupervised learning method to find hidden topics in the text.
# We'll use the Gensim library for this task.

import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize

# Tokenize the cleaned text: Break the 'Cleaned_Body' column (which contains text data) into individual tokens (words).
df['Tokens'] = df['Cleaned_Body'].apply(lambda x: word_tokenize(str(x)))

# Create a dictionary from the tokens.
# A dictionary assigns a unique id to each word.
dictionary = corpora.Dictionary(df['Tokens'])

# Create a corpus: a collection of word frequencies in each document (post).
# The corpus is a list of bag-of-words representations for each document.
corpus = [dictionary.doc2bow(tokens) for tokens in df['Tokens']]

# Train the LDA model to discover 5 topics in the corpus.
# 'num_topics' defines the number of topics we want the model to extract.
# 'passes' defines the number of iterations the model will run to converge on a solution.
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Display the top words in each of the 5 topics extracted by the model.
# The 'print_topics' function shows the words that are most representative of each topic.
print("\n--- Top words in each topic ---\n")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

#####################################
# 3. Keyword Extraction using TF-IDF
#####################################

# TF-IDF (Term Frequency-Inverse Document Frequency) is used to find the most important words in a document.
# It assigns higher importance to words that are frequent in a document but not too common across all documents.

from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF-IDF Vectorizer.
# 'max_features=10' means we want to extract the 10 most important words.
tfidf_vectorizer = TfidfVectorizer(max_features=10)

# Apply the TF-IDF vectorizer to the cleaned text data in the 'Cleaned_Body' column.
# This will convert the text into numerical values representing the importance of each word.
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Body'].dropna())

# Get the top 10 keywords based on TF-IDF scores.
# These keywords are considered the most important across all documents (posts).
keywords = tfidf_vectorizer.get_feature_names_out()
print(f"\nTop Keywords: {keywords}")
