import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the cleaned Reddit data
# The CSV file 'cleaned_reddit_posts.csv' contains the cleaned Reddit posts after preprocessing (e.g., removing stopwords, punctuation).
# We're loading this data into a Pandas DataFrame for further analysis.
df = pd.read_csv('cleaned_reddit_posts.csv')

# Concatenate all cleaned text from the 'Cleaned_Body' column into a single string
# 'Cleaned_Body' contains the preprocessed text for each Reddit post (no punctuation, lowercased, etc.).
# We combine all the text from this column into one large string, so we can analyze word frequencies across all posts.
all_words = ' '.join(df['Cleaned_Body'].dropna())  # dropna() ensures there are no missing values (NaN)

# Split the text into individual words
# This step breaks the large string of text into individual words, which we'll analyze next.
word_list = all_words.split()  # Splitting by whitespace creates a list of all the words

# Count word frequencies using the Counter class
# We use Python's Counter from the 'collections' module to count the occurrence of each word in the word_list.
# Counter generates a dictionary-like structure where keys are words and values are the frequencies.
word_counts = Counter(word_list)

# Convert the word frequencies to a DataFrame for easy manipulation
# We take the top 20 most common words using the most_common(20) method and create a DataFrame.
# This makes it easier to manipulate the data and visualize it.
word_freq_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

# Bar Chart: Visualize the top 20 most frequent words
# We create a bar chart to visualize the top 20 most frequent words and their corresponding frequencies.
plt.figure(figsize=(10, 6))  # Set the figure size for the plot
plt.barh(word_freq_df['Word'], word_freq_df['Frequency'], color='skyblue')  # Horizontal bar chart with sky blue color
plt.xlabel('Frequency')  # Label for the x-axis (number of occurrences of each word)
plt.title('Top 20 Most Frequent Words in Reddit Posts')  # Title of the plot
plt.gca().invert_yaxis()  # Invert y-axis so the most frequent words are at the top
plt.show()  # Display the plot

# Generate a Word Cloud
# A word cloud is a visual representation of text data where more frequent words appear larger.
# We generate a word cloud from the concatenated text ('all_words') using the WordCloud library.
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Display the Word Cloud
# We display the word cloud as an image using Matplotlib.
plt.figure(figsize=(10, 6))  # Set the figure size for the word cloud
plt.imshow(wordcloud, interpolation='bilinear')  # 'bilinear' interpolation for smooth display of the word cloud
plt.axis('off')  # Remove axis labels for a cleaner display
plt.show()  # Display the word cloud
