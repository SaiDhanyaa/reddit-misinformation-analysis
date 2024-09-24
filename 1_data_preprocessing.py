import pandas as pd
import re # Regular expressions for removing unwanted text patterns
import string  # String module for removing punctuation
from nltk.corpus import stopwords  # To remove common English stopwords
from nltk.tokenize import word_tokenize  # Tokenizes text into words

# Function to clean the text by removing unwanted elements (URLs, punctuation, stopwords, etc.)
def clean_text(text):
    # Remove URLs from the text (any sequence starting with 'http')
    text = re.sub(r'http\S+', '', text)
    
    # Remove any text inside brackets [ ... ] since it may not contribute to analysis
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove numbers from the text (words that contain digits are removed)
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Remove punctuation using Python's string library (e.g., !, ?, ., etc.)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert the text to lowercase to ensure consistency (e.g., 'The' and 'the' are treated the same)
    text = text.lower()
    
    # Tokenize the cleaned text, splitting it into individual words (tokens)
    tokens = word_tokenize(text)
    
    # Create a set of English stop words (e.g., 'and', 'the', 'is') to remove common but uninformative words
    stop_words = set(stopwords.words('english'))
    
    # Remove stop words from the list of tokens
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # Join the cleaned tokens back into a single string for further analysis
    cleaned_text = ' '.join(cleaned_tokens)
    
    # Return the cleaned text
    return cleaned_text

# Function to preprocess the dataset
def preprocess_data(input_file, output_file):
    # Step 1: Load the Reddit data from the CSV file into a Pandas DataFrame for easier manipulation
    df = pd.read_csv(input_file)

    # Step 2: Apply the clean_text function to each row in the 'Title' column
    # The lambda function ensures the clean_text function is applied to every value in the 'Title' column.
    df['Cleaned_Body'] = df['Title'].apply(lambda x: clean_text(str(x)))  # Clean the text in the 'Title' column

    # Step 3: Save the cleaned DataFrame (with the new 'Cleaned_Body' column) to a new CSV file
    df.to_csv(output_file, index=False)  # Save the cleaned data without adding row numbers

# Main block to execute the preprocessing function when the script is run
if __name__ == "__main__":
    # Input file is the raw Reddit posts CSV, and the output will be the cleaned version of this file
    input_file = 'reddit_posts.csv'
    output_file = 'cleaned_reddit_posts.csv'
    
    # Step 4: Call the preprocess_data function to clean the text and save the output
    preprocess_data(input_file, output_file)
