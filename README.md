# Reddit Misinformation Analysis

## Overview

This project is aimed at analyzing Reddit posts to detect misinformation, hate speech, and neutral content using various text analysis and machine learning techniques. The project consists of several steps, including data extraction, preprocessing, exploratory data analysis (EDA), text analysis using NLP, manual labeling, and training a classification model to predict post categories.

## Project Structure

The project is divided into the following files:

1. **`0_reddit_data_extractor.py`**: Extracts Reddit posts using the PRAW API and saves the data to a CSV file.
2. **`1_data_preprocessing.py`**: Cleans the text data by removing URLs, punctuation, numbers, and stopwords, and tokenizes the text.
3. **`2_eda.py`**: Performs exploratory data analysis (EDA) on the text data, including visualizing word frequencies and generating a word cloud.
4. **`3_text_analysis.py`**: Conducts sentiment analysis, topic modeling (using LDA), and keyword extraction.
5. **`4_manual_labeling.py`**: Manually assigns labels to posts as "misinformation", "hate speech", or "neutral" based on predefined keywords.
6. **`5_classification_model.py`**: Trains a Naive Bayes classification model to predict the category of Reddit posts and evaluates its performance using accuracy, precision, recall, and F1-score.

## Prerequisites

- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `nltk`
  - `gensim`
  - `scikit-learn`
  - `matplotlib`
  - `wordcloud`
  
You can install the required packages by running:
```bash
pip install pandas numpy nltk gensim scikit-learn matplotlib wordcloud
```

## How to Run the Project

1. **Data Extraction (`0_reddit_data_extractor.py`):**
   - This script extracts Reddit posts from a specified subreddit using the PRAW API and saves the data into a CSV file.
   - **Command:** `python 0_reddit_data_extractor.py`
   - **Output:** A CSV file named `reddit_posts.csv` containing the extracted Reddit posts.

2. **Data Preprocessing (`1_data_preprocessing.py`):**
   - This script cleans the extracted data by removing unnecessary elements like URLs, punctuation, and prepares it for further analysis.
   - **Command:** `python 1_data_preprocessing.py`
   - **Output:** A cleaned dataset saved as `cleaned_reddit_posts.csv`.

3. **Exploratory Data Analysis (EDA) (`2_eda.py`):**
   - This script performs basic exploratory data analysis (EDA) on the cleaned data, including word frequency visualization and generating a word cloud.
   - **Command:** `python 2_eda.py`
   - **Output:** Word frequency visualizations and word cloud images saved in the `plots` directory.

4. **Text Analysis using NLP (`3_text_analysis.py`):**
   - This script performs sentiment analysis using the VADER model, keyword extraction using TF-IDF, and topic modeling using Latent Dirichlet Allocation (LDA).
   - **Command:** `python 3_text_analysis.py`
   - **Output:** Prints sentiment scores, top keywords, and topics found in the Reddit posts to the console.

5. **Manual Labeling (`4_manual_labeling.py`):**
   - This script manually assigns labels to the posts as "misinformation", "hate speech", or "neutral" based on predefined keywords.
   - **Command:** `python 4_manual_labeling.py`
   - **Output:** A labeled dataset saved as `labeled_reddit_posts.csv`, with an additional column 'Category' for the assigned labels.

6. **Classification Model (`5_classification_model.py`):**
   - This script trains a Naive Bayes classification model on the manually labeled data to predict the category (misinformation, hate speech, or neutral) of a Reddit post. It also evaluates the model’s performance using metrics like accuracy, precision, recall, and F1-score.
   - **Command:** `python 5_classification_model.py`
   - **Output:** Prints the classification performance metrics (accuracy, precision, recall, F1-score) to the console.

## Key Features

- **Text Cleaning and Preprocessing:** Removes noise from raw text, including URLs, punctuation, and stopwords.
- **Sentiment Analysis:** Analyzes the sentiment (positive, negative, or neutral) of Reddit posts using VADER.
- **Topic Modeling:** Uses Latent Dirichlet Allocation (LDA) to discover underlying topics in the posts.
- **Keyword Extraction:** Identifies the most important words in the dataset using TF-IDF.
- **Manual Labeling:** Automatically assigns labels (misinformation, hate speech, neutral) based on the presence of certain keywords.
- **Machine Learning Model:** Trains and evaluates a Naive Bayes classification model for text classification.

## Future Improvements

- **Advanced Classification Models:** Experiment with more sophisticated models like Random Forest, Logistic Regression, or deep learning models.
- **Improved Labeling:** Use a more robust dataset or fine-tune the labeling mechanism for better classification accuracy.
- **Real-time Data:** Set up a pipeline to continuously monitor and classify Reddit posts in real-time.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or inquiries, feel free to contact the project maintainer at [your-email@example.com].
