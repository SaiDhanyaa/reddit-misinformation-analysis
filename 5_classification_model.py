from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os


# Create 'results' directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# ---- Load the Data ----
# Load the labeled Reddit data (which should contain the 'Category' column) from a CSV file
# 'cleaned_reddit_posts.csv' is the file containing cleaned text posts and the labels (Category)
df = pd.read_csv('labeled_reddit_posts.csv')

# ---- Check if 'Category' Column Exists ----
# Ensure that the 'Category' column exists in the dataset
if 'Category' not in df.columns:
    raise KeyError("The 'Category' column is missing from the dataset. Make sure the data has been labeled.")

# ---- Convert Text Data to Numerical Features using TF-IDF ----
# TF-IDF (Term Frequency-Inverse Document Frequency) is used to transform the cleaned text into numerical features
tfidf = TfidfVectorizer(max_features=1000)  # Consider only the top 1000 most important words

# Transform the 'Cleaned_Body' text column into a numerical matrix of word features
X = tfidf.fit_transform(df['Cleaned_Body'].dropna())  # Ensure no missing values

# ---- Define the Target Column ----
# The 'Category' column contains the labels (misinformation, hate speech, neutral)
y = df['Category']

# ---- Split the Data into Training and Test Sets ----
# We split the data into training (70%) and testing (30%) sets to evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- Train a Naive Bayes Classifier ----
# Using the Multinomial Naive Bayes algorithm for text classification
model = MultinomialNB()
model.fit(X_train, y_train)  # Train the model with the training data

# ---- Make Predictions on the Test Set ----
# Use the trained model to predict the categories of the test data
y_pred = model.predict(X_test)

# ---- Evaluate the Model ----
# Calculate and print the accuracy, precision, recall, and F1-score of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

# ---- Print the Evaluation Metrics ----
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save evaluation metrics to a text file
with open('results/evaluation_metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")



# Assuming you have y_test (true labels) and y_pred (predicted labels)
# Generate classification report

# ---- Print Classification Report ----
print("Classification Report:")
classification_report_str = classification_report(y_test, y_pred)
print(classification_report_str)

# Save classification report to a text file
with open('results/classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(classification_report_str)




# ---- Generate and Save Confusion Matrix Plot ----
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Save confusion matrix plot to a PNG file
plt.savefig('results/confusion_matrix.png', bbox_inches='tight')

# Display the confusion matrix plot
plt.show()