import pandas as pd

# Load the cleaned Reddit data from the CSV file
# This data contains the Reddit posts that have already been cleaned (e.g., no URLs, punctuation, etc.)
df = pd.read_csv('cleaned_reddit_posts.csv')

# Function to manually assign labels to posts based on the presence of specific keywords
def assign_label(text):
    """
    This function assigns labels to text data based on the presence of certain keywords.
    It checks if specific words related to 'misinformation' or 'hate speech' appear in the text.
    
    Args:
    - text (str): The cleaned text of a Reddit post.
    
    Returns:
    - str: The label ('misinformation', 'hate_speech', or 'neutral') depending on the detected keywords.
    """
    
    # Expanded list of keywords typically associated with misinformation
    # These words indicate that the post might be spreading false information or misleading content
    # Expanded list of keywords typically associated with misinformation
    # These words indicate that the post might be spreading false information or misleading content
    misinformation_keywords = [
        'fake', 'false', 'hoax', 'misleading', 'conspiracy', 'untrue', 'scam', 
        'debunked', 'fact-check', 'disinformation', 'fraud', 'fabricated', 
        'propaganda', 'rumor', 'discredited', 'distorted', 'baseless', 'forged', 
        'manipulated', 'phishing', 'spam', 'clickbait', 'deceptive', 'falsified',
        'lie', 'lies', 'hoaxes', 'malinformation', 'misinfo', 'disinfo', 'unverified',
        'exaggerated', 'falsehood', 'fakery', 'manipulation', 'conspiracies', 
        'pseudoscience', 'myth', 'incorrect', 'fake news', 'misinformed', 
        'false claim', 'false narrative', 'altered', 'counterfeit', 'deceptive tactics', 
        'plagiarism', 'scare tactic', 'deepfake', 'infodemic', 'untrustworthy', 
        'deception', 'misreporting', 'misrepresentation', 'tampered', 'unconfirmed'
    ]

    # Expanded list of keywords for hate speech
    # These words indicate that the post might contain hateful or discriminatory language
    hate_speech_keywords = [
        'hate', 'racist', 'violent', 'extremist', 'offensive', 'bigotry', 
        'xenophobic', 'homophobic', 'discriminatory', 'harassment', 'anti-semitic', 
        'slur', 'genocide', 'terrorism', 'ethnic cleansing', 'antisemitism', 
        'homophobia', 'transphobia', 'islamophobia', 'sexist', 'derogatory', 
        'hateful', 'intolerant', 'slurs', 'threat', 'ethnic slur', 'racial slur', 
        'discrimination', 'hostility', 'bullying', 'cyberbullying', 'insult', 
        'mockery', 'demeaning', 'defamatory', 'violence', 'antisemitic', 'racism', 
        'aggression', 'neo-nazi', 'supremacy', 'white supremacy', 'sexism', 
        'misogyny', 'homophobe', 'disrespectful', 'dehumanizing', 'abusive', 
        'incite violence', 'hate crime', 'slander', 'inflammatory', 'persecution', 
        'racial abuse', 'victimization', 'marginalization', 'oppression', 
        'prejudice', 'segregation', 'xenophobe', 'harassing', 'denigration', 
        'alienation', 'lynching', 'bias', 'hate speech', 'exclusion'
    ]


    # Check if any of the 'misinformation' keywords appear in the text
    if any(word in text for word in misinformation_keywords):
        # If one or more misinformation keywords are found, label the post as 'misinformation'
        return 'misinformation'
    
    # Check if any of the 'hate speech' keywords appear in the text
    elif any(word in text for word in hate_speech_keywords):
        # If one or more hate speech keywords are found, label the post as 'hate_speech'
        return 'hate_speech'
    
    # If neither misinformation nor hate speech keywords are found, label the post as 'neutral'
    else:
        return 'neutral'

# Apply the 'assign_label' function to each post in the 'Cleaned_Body' column
# This will create a new column 'Category' that holds the manually assigned label for each post
# The lambda function is used to apply 'assign_label' to every row in the DataFrame
df['Category'] = df['Cleaned_Body'].apply(lambda x: assign_label(str(x)))

# Save the new DataFrame, including the 'Category' column, to a new CSV file
# This labeled dataset can now be used for training machine learning models or further analysis
df.to_csv('labeled_reddit_posts.csv', index=False)

# Display the first few rows of the DataFrame to verify that the labels were correctly assigned
# The first 60 rows are shown to see a good sample of the labeled data
print(df[['Title', 'Cleaned_Body', 'Category']].head(50))
