import praw  #we’ll use PRAW (Python Reddit API Wrapper) to interact with Reddit’s API
import pandas as pd

# Reddit API credentials: These are the details you get when you create an app on Reddit.
# They allow you to authenticate and interact with Reddit's API.
client_id = 'bgKcbuJwuzxc5rf9IsU3cw'  # Unique ID for your Reddit app
client_secret = 'i-mAQI5whBaY7o6JQhWMq8E_ah1tyQ'  # Secret key for your Reddit app
user_agent = 'DHANYAPRIYA SOMASUNDARAM Social Media Data Scraper'  # A descriptive name for your app/script

# Authenticate with Reddit: This step creates a Reddit instance using the credentials.
# This instance is used to interact with Reddit's API and fetch data.
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Define the subreddit to scrape: A subreddit is a specific community within Reddit that focuses on a particular topic.
# In this case, we're choosing the "news" subreddit, which contains posts related to current news.
subreddit_name = "news"
subreddit = reddit.subreddit(subreddit_name)

# Fetch posts: 
# `subreddit.hot(limit=100)` fetches the top 100 posts from the "hot" section of the subreddit. 
# The "hot" section typically contains trending or popular posts.
# We loop through each post and extract important attributes like title, body, score (upvotes), and created date.

posts = []
for post in subreddit.hot(limit=100):  # Loop through the top 100 posts in the subreddit
    posts.append([post.title, post.selftext, post.score, post.created])  # Append post details to the list

# Convert posts to a DataFrame: Once we have collected the posts in a list, we convert it to a DataFrame (tabular format)
# using pandas. Each row represents a Reddit post, and the columns store post details (title, body, score, created date).
df = pd.DataFrame(posts, columns=['Title', 'Body', 'Score', 'Created'])  # Create DataFrame with appropriate column names

# Convert Unix timestamp to a readable date: Reddit stores the post creation time as a Unix timestamp (seconds since Jan 1, 1970).
# We convert this to a more readable date and time format using pandas' `to_datetime` function.
df['Created'] = pd.to_datetime(df['Created'], unit='s')  # Convert 'Created' from Unix timestamp to human-readable datetime

# Save the DataFrame to a CSV file: The collected data (title, body, score, date) is now saved to a CSV file called `reddit_posts.csv`.
# This allows us to analyze or visualize the data later.
df.to_csv('reddit_posts.csv', index=False)  # Save the DataFrame to a CSV file, without saving row indices

# Display the first few rows of the data: We print the first 5 rows of the DataFrame to check if the data was fetched and formatted correctly.
# This helps verify that the scraping process worked as expected.
print(df.head())  # Display the first few rows of the DataFrame
