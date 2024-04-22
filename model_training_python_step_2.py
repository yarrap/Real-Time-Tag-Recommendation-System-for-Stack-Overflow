## importing libraries
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

## calculating the start time 
start_time = time.time()

# Read data from CSV file into a pandas DataFrame
df = pd.read_csv("data.csv")

# Remove rows with any missing values
df = df.dropna(axis=0)

# Truncate 'Body' and 'Title' columns to 300 characters
df['Body'] = df['Body'].str.slice(0, 300)
df['Title'] = df['Title'].str.slice(0, 300)

# Explode DataFrame to have one row for each 'Tag'
exploded_df = df.explode('Tags').reset_index(drop=True)

# Separate 'Title' and 'Body' columns
X_title = exploded_df['Title']
X_body = exploded_df['Body']

# Explode DataFrame further to have one row for each 'Tag' separately
exploded_df = pd.DataFrame(exploded_df).explode('Tags').reset_index(drop=True)

# Label encode the 'Tags' column to convert categorical tags to numerical values
encoder = LabelEncoder()
exploded_df['Encoded_Values'] = encoder.fit_transform(exploded_df['Tags'])

# Extract encoded tags as labels
y = exploded_df['Encoded_Values']

# Create TF-IDF vectorizers for 'Title' and 'Body' separately
vectorizer_title = TfidfVectorizer(
    analyzer = 'word', 
    strip_accents = None, 
    encoding = 'utf-8', 
    preprocessor=None, 
    max_features=1000)

vectorizer_body = TfidfVectorizer(
    analyzer = 'word', 
    strip_accents = None, 
    encoding = 'utf-8', 
    preprocessor=None, 
    max_features=1000)

# Transform 'Title' and 'Body' text data into TF-IDF vectors
X_title_vect = vectorizer_title.fit_transform(X_title)
X_body_vect = vectorizer_body.fit_transform(X_body)

# Combine TF-IDF vectors horizontally
X = hstack([X_title_vect, X_body_vect])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Create a Random Forest classifier model
rf_model = RandomForestClassifier()

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculating runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")

