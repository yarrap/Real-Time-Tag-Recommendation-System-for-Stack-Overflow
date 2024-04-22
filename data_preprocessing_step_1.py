## importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk   
import re
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# for properly interpreting the text data
ENCODING='ISO-8859-1'

## reading the questions dataframe
df_q = pd.read_csv("Questions.csv", encoding=ENCODING)

## reading the tags dataframe
df_t = pd.read_csv("Tags.csv", encoding=ENCODING)

df_t['Tag'] = df_t['Tag'].astype(str)
# group all tags given to same question into a single string
grouped_tags = df_t.groupby('Id')['Tag'].apply(lambda tags: ' '.join(tags))
grouped_tags.head()


# reset index for simplicity
grouped_tags.reset_index()
# Creating a new DataFrame df_tags_final using the updated grouped_tags DataFrame
df_tags_final = pd.DataFrame({'Id': grouped_tags.index, 'Tags': grouped_tags.values})
df_tags_final.head()

# Merge questions and tags into a single dataframe
df = df_q.merge(df_tags_final, on='Id')
df.head()

# remove questions with score lower than 5 as those are non-relevant
df = df[df['Score'] > 5]
print(df.shape)

# changing all the tags into lower case
df['Tags'] = df['Tags'].apply(lambda tags: tags.lower().split())
df.head()

# get all tags in the dataset
all_tags = []
for tags in df['Tags'].values:
    for tag in tags:
        all_tags.append(tag)
        
# create a frequency list of the tags
tag_freq = nltk.FreqDist(list(all_tags))

# get most common tags
tag_freq.most_common(25)

# get the most common 50 tags without the count
tag_features = list(map(lambda x: x[0], tag_freq.most_common(50)))


#Filters a list of tags, keeping only those that exist in the tag_features list.
def keep_common(tags):
   
    filtered_tags = []
    
    # filter tags
    for tag in tags:
        if tag in tag_features:
            filtered_tags.append(tag)
    
    # return the filtered tag list
    return filtered_tags

# apply the function to filter in dataset
df['Tags'] = df['Tags'].apply(lambda tags: keep_common(tags))
print("dOUBT")
df.head()

# set the Tags column as None for those that do not have a most common tag
df['Tags'] = df['Tags'].apply(lambda tags: tags if len(tags) > 0 else None)

# Now we will drop all the columns that contain None in Tags column
df.dropna(subset=['Tags'], inplace=True)

# initialising tokeniser, stemmer and stop_words for data preprocessing
tokenizer = ToktokTokenizer()
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

## Preprocess the text for vectorization
# - Remove HTML
# - Remove stopwords
# - Remove special characters
# - Convert to lowercase
# - Stemming

def remove_html(text):
    # Remove html and convert to lowercase
    return re.sub(r"\<[^\>]\>", "", text).lower()

def remove_stopwords(text):    
    # tokenize the text
    words = tokenizer.tokenize(text) 
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def remove_punc(text):
    #tokenize
    tokens = tokenizer.tokenize(text)
    # remove punctuations from each token
    tokens = list(map(lambda token: re.sub(r"[^A-Za-z0-9]+", " ", token).strip(), tokens))
    # remove empty strings from tokens
    tokens = list(filter(lambda token: token, tokens))
    return ' '.join(map(str, tokens))

def stem_text(text):
    #tokenize
    tokens = tokenizer.tokenize(text)
    # stem each token
    tokens = list(map(lambda token: stemmer.stem(token), tokens))
    return ' '.join(map(str, tokens))

# drop Id and Score columns since we don't need them
df.drop(columns=['Id', 'Score'], inplace=True)

# apply preprocessing to title 
df['Title'] = df['Title'].apply(lambda x: remove_html(x))
df['Title'] = df['Title'].apply(lambda x: remove_stopwords(x))
df['Title'] = df['Title'].apply(lambda x: remove_punc(x))
df['Title'] = df['Title'].apply(lambda x: stem_text(x))

# apply preprocessing to body
df['Body'] = df['Body'].apply(lambda x: remove_html(x))
df['Body'] = df['Body'].apply(lambda x: remove_stopwords(x))
df['Body'] = df['Body'].apply(lambda x: remove_punc(x))
df['Body'] = df['Body'].apply(lambda x: stem_text(x))

# writing the preprocessed data to a csv file
df.to_csv('data.csv',index=False)

