import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
import numpy as np


# #### Loading the Data

path = '/Users/jennishachristinamartin/Downloads/merged_training.pkl'
# loading the data
d = pd.read_pickle(path)
print(d.head())


# #### Exploring the Data

print(f"Number of rows: {len(d)}")
print(f"Number of variables: {len(d.columns)}\n")
print(d.info())


# #### Summarizing Data

# Getting the summary
print("\n Statistics:")
print(d.describe())


# #### Checking for Missing Values

print(d.isnull().sum())


# checking for NaN missing values
nan_values = d.isna().sum()

# printing the number of NaN values for each column
print("\nNumber of NaN values in each column:")
print(nan_values)


print(d.columns)


# #### Identifying the Numeric Variables and Categorical Variables


# getting the numerical variables
numerical_vars = d.select_dtypes(include=['number']).columns.tolist()
# getting the categorical variables
categorical_vars = d.select_dtypes(include=['object', 'category']).columns.tolist()
# displaying the numerical variables
print("Numerical variables:", numerical_vars)
# displaying the categorical variables
print("Categorical variables:", categorical_vars)


# #### Exploring the Label Distribution


label_dis = d['emotions'].value_counts()
print(label_dis)


# ####  Distribution of Each Emotion


# plotting the distribution of each emotion in the data set
plt.figure(figsize=(12, 6))
# counting the occurrences of each emotion
label_counts = d['emotions'].value_counts()  
sns.barplot(x=label_counts.index, y=label_counts.values, palette="coolwarm")

plt.title("Emotion Distribution", fontsize=16)
plt.xlabel("Emotion", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45)
plt.show()


# #### The Most Common Sentences from Each Emotion


# displaying the most common sentence per emotion
from collections import Counter

# finding the most common sentence for each emotion
for emotion in d['emotions'].unique():
    # getting all the sentences for the emotion
    sentences = d[d['emotions'] == emotion]['text']
    
    # counting the occurrences of each sentence
    common_sentence = Counter(sentences).most_common(1)[0]  
    
    # displaying the emotion and its most common sentences
    print(f"Emotion: {emotion}")
    print(f"Most Common Sentence: '{common_sentence[0]}'")
    print(f"Occurrences: {common_sentence[1]}")
    print("\n" + "-"*60 + "\n")


# #### Preprocessing the text

def preprocess_text(text):
    # converting to lowercase and removing punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # tokenizing and removing stopwords
    tokens = [word for word in word_tokenize(text) if word not in stopwords.words('english')]
     # joining the tokens back into a single string
    return ' '.join(tokens)

d['cleaned_text'] = d['text'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else "")


print(d['cleaned_text'])


# ### Visualizations

# #### Analyzing the length of the text


d['Text_Length'] = d['cleaned_text'].apply(lambda x: len(x.split()))

# plotting the distribution of text length per emotion
plt.figure(figsize=(12, 6))
sns.boxplot(data=d, x='emotions', y='Text_Length')
plt.title("Text Length by Emotion Analysis")
plt.xlabel("Emotions")
plt.ylabel("Text Length (Number of Words)")
plt.xticks(rotation=45)
plt.show()


# #### Vocabulary Size and Common Words Analysis

# calculating the vocabulary size for each emotion
vocab_sizes = d.groupby('emotions')['cleaned_text'].apply(lambda x: len(set(' '.join(x).split())))
print("Vocabulary Size by Emotions:\n", vocab_sizes)


# #### Top 10 Unique Word Pairs

# extracting the word pairs for each emotion
emotion_bigrams = {}
vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')

for emotion in d['emotions'].unique():
    text_data = d[d['emotions'] == emotion]['cleaned_text']
    bigram_matrix = vectorizer.fit_transform(text_data)
    bigram_counts = dict(zip(vectorizer.get_feature_names_out(), bigram_matrix.sum(axis=0).A1))
    emotion_bigrams[emotion] = Counter(bigram_counts)

# Identifying the common word pairs across all the emotions and filtering them out
common_bigrams = set.intersection(*[set(bigrams.keys()) for bigrams in emotion_bigrams.values()])
unique_emotion_bigrams = {emotion: [bigram for bigram in bigrams if bigram not in common_bigrams]
                          for emotion, bigrams in emotion_bigrams.items()}

# displaying the top 10 unique word pairs for each emotion
for emotion, bigrams in unique_emotion_bigrams.items():
    top_bigrams = Counter({bigram: emotion_bigrams[emotion][bigram] for bigram in bigrams}).most_common(10)
    
    # creating a data frame
    bigram_df = pd.DataFrame(top_bigrams, columns=['Bigram', 'Count'])
    
    # plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='Bigram', data=bigram_df, palette='viridis')
    plt.title(f"Top Unique Word Pairs for Each Emotion: {emotion}")
    plt.xlabel("Frequency")
    plt.ylabel("Word Pairs")
    plt.show()


# #### Comparing Sentiment Polarity for Each Emotion

d['polarity'] = d['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

plt.figure(figsize=(12, 8))
sns.boxplot(data=d, x='emotions', y='polarity', palette='coolwarm')
plt.title("Sentiment Polarity Distribution by Emotion")
plt.xlabel("Emotion")
plt.ylabel("Sentiment Polarity")
plt.xticks(rotation=45)
plt.show()
