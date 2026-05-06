import requests
import zipfile
import io
import os
import pandas as pd
import nltk
import re
import numpy as np

#load in the url of the dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
#download the dataset
response = requests.get(url)
if response.status_code == 200:
    print("Download successful.")

    #extract dataset
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("sms_spam_collection")
        print("extraction successful") 
    
else:
    print("Failed to download the dataset.") 
    
#listing the extracted files
extracted_files = os.listdir("sms_spam_collection")
print("Extracted files:", extracted_files)

#loading the dataset
df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label","message"],
)

#displaying basic info about the dataset
print("-----HEAD-----")
print(df.head())
print("-----Describe-----")
print(df.describe())
print("-----Info-----")
print(df.info())

#check for missing values
print("Missing values:\n", df.isnull().sum())

#check for duplicates 
print("Duplicate entries:", df.duplicated().sum())

#check for duplicates if any
df = df.drop_duplicates()

#Download necessary ntlk data files 
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

print("==Before any preprocessing==")
print(df.head(5))

#converting all of the messages to lowercase
df["message"] = df["message"].str.lower()
print("\n==After lower casting==")
print(df["message"].head(5))

#removing non-essential punctuation and numbers and keep useful symbols like $ and !
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]","",x))
print("\n=== After removing punctuation and numbers===")
print(df["message"].head(5))

from nltk.tokenize import word_tokenize
#split each message into individual tokens
df["message"]=df["message"].apply(word_tokenize)
print("\n===After tokenization===")
print(df["message"].head(5))

from nltk.corpus import stopwords
#define set of english words to stop words and remove them from tokens
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\n===After removing stop words===")
print(df["message"].head(5))

from nltk.stem import PorterStemmer
#stem each token to reduce words into their base form 
stemmer = PorterStemmer()
df["message"]= df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\n===After stemming===")
print(df["message"].head(5))

#rejoin tokens into a single string for feature extraction
df["message"]= df["message"].apply(lambda x: " ".join(x))
print("\n===After joining tokens back into strings===")
print(df["message"].head(5))

from sklearn.feature_extraction.text import CountVectorizer

#intialise countvertorizer with bigrams, min_df, and max_df to focus on relevant terms
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))
#fit and transform message column
x = vectorizer.fit_transform(df["message"])

#labels(target variable)
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

#pipeline to ensure data flows through Vectorization before Classification
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])

from sklearn.model_selection import GridSearchCV

#Define variations of alphas we're testing
param_grid = { "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0] } 

#5 fold cross validation ensuring model is tested rigorously across different slices of data
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")

#Execute search
grid_search.fit(df["message"], y)

#Extract winner
best_model = grid_search.best_estimator_
print("Optimal alpha found:", grid_search.best_params_)

new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]

# Preprocess function that mirrors the training-time preprocessing
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens) 

# Preprocess and vectorize messages
processed_messages = [preprocess_message(msg) for msg in new_messages] 


# Transform preprocessed messages into feature vectors
X_new = best_model.named_steps["vectorizer"].transform(processed_messages)

# Predict with the trained classifier
predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_probabilities = best_model.named_steps["classifier"].predict_proba(X_new)

# Display predictions and probabilities for each evaluated message
for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
    spam_probability = prediction_probabilities[i][1]  # Probability of being spam
    ham_probability = prediction_probabilities[i][0]   # Probability of being not spam
    
    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Not-Spam Probability: {ham_probability:.2f}")
    print("-" * 50) 

import joblib

# Save the trained model to a file for future use
model_filename = 'spam_detection_model.joblib'
joblib.dump(best_model, model_filename)

print(f"Model saved to {model_filename}")


# Load the saved model
loaded_model = joblib.load(model_filename)

# Preprocess new messages before prediction
new_data_processed = [preprocess_message(msg) for msg in new_messages]

# Make predictions on the preprocessed data
predictions = loaded_model.predict(new_data_processed) 
