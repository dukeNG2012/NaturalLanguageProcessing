import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset into a DataFrame
data = pd.read_csv("train_data_NLP_DucMinh.csv")

# Split the data into training and testing sets
X = data[["#1 String", "#2 String"]]
y = data["Quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train["#1 String"])
X_test_tfidf = tfidf_vectorizer.transform(X_test["#1 String"])

# Initialize and train a binary classification model (e.g., Logistic Regression)
clf = LogisticRegression()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
