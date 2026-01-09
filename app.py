import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("Reviews.csv")

# Keep only required columns
data = data[['Text', 'Score']]

# Convert scores to sentiment
# 1,2 -> negative (0)
# 4,5 -> positive (1)
data = data[data['Score'] != 3]   # remove neutral
data['sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)

# Take sample for speed (optional but recommended)
data = data.sample(5000)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['cleaned_review'] = data['Text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test custom input
sample = ["This product is horrible and a waste of money"]
sample_clean = [clean_text(sample[0])]
sample_vec = vectorizer.transform(sample_clean).toarray()
prediction = model.predict(sample_vec)

if prediction[0] == 1:
    print("Sentiment: Positive 😊")
else:
    print("Sentiment: Negative 😞")
