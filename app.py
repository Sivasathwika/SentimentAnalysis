from flask import Flask, render_template, request
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
data = pd.read_csv("Reviews.csv")
data = data[['Text', 'Score']]
data = data[data['Score'] != 3]
data['sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)

# Use sample for performance
data = data.sample(5000)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.remove('not')


def clean_text(text):
    if pd.isna(text):
        return ""

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()

    processed_words = []
    negate = False

    for word in words:
        if word == 'not':
            negate = True
            processed_words.append(word)
            continue

        if negate:
            processed_words.append("not_" + word)
            negate = False
        else:
            if word not in stop_words:
                processed_words.append(stemmer.stem(word))

    return ' '.join(processed_words)


data['cleaned_review'] = data['Text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review']).toarray()
y = data['sentiment']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        review = request.form['review']
        review_clean = clean_text(review)
        review_vec = vectorizer.transform([review_clean]).toarray()
        prediction = model.predict(review_vec)

        if prediction[0] == 1:
            result = "Positive 😊"
        else:
            result = "Negative 😞"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
