from flask import Flask, render_template, request
from pymongo import MongoClient
import joblib
from config import MONGO_URI, DB_NAME

# ---------------------
# Initialize Flask app
# ---------------------
app = Flask(__name__)

# ---------------------
# Connect to MongoDB
# ---------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db['news_articles']

# ---------------------
# Load trained model & vectorizer
# ---------------------
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# ---------------------
# Routes
# ---------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news_text']

    # Transform and predict
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]

    # Store prediction in MongoDB
    collection.insert_one({
        "text": text,
        "prediction": prediction
    })

    return render_template('index.html', prediction=prediction, text=text)

# ---------------------
# Run the app
# ---------------------
if __name__ == '__main__':
    app.run(debug=True)
