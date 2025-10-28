import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---------------------
# 1️⃣ Load Dataset
# ---------------------
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine them
df = pd.concat([fake_df, true_df])
df = df.sample(frac=1).reset_index(drop=True)  # shuffle

# ---------------------
# 2️⃣ Split Data
# ---------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------
# 3️⃣ Vectorize Text
# ---------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------
# 4️⃣ Train Model
# ---------------------
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# ---------------------
# 5️⃣ Evaluate
# ---------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# ---------------------
# 6️⃣ Save Model & Vectorizer
# ---------------------
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Model and vectorizer saved to /models/")
