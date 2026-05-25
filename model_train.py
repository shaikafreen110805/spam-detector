import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import urllib.request
import os

os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

print("="*60)
print("SPAM DETECTION MODEL TRAINING")
print("="*60)

print("\n📥 Downloading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
urllib.request.urlretrieve(url, "sms.tsv")

print("📊 Loading data...")
df = pd.read_csv('sms.tsv', sep='\t', header=None, names=['label', 'message'])
print(f"✅ Loaded {len(df)} messages")

nltk.download('stopwords', quiet=True)

print("🧹 Cleaning text...")
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Mark URL patterns (replace with [URL] token - helps detect phishing)
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    text = re.sub(r'www\.[^\s]+', '[URL]', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    
    # Keep important phishing indicators
    # ($, %, numbers might indicate scams)
    text = re.sub(r'\b\d{4,}\b', '[NUMBER]', text)  # Large numbers become [NUMBER]
    
    # Remove punctuation but keep important symbols
    text = re.sub(r'[^\w\s\[\]\/]', ' ', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stop words but keep important phishing indicators
    important_words = {'urgent', 'verify', 'account', 'suspended', 'limited', 'click', 'confirm'}
    words = [stemmer.stem(word) for word in words 
             if word not in stop_words or word in important_words]
    
    return ' '.join(words)
df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print("🔢 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🤖 Training Naive Bayes...")
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*60)
print("📊 MODEL PERFORMANCE")
print("="*60)
print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall:    {recall:.4f}")
print(f"✅ F1-Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Spam Detection')
plt.savefig('static/confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()

# Feature Importance
feature_names = vectorizer.get_feature_names_out()
spam_coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]
top_idx = spam_coef.argsort()[-10:][::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(10), spam_coef[top_idx])
plt.yticks(range(10), [feature_names[i] for i in top_idx])
plt.xlabel('Importance Score')
plt.title('Top 10 Spam Indicators')
plt.tight_layout()
plt.savefig('static/feature_importance.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n💾 Saving model...")
joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("\n🎉 Training complete! Model ready for web app.")