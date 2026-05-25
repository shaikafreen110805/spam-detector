from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load model
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict(email_text):
    cleaned = clean_text(email_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    label = "spam" if pred == 1 else "ham"
    confidence = prob[1] if pred == 1 else prob[0]
    
    return label, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    email_text = data.get('email', '')
    
    if not email_text.strip():
        return jsonify({'error': 'Please enter text'}), 400
    
    label, confidence = predict(email_text)
    
    return jsonify({
        'label': label,
        'confidence': round(confidence * 100, 2)
    })
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)