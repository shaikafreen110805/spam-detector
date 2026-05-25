from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Important words to keep even if they're stop words
important_words = {'urgent', 'verify', 'account', 'suspended', 'limited', 
                   'click', 'confirm', 'update', 'alert', 'security', 
                   'paypal', 'bank', 'amazon', 'netflix', 'apple', 'microsoft'}

def clean_text(text):
    """Enhanced text cleaning with phishing pattern preservation"""
    # Convert to lowercase
    text = text.lower()
    
    # Mark URLs as special token - helps detect phishing links
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    text = re.sub(r'www\.[^\s]+', '[URL]', text)
    
    # Mark email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    
    # Mark large numbers (potential money amounts)
    text = re.sub(r'\b\d{4,}\b', '[AMOUNT]', text)
    
    # Remove punctuation but keep important symbols
    text = re.sub(r'[^\w\s\[\]\/]', ' ', text)
    
    # Tokenize
    words = text.split()
    
    # Stem and filter stop words (keep important ones)
    words = [stemmer.stem(word) for word in words 
             if word not in stop_words or word in important_words]
    
    return ' '.join(words)

def extract_phishing_indicators(text):
    """Extract specific red flags for UI display"""
    indicators = []
    text_lower = text.lower()
    
    # Financial/brand mentions
    if any(word in text_lower for word in ['paypal', 'bank', 'chase', 'wells fargo', 'boa', 'bank of america']):
        indicators.append("🏦 Financial account mentioned")
    if any(word in text_lower for word in ['amazon', 'netflix', 'apple', 'microsoft', 'google']):
        indicators.append("📦 Well-known brand mentioned")
    
    # Suspicious requests
    if any(word in text_lower for word in ['verify', 'confirm']):
        indicators.append("🔐 Verification requested")
    if any(word in text_lower for word in ['update', 'renew', 'upgrade']):
        indicators.append("📝 Account update requested")
    if any(word in text_lower for word in ['click here', 'click the link', 'visit']):
        indicators.append("🔗 Suspicious link detected")
    
    # Urgency/pressure tactics
    if any(word in text_lower for word in ['urgent', 'immediately', 'asap']):
        indicators.append("⏰ Urgency/pressure tactics")
    if any(word in text_lower for word in ['suspended', 'limited', 'locked', 'disabled', 'closed']):
        indicators.append("⚠️ Account threat detected")
    if any(word in text_lower for word in ['48 hours', '24 hours', 'soon']):
        indicators.append("⌛ Time pressure detected")
    
    # Scam patterns
    if any(word in text_lower for word in ['winner', 'won', 'prize', 'congratulations']):
        indicators.append("🏆 Lottery/prize claim")
    if any(word in text_lower for word in ['million', 'billion', 'cash']):
        indicators.append("💰 Unrealistic money offer")
    if 'http' in text_lower or 'https' in text_lower:
        indicators.append("🔗 External link present")
    
    # Grammar/spelling issues
    if re.search(r'[A-Z]{5,}', text):
        indicators.append("📢 Excessive capitalization")
    
    return indicators

def predict_email(text):
    """Main prediction function"""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    
    label = "spam" if prediction == 1 else "ham"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    # Extract phishing indicators
    indicators = extract_phishing_indicators(text)
    
    # Generate advice based on prediction
    if label == "spam":
        advice = "⚠️ DO NOT click any links, reply to the sender, or share personal information. Mark as spam and delete."
    else:
        advice = "✓ No spam patterns detected. However, always verify unexpected emails from known brands by visiting their official website directly."
    
    return label, round(confidence * 100, 2), indicators, advice

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
        return jsonify({'error': 'Please enter email text'}), 400
    
    label, confidence, indicators, advice = predict_email(email_text)
    
    return jsonify({
        'label': label,
        'confidence': confidence,
        'indicators': indicators,
        'advice': advice
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)