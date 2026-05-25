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
    if any(word in text_lower for word in ['amazon', 'netflix', 'apple', 'microsoft', 'google', 'ebay', 'walmart']):
        indicators.append("📦 Well-known brand mentioned")
    
    # Suspicious requests
    if any(word in text_lower for word in ['verify', 'confirm']):
        indicators.append("🔐 Verification requested")
    if any(word in text_lower for word in ['update', 'renew', 'upgrade']):
        indicators.append("📝 Account update requested")
    if any(word in text_lower for word in ['click here', 'click the link', 'visit', 'http', 'https']):
        indicators.append("🔗 Suspicious link detected")
    
    # Urgency/pressure tactics
    if any(word in text_lower for word in ['urgent', 'immediately', 'asap', 'right now']):
        indicators.append("⏰ Urgency/pressure tactics")
    if any(word in text_lower for word in ['suspended', 'limited', 'locked', 'disabled', 'closed', 'restricted']):
        indicators.append("⚠️ Account threat detected")
    if any(word in text_lower for word in ['48 hours', '24 hours', 'soon', 'today', 'immediately']):
        indicators.append("⌛ Time pressure detected")
    
    # Scam patterns
    if any(word in text_lower for word in ['winner', 'won', 'prize', 'congratulations', 'claim']):
        indicators.append("🏆 Lottery/prize claim")
    if any(word in text_lower for word in ['million', 'billion', 'cash', 'free', 'money']):
        indicators.append("💰 Unrealistic money offer")
    
    return indicators

def force_phishing_detection(text, label, confidence):
    """FORCE detection of obvious phishing emails - Rule-based override"""
    text_lower = text.lower()
    
    # Keywords that ALWAYS indicate phishing
    phishing_keywords = [
        'bank of america', 'boa', 'chase', 'wells fargo', 'paypal',
        'verify your account', 'account has been limited', 'unusual activity',
        'confirm your identity', 'account suspended', 'click here to verify',
        'within 24 hours', 'account will be closed', 'suspicious login',
        'unusual login', 'restricted access', 'verify immediately',
        'account limited', 'temporarily restricted', 'security alert'
    ]
    
    # Count matches
    matches = sum(1 for keyword in phishing_keywords if keyword in text_lower)
    
    # Check for URL patterns
    has_url = 'http://' in text_lower or 'https://' in text_lower or '.com' in text_lower
    
    # Force SPAM if multiple indicators
    if matches >= 2 and has_url:
        print(f"⚠️ FORCED PHISHING: {matches} keyword matches + URL")
        return 'spam', 95.0
    
    if matches >= 3:
        print(f"⚠️ FORCED PHISHING: {matches} keyword matches")
        return 'spam', 95.0
    
    # Specific brand phishing detection
    brand_phishing = [
        ('bank of america', 'verify', 'spam'),
        ('bank of america', 'alert', 'spam'),
        ('bank of america', 'unusual', 'spam'),
        ('paypal', 'verify', 'spam'),
        ('paypal', 'limited', 'spam'),
        ('paypal', 'suspended', 'spam'),
        ('amazon', 'payment', 'spam'),
        ('amazon', 'cancelled', 'spam'),
        ('netflix', 'expire', 'spam'),
        ('netflix', 'subscription', 'spam'),
        ('chase', 'alert', 'spam'),
        ('wells fargo', 'alert', 'spam'),
    ]
    
    for brand, action, result in brand_phishing:
        if brand in text_lower and action in text_lower:
            if has_url or 'click' in text_lower:
                print(f"⚠️ FORCED PHISHING: {brand} {action} scam")
                return 'spam', 95.0
    
    # Check for known scam patterns
    scam_patterns = [
        'won', 'winner', 'prize', 'lottery', 'million',
        'click here to claim', 'verify your account', 'account suspended'
    ]
    
    scam_matches = sum(1 for pattern in scam_patterns if pattern in text_lower)
    if scam_matches >= 2 and has_url:
        print(f"⚠️ FORCED PHISHING: {scam_matches} scam patterns + URL")
        return 'spam', 95.0
    
    return label, confidence

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
    
    # FORCE PHISHING DETECTION - This overrides the model
    label, confidence = force_phishing_detection(text, label, confidence)
    
    # Generate advice based on final prediction
    if label == "spam":
        advice = "⚠️ DO NOT click any links, reply to the sender, or share personal information. Mark as spam and delete immediately."
        # Add critical indicator if forced
        if confidence >= 95:
            indicators.insert(0, "🔴 CRITICAL: Confirmed phishing attempt - DO NOT respond")
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