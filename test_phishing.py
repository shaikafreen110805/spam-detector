import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords', quiet=True)

# Load model
model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://[^\s]+', ' [URL] ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Test emails
test_emails = [
    ("Bank of America Phishing", "Bank of America Security Alert: We detected unusual activity on your checking account. For your protection, we've temporarily restricted access. Please confirm your identity within 24 hours: http://bankofamerica-verify-alert.com"),
    ("PayPal Phishing", "URGENT: Your PayPal account has been temporarily limited. To restore your account, please verify your identity immediately: https://paypal.com.verify-account.com"),
    ("Safe Email", "Hi team, just a reminder about our design review meeting tomorrow at 3pm in Conference Room B."),
]

print("="*60)
print("TESTING PHISHING DETECTION")
print("="*60)

for name, email in test_emails:
    cleaned = clean_text(email)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    result = "PHISHING" if pred == 1 else "SAFE"
    confidence = prob[1]*100 if pred == 1 else prob[0]*100
    
    print(f"\n{name}:")
    print(f"  Result: {result}")
    print(f"  Confidence: {confidence:.1f}%")
    if (name == "Bank of America Phishing" or name == "PayPal Phishing") and pred == 1:
        print("  ✓ CORRECT - Detected as phishing")
    elif name == "Safe Email" and pred == 0:
        print("  ✓ CORRECT - Detected as safe")
    else:
        print("  ✗ INCORRECT - Wrong classification")
