import pandas as pd
import re

print("="*60)
print("ADDING PHISHING EMAILS TO TRAINING DATA")
print("="*60)

# Load existing dataset
print("\n📥 Loading existing SMS dataset...")
df = pd.read_csv('sms.tsv', sep='\t', header=None, names=['label', 'message'])
print(f"Original size: {len(df)} messages")

# Advanced phishing emails dataset
phishing_emails = [
    # PayPal phishing
    ("spam", "URGENT: Your PayPal account has been temporarily limited. Verify your identity now: https://paypal.com.verify-account.com"),
    ("spam", "PayPal: Unusual login detected from unknown device. Confirm your account: http://paypal-secure.net/login"),
    ("spam", "Your PayPal account will be suspended in 24 hours. Click here to restore: https://paypal-resolution.com"),
    
    # Bank phishing
    ("spam", "Bank of America Alert: Suspicious transaction detected. Verify your account immediately: http://bankofamerica-verify.com"),
    ("spam", "Chase Bank: Your online banking has been locked due to multiple failed attempts. Unlock now: https://chase-online-banking.net"),
    ("spam", "Wells Fargo: Unusual activity on your account. Confirm your information: http://wellsfargo-alerts.com"),
    
    # Amazon phishing
    ("spam", "Amazon: Your order #892347 has been cancelled. Verify payment method: https://amazon-payment-verify.com"),
    ("spam", "Amazon Prime membership expired. Renew now to avoid service interruption: http://amazon-prime-renewal.net"),
    ("spam", "Suspicious login to your Amazon account from Russia. Secure your account: https://amazon-security-alert.com"),
    
    # Netflix/Streaming
    ("spam", "Netflix: Your subscription has expired. Update payment info: http://netflix-billing-update.com"),
    ("spam", "Hulu: Account verification required. Click here: https://hulu-account-verify.net"),
    ("spam", "Disney+: Your billing information is out of date. Update now: http://disneyplus-payment.com"),
    
    # Apple/iCloud
    ("spam", "Apple ID: Your account has been disabled. Verify now: https://appleid-verification.com"),
    ("spam", "iCloud storage full. Upgrade now to keep your photos: http://icloud-upgrade.net"),
    
    # Microsoft/Outlook
    ("spam", "Microsoft: Unusual sign-in activity. Review recent activity: https://microsoft-account-security.com"),
    ("spam", "Your Outlook account will be closed. Click here to keep it active: http://outlook-keep-alive.net"),
    
    # Job scams
    ("spam", "We reviewed your resume! You're hired for $85,000/year remote job. Send documents: http://fake-job-offer.com"),
    ("spam", "Google is hiring in your area! $120k starting. Apply here: https://google-careers-fake.com"),
    
    # Lottery/Contest
    ("spam", "Congratulations! You won $5,000,000 in the UK Lottery. Claim now: http://lottery-claim-fake.com"),
    ("spam", "Publishers Clearing House: You are a winner! Click to claim $2M: https://pch-winner-fake.net"),
    
    # Fake invoices
    ("spam", "Your McAfee subscription has been renewed: $349.99. Cancel here: http://mcafee-cancel-fake.com"),
    ("spam", "Invoice #INV-89234 for $499.00 is due. View invoice: https://invoice-payment-fake.com"),
    
    # Romance scams
    ("spam", "I saw your profile and want to connect. Check my photos here: http://dating-profile-fake.com"),
    ("spam", "Hello dear, I'm from UK. I need your help with $5M transfer. Contact me: http://help-transfer-fake.com"),
    
    # Tech support
    ("spam", "Your Windows license has expired. Renew now for $99: http://windows-license-fake.com"),
    ("spam", "Norton Alert: Virus detected on your computer. Scan now: https://norton-scan-fake.net"),
]

# Add to dataframe
for label, msg in phishing_emails:
    df = pd.concat([df, pd.DataFrame([[label, msg]], columns=df.columns)], ignore_index=True)

print(f"\n✅ Added {len(phishing_emails)} advanced phishing examples")
print(f"New dataset size: {len(df)} messages")

# Show distribution
spam_count = df[df['label'] == 'spam'].shape[0]
ham_count = df[df['label'] == 'ham'].shape[0]
print(f"\n📊 New distribution:")
print(f"   Spam: {spam_count}")
print(f"   Ham:  {ham_count}")

# Save updated dataset
df.to_csv('sms.tsv', sep='\t', header=False, index=False)
print("\n💾 Updated dataset saved to sms.tsv")

print("\n🎉 Ready to retrain model with phishing detection!")