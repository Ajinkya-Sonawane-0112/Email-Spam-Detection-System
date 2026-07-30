from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime  # Import for real email timestamp
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')

# Logging setup
logging.basicConfig(filename='spam_classifier.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
try:
    data = pd.read_csv("mail_data.csv")
    data.columns = ["Label", "Email"]
    data.dropna(subset=["Email", "Label"], inplace=True)
    data["Label"] = data["Label"].map({"spam": 1, "ham": 0})
except Exception as e:
    logging.error(f"Dataset loading failed: {e}")
    exit()

# Preprocess email text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

data["Email"] = data["Email"].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data["Email"], data["Label"], test_size=0.3, stratify=data["Label"], random_state=42
)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)
model_accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model accuracy: {model_accuracy:.2f}")

spam_emails = []
inbox_emails = []

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Function to get actual sent time
def get_email_date(raw_date):
    try:
        return parsedate_to_datetime(raw_date).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "Unknown Date"

@app.route('/scan', methods=['POST'])
def scan_emails():
    global spam_emails, inbox_emails
    spam_emails.clear()
    inbox_emails.clear()

    email_addr = request.form['email']
    password = request.form['password']
    imap_server = request.form['imap']

    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_addr, password)
        mail.select("inbox")
        result, data = mail.search(None, "ALL")
        email_ids = data[0].split()

        if not email_ids:
            return jsonify({'status': 'success', 'message': 'No emails found in inbox', 'stats': {'total': 0, 'spam': 0, 'ham': 0}, 'inbox': [], 'spam': []})

        total_emails = min(len(email_ids), 50)
        for i, email_id in enumerate(email_ids[-50:]):
            res, msg_data = mail.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_email_header(msg["Subject"])
                    sender = decode_email_header(msg["From"])
                    sent_time = get_email_date(msg["Date"])  # Get actual sent time

                    # Extract email body (simplified)
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                break
                    else:
                        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

                    text_to_analyze = f"{subject} {sender} {body[:200]}"
                    email_vectorized = vectorizer.transform([preprocess_text(text_to_analyze)])
                    prediction = model.predict(email_vectorized)[0]
                    confidence = model.predict_proba(email_vectorized)[0][prediction] * 100

                    email_info = {
                        'id': email_id.decode(),
                        'subject': subject,
                        'sender': sender,
                        'body_preview': body[:100],
                        'confidence': confidence,
                        'timestamp': sent_time,  # Show real sent time
                        'is_spam': bool(prediction)
                    }
                    inbox_emails.append(email_info)
                    if prediction == 1:
                        spam_emails.append(email_info)

        mail.logout()
        stats = {
            'total': len(inbox_emails),
            'spam': len(spam_emails),
            'ham': len(inbox_emails) - len(spam_emails)
        }
        logging.info(f"Processed {stats['total']} emails, found {stats['spam']} spam")
        return jsonify({
            'status': 'success',
            'message': "Scan completed successfully",
            'stats': stats,
            'inbox': inbox_emails,
            'spam': spam_emails,
            'accuracy': model_accuracy
        })

    except Exception as e:
        logging.error(f"Scan failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/view_inbox', methods=['GET'])
def view_inbox():
    return jsonify({'status': 'success', 'inbox': inbox_emails})

@app.route('/view_spam', methods=['GET'])
def view_spam():
    return jsonify({'status': 'success', 'spam': spam_emails})

@app.route('/delete_spam', methods=['POST'])
def delete_spam():
    global spam_emails, inbox_emails
    if not spam_emails:
        return jsonify({'status': 'success', 'message': 'No spam emails to delete'})

    email_addr = request.form['email']
    password = request.form['password']
    imap_server = request.form['imap']

    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_addr, password)
        mail.select("inbox")

        for email_info in spam_emails:
            mail.store(email_info['id'], "+FLAGS", "\\Deleted")

        mail.expunge()
        mail.logout()

        inbox_emails = [email for email in inbox_emails if not email['is_spam']]
        spam_emails.clear()
        stats = {
            'total': len(inbox_emails),
            'spam': 0,
            'ham': len(inbox_emails)
        }
        logging.info("Spam emails deleted successfully")
        return jsonify({'status': 'success', 'message': 'All spam emails deleted!', 'stats': stats, 'inbox': inbox_emails})

    except Exception as e:
        logging.error(f"Deletion failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

def decode_email_header(header):
    if not header:
        return "No Subject/Sender"
    decoded_parts = decode_header(header)
    result = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            result += part.decode(encoding or 'utf-8', errors='ignore')
        else:
            result += part
    return result

if __name__ == '__main__':
    app.run(debug=True)
