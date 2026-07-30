from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
from email.utils import parsedate_to_datetime
import logging

app = FastAPI()

# Logging setup
logging.basicConfig(filename='spam_classifier.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset and train model at cold start. If mail_data.csv is missing, the endpoints will fail gracefully.
try:
    data = pd.read_csv("mail_data.csv")
    if list(data.columns)[:2] != ['Label', 'Email']:
        # Try to handle common variants
        data.columns = [data.columns[0], data.columns[1]]
    data = data.rename(columns={data.columns[0]: 'Label', data.columns[1]: 'Email'})
    data.dropna(subset=["Email", "Label"], inplace=True)
    data['Label'] = data['Label'].map({'spam': 1, 'ham': 0}).fillna(0).astype(int)
except Exception as e:
    logging.error(f"Dataset loading failed: {e}")
    data = None


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    return text

vectorizer = None
model = None
model_accuracy = 0.0

if data is not None:
    try:
        data['Email'] = data['Email'].apply(preprocess_text)
        X_train, X_test, y_train, y_test = train_test_split(
            data['Email'], data['Label'], test_size=0.3, stratify=data['Label'], random_state=42
        )
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_test_vectorized)
        model_accuracy = float(accuracy_score(y_test, y_pred))
        logging.info(f"Model trained. Accuracy: {model_accuracy:.4f}")
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        vectorizer = None
        model = None

# In-memory storage for scanned emails while the server instance is warm.
spam_emails = []
inbox_emails = []


def get_email_date(raw_date):
    try:
        return parsedate_to_datetime(raw_date).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "Unknown Date"


def decode_email_header(header):
    if not header:
        return "No Subject/Sender"
    decoded_parts = decode_header(header)
    result = ""
    for part, encoding in decoded_parts:
        try:
            if isinstance(part, bytes):
                result += part.decode(encoding or 'utf-8', errors='ignore')
            else:
                result += part
        except Exception:
            result += str(part)
    return result


@app.post('/scan')
async def scan(request: Request):
    """Scan the IMAP inbox credentials provided in form-data and return classified emails.
    Accepts form-encoded fields: email, password, imap
    """
    global spam_emails, inbox_emails, vectorizer, model, model_accuracy
    form = await request.form()
    email_addr = form.get('email')
    password = form.get('password')
    imap_server = form.get('imap')

    spam_emails = []
    inbox_emails = []

    if not all([email_addr, password, imap_server]):
        return JSONResponse({ 'status': 'error', 'message': 'Missing email, password, or imap server' })

    if vectorizer is None or model is None:
        return JSONResponse({ 'status': 'error', 'message': 'Model not available on server' })

    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_addr, password)
        mail.select('inbox')
        result, data = mail.search(None, 'ALL')
        email_ids = data[0].split() if data and data[0] else []

        if not email_ids:
            return JSONResponse({'status': 'success', 'message': 'No emails found in inbox', 'stats': {'total': 0, 'spam': 0, 'ham': 0}, 'inbox': [], 'spam': []})

        for email_id in email_ids[-50:]:
            res, msg_data = mail.fetch(email_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_email_header(msg.get('Subject'))
                    sender = decode_email_header(msg.get('From'))
                    sent_time = get_email_date(msg.get('Date'))

                    body = ''
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                try:
                                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                except Exception:
                                    body = str(part.get_payload())
                                break
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                        except Exception:
                            body = str(msg.get_payload())

                    text_to_analyze = f"{subject} {sender} {body[:200]}"
                    email_vectorized = vectorizer.transform([preprocess_text(text_to_analyze)])
                    prediction = int(model.predict(email_vectorized)[0])
                    confidence = float(model.predict_proba(email_vectorized)[0].max() * 100) if hasattr(model, 'predict_proba') else None

                    email_info = {
                        'id': email_id.decode() if isinstance(email_id, bytes) else str(email_id),
                        'subject': subject,
                        'sender': sender,
                        'body_preview': body[:100],
                        'confidence': float(confidence) if confidence is not None else None,
                        'timestamp': sent_time,
                        'is_spam': bool(prediction)
                    }
                    inbox_emails.append(email_info)
                    if prediction == 1:
                        spam_emails.append(email_info)

        mail.logout()
        stats = { 'total': len(inbox_emails), 'spam': len(spam_emails), 'ham': len(inbox_emails) - len(spam_emails) }
        logging.info(f"Processed {stats['total']} emails, found {stats['spam']} spam")
        return JSONResponse({ 'status': 'success', 'message': 'Scan completed successfully', 'stats': stats, 'inbox': inbox_emails, 'spam': spam_emails, 'accuracy': model_accuracy })

    except Exception as e:
        logging.error(f"Scan failed: {e}")
        return JSONResponse({ 'status': 'error', 'message': str(e) })


@app.get('/view_inbox')
async def view_inbox():
    return JSONResponse({ 'status': 'success', 'inbox': inbox_emails })


@app.get('/view_spam')
async def view_spam():
    return JSONResponse({ 'status': 'success', 'spam': spam_emails })


@app.post('/delete_spam')
async def delete_spam(request: Request):
    global spam_emails, inbox_emails
    form = await request.form()
    email_addr = form.get('email')
    password = form.get('password')
    imap_server = form.get('imap')

    if not spam_emails:
        return JSONResponse({ 'status': 'success', 'message': 'No spam emails to delete' })

    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_addr, password)
        mail.select('inbox')

        for email_info in spam_emails:
            try:
                mail.store(email_info['id'], '+FLAGS', '\\Deleted')
            except Exception:
                # continue deleting other messages even if one fails
                logging.warning(f"Failed to mark {email_info.get('id')} for deletion")

        mail.expunge()
        mail.logout()

        inbox_emails = [email for email in inbox_emails if not email['is_spam']]
        spam_emails = []
        stats = {'total': len(inbox_emails), 'spam': 0, 'ham': len(inbox_emails)}
        logging.info('Spam emails deleted successfully')
        return JSONResponse({ 'status': 'success', 'message': 'All spam emails deleted!', 'stats': stats, 'inbox': inbox_emails })

    except Exception as e:
        logging.error(f"Deletion failed: {e}")
        return JSONResponse({ 'status': 'error', 'message': str(e) })
