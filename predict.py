from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import re
import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import logging
import csv
from typing import List

app = FastAPI()

# Logging setup
logging.basicConfig(filename='spam_classifier.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Simple rule-based spam keyword list (lightweight alternative to sklearn model for serverless)
SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'prize', 'click', 'buy', 'urgent', 'credit', 'offer',
    'cheap', 'cash', 'loan', 'subscribe', 'winner', 'claim', 'congratulations', 'guarantee'
]

# In-memory storage for scanned emails while the server instance is warm.
spam_emails: List[dict] = []
inbox_emails: List[dict] = []
model_accuracy: float = 0.0


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", '', text)
    return text


def rule_based_predict(text: str) -> (int, float):
    """Return (prediction, confidence)
    prediction: 1 for spam, 0 for ham
    confidence: 0-100
    """
    txt = preprocess_text(text)
    count = 0
    for kw in SPAM_KEYWORDS:
        if kw in txt:
            count += txt.count(kw)
    # More signals (simple heuristics)
    if any(p in txt for p in ['http://', 'https://']):
        count += 1
    if any(char.isdigit() for char in txt) and len(re.findall(r"\d{5,}", txt)):
        count += 1

    if count == 0:
        return 0, 25.0
    # confidence scaled
    confidence = min(95.0, 30.0 + count * 15.0)
    return 1, confidence


# Try to compute a rough accuracy against mail_data.csv using the same heuristic.
try:
    total = 0
    correct = 0
    with open('mail_data.csv', newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        # Find label/text columns
        # Common formats: Label,Email or label,text etc.
        if header and len(header) >= 2 and ('label' in header[0].lower() or 'label' in header[1].lower()):
            # header exists, determine indices
            label_idx = 0
            text_idx = 1
            # try to find exact
            for i, h in enumerate(header):
                if 'label' in h.lower():
                    label_idx = i
                if 'email' in h.lower() or 'text' in h.lower() or 'body' in h.lower():
                    text_idx = i
        else:
            # assume first two columns are label and email
            label_idx = 0
            text_idx = 1
            # if header looks like data, treat it as row
            if header is not None and len(header) >= 2 and header[0].lower() in ('spam', 'ham'):
                # first row is data
                try:
                    lbl = header[0]
                    txt = header[1]
                    pred, _ = rule_based_predict(txt)
                    actual = 1 if lbl.strip().lower() == 'spam' else 0
                    total += 1
                    correct += 1 if pred == actual else 0
                except Exception:
                    pass

        for row in reader:
            if len(row) <= max(label_idx, text_idx):
                continue
            label = row[label_idx].strip().lower()
            text = row[text_idx]
            pred, _ = rule_based_predict(text)
            actual = 1 if label == 'spam' else 0
            total += 1
            if pred == actual:
                correct += 1
    if total > 0:
        model_accuracy = (correct / total)
    else:
        model_accuracy = 0.0
    logging.info(f"Rule-based model accuracy on provided dataset: {model_accuracy:.4f} (computed at cold-start)")
except FileNotFoundError:
    logging.warning('mail_data.csv not found; model_accuracy set to 0')
    model_accuracy = 0.0
except Exception as e:
    logging.error(f"Failed to compute dataset accuracy: {e}")
    model_accuracy = 0.0


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
    global spam_emails, inbox_emails, model_accuracy
    form = await request.form()
    email_addr = form.get('email')
    password = form.get('password')
    imap_server = form.get('imap')

    spam_emails = []
    inbox_emails = []

    if not all([email_addr, password, imap_server]):
        return JSONResponse({ 'status': 'error', 'message': 'Missing email, password, or imap server' })

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
                    prediction, confidence = rule_based_predict(text_to_analyze)

                    email_info = {
                        'id': email_id.decode() if isinstance(email_id, bytes) else str(email_id),
                        'subject': subject,
                        'sender': sender,
                        'body_preview': body[:100],
                        'confidence': float(confidence),
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
