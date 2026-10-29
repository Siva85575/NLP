import os
import pickle
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'predictions.db')

# Load trained ML model & TF-IDF vectorizer
model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'), 'rb'))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, review TEXT, sentiment TEXT,
        positive REAL, neutral REAL, negative REAL, created_at TEXT)''')
    return conn

def analyze(review_text):
    probabilities = model.predict_proba(vectorizer.transform([review_text]))[0]
    classes = list(model.classes_)
    probs = {c: round(float(p) * 100, 2) for c, p in zip(classes, probabilities)}
    label = classes[probabilities.argmax()].capitalize()
    emoji = {'Positive': '😊', 'Negative': '😢'}.get(label, '😐')
    return label, emoji, probs

def get_page_context(conn, start=None, end=None):
    where_clause, params = '', []
    if start:
        where_clause += ' AND DATE(created_at) >= ?'
        params.append(start)
    if end:
        where_clause += ' AND DATE(created_at) <= ?'
        params.append(end)
    where = (' WHERE 1=1' + where_clause) if where_clause else ''

    history = [dict(row) for row in conn.execute('SELECT * FROM predictions' + where + ' ORDER BY id DESC', params)]
    counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for row in conn.execute('SELECT sentiment, COUNT(*) AS cnt FROM predictions' + where + ' GROUP BY sentiment', params):
        counts[row['sentiment'].capitalize()] = row['cnt']
    daily = [dict(row) for row in conn.execute(
        'SELECT DATE(created_at) AS day, COUNT(*) AS count FROM predictions' + where +
        ' GROUP BY DATE(created_at) ORDER BY DATE(created_at)', params)]

    analytics = {'total': len(history), 'sentiment_counts': counts, 'daily_counts': daily}
    latest = conn.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 1').fetchone()

    result = {}
    if latest:
        latest_dict = dict(latest)
        result = {
            'prediction_text': latest_dict['sentiment'],
            'review': latest_dict['review'],
            'emoji': {'Positive': '😊', 'Negative': '😢'}.get(latest_dict['sentiment'], '😐'),
            'positive': latest_dict['positive'],
            'neutral': latest_dict['neutral'],
            'negative': latest_dict['negative']
        }
    return {**result, 'history': history, 'analytics': analytics}

@app.route('/')
def home():
    conn = get_db()
    data = get_page_context(conn)
    conn.close()
    return render_template('index.html', **data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))
    review = request.form.get('review', '').strip()
    if not review:
        return jsonify({'error': 'Please enter a review.'}), 400

    sentiment, emoji, probs = analyze(review)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db()
    conn.execute(
        'INSERT INTO predictions (review, sentiment, positive, neutral, negative, created_at) VALUES (?,?,?,?,?,?)',
        (review, sentiment, probs['Positive'], probs['Neutral'], probs['Negative'], now)
    )
    conn.commit()
    data = get_page_context(conn)
    conn.close()

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        data.update({'sentiment': sentiment, 'emoji': emoji, 'review': review,
                     'positive': probs['Positive'], 'neutral': probs['Neutral'], 'negative': probs['Negative']})
        return jsonify(data)
    return render_template('index.html', **data)

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    reviews = [r.strip() for r in request.form.get('reviews', '').splitlines() if r.strip()]
    if not reviews:
        return jsonify({'error': 'Please enter at least one statement.'}), 400

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db()
    results = []
    for r in reviews:
        s, e, p = analyze(r)
        cur = conn.execute(
            'INSERT INTO predictions (review, sentiment, positive, neutral, negative, created_at) VALUES (?,?,?,?,?,?)',
            (r, s, p['Positive'], p['Neutral'], p['Negative'], now)
        )
        results.append({'id': cur.lastrowid, 'review': r, 'sentiment': s, 'emoji': e, 'positive': p['Positive'], 'neutral': p['Neutral'], 'negative': p['Negative']})
    conn.commit()
    data = get_page_context(conn)
    data['results'] = results
    conn.close()
    return jsonify(data)

@app.route('/history')
def history():
    conn = get_db()
    data = get_page_context(conn, request.args.get('start'), request.args.get('end'))
    conn.close()
    return jsonify({'history': data['history'], 'analytics': data['analytics']})

@app.route('/report')
def report():
    from report import build_report
    start = request.args.get('start')
    end = request.args.get('end')
    conn = get_db()
    where_clause, params = '', []
    if start:
        where_clause += ' AND DATE(created_at) >= ?'
        params.append(start)
    if end:
        where_clause += ' AND DATE(created_at) <= ?'
        params.append(end)
    where = (' WHERE 1=1' + where_clause) if where_clause else ''
    items = [dict(r) for r in conn.execute('SELECT * FROM predictions' + where + ' ORDER BY id DESC', params)]
    if not items:
        items = [dict(r) for r in conn.execute('SELECT * FROM predictions ORDER BY id DESC')]
    conn.close()
    pdf_bytes = build_report(items, start, end)
    return Response(pdf_bytes, mimetype='application/pdf', headers={'Content-Disposition': 'attachment; filename=sentiment_report.pdf'})

@app.route('/report_bulk')
def report_bulk():
    from report import build_report
    ids = [int(i) for i in request.args.get('ids', '').split(',') if i.isdigit()]
    conn = get_db()
    if ids:
        placeholders = ','.join(['?'] * len(ids))
        items = [dict(r) for r in conn.execute(f'SELECT * FROM predictions WHERE id IN ({placeholders}) ORDER BY id DESC', ids)]
    else:
        items = []
    if not items:
        items = [dict(r) for r in conn.execute('SELECT * FROM predictions ORDER BY id DESC')]
    conn.close()
    pdf_bytes = build_report(items)
    return Response(pdf_bytes, mimetype='application/pdf', headers={'Content-Disposition': 'attachment; filename=batch_sentiment_report.pdf'})


@app.route('/delete/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    conn = get_db()
    conn.execute('DELETE FROM predictions WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()
    return jsonify({'ok': True})

@app.route('/delete_selected', methods=['POST'])
def delete_selected():
    ids = [i for i in request.form.getlist('ids') if i.isdigit()]
    if not ids:
        return jsonify({'error': 'Please select at least one record.'}), 400
    conn = get_db()
    conn.execute('DELETE FROM predictions WHERE id IN (%s)' % ','.join(['?'] * len(ids)), ids)
    conn.commit()
    conn.close()
    return jsonify({'ok': True})

@app.route('/delete_all', methods=['POST'])
def delete_all():
    conn = get_db()
    conn.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return jsonify({'ok': True})

if __name__ == '__main__':
    import threading, webbrowser
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=True)

