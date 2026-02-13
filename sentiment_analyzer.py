#!/usr/bin/env python3
"""
NLP Sentiment Analyzer
Advanced sentiment analysis system with multiple models and web interface.
"""

from flask import Flask, request, jsonify, render_template_string
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SentimentAnalyzer:
    """Advanced sentiment analysis with multiple approaches."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression()
        self.stop_words = set(stopwords.words('english'))
        self.trained = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def textblob_sentiment(self, text):
        """Get sentiment using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    
    def train_custom_model(self, texts, labels):
        """Train custom sentiment model."""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train model
        self.model.fit(X, labels)
        self.trained = True
        
        return self.model.score(X, labels)
    
    def predict_sentiment(self, text):
        """Predict sentiment using custom model."""
        if not self.trained:
            return self.textblob_sentiment(text)
        
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0].max()
        
        return prediction, probability
    
    def analyze_batch(self, texts):
        """Analyze sentiment for multiple texts."""
        results = []
        for text in texts:
            if self.trained:
                sentiment, confidence = self.predict_sentiment(text)
            else:
                sentiment, confidence = self.textblob_sentiment(text)
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': round(confidence, 3)
            })
        
        return results
    
    def save_model(self, filename='sentiment_model.pkl'):
        """Save trained model."""
        if self.trained:
            model_data = {
                'vectorizer': self.vectorizer,
                'model': self.model
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        return False
    
    def load_model(self, filename='sentiment_model.pkl'):
        """Load trained model."""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.trained = True
            return True
        except Exception:
            return False

# Flask Web Application
app = Flask(__name__)
analyzer = SentimentAnalyzer()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NLP Sentiment Analyzer</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        textarea, input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        textarea { height: 120px; resize: vertical; }
        button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .positive { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .negative { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .neutral { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .batch-results { margin-top: 20px; }
        .batch-item { padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #007bff; background: #f8f9fa; }
        .tabs { display: flex; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #e9ecef; border: none; cursor: pointer; margin-right: 5px; border-radius: 5px 5px 0 0; }
        .tab.active { background: #007bff; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 NLP Sentiment Analyzer</h1>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('single')">Single Text</button>
            <button class="tab" onclick="showTab('batch')">Batch Analysis</button>
        </div>
        
        <div id="single" class="tab-content active">
            <form id="singleForm">
                <div class="form-group">
                    <label for="text">Enter text to analyze:</label>
                    <textarea id="text" name="text" placeholder="Type your text here..." required></textarea>
                </div>
                <button type="submit">Analyze Sentiment</button>
            </form>
            <div id="singleResult"></div>
        </div>
        
        <div id="batch" class="tab-content">
            <form id="batchForm">
                <div class="form-group">
                    <label for="batchText">Enter multiple texts (one per line):</label>
                    <textarea id="batchText" name="batchText" placeholder="Text 1&#10;Text 2&#10;Text 3..." required></textarea>
                </div>
                <button type="submit">Analyze Batch</button>
            </form>
            <div id="batchResult"></div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }

        document.getElementById('singleForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const text = document.getElementById('text').value;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('singleResult');
                
                resultDiv.innerHTML = `
                    <div class="result ${result.sentiment}">
                        <strong>Sentiment:</strong> ${result.sentiment.toUpperCase()}<br>
                        <strong>Confidence:</strong> ${result.confidence}
                    </div>
                `;
            } catch (error) {
                console.error('Error:', error);
            }
        });

        document.getElementById('batchForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const texts = document.getElementById('batchText').value.split('\\n').filter(t => t.trim());
            
            try {
                const response = await fetch('/analyze_batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({texts: texts})
                });
                
                const results = await response.json();
                const resultDiv = document.getElementById('batchResult');
                
                let html = '<div class="batch-results"><h3>Batch Analysis Results:</h3>';
                results.forEach(result => {
                    html += `
                        <div class="batch-item">
                            <strong>Text:</strong> ${result.text}<br>
                            <strong>Sentiment:</strong> ${result.sentiment.toUpperCase()} 
                            (Confidence: ${result.confidence})
                        </div>
                    `;
                });
                html += '</div>';
                
                resultDiv.innerHTML = html;
            } catch (error) {
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze single text sentiment."""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if analyzer.trained:
        sentiment, confidence = analyzer.predict_sentiment(text)
        return jsonify({
            'sentiment': sentiment,
            'confidence': confidence
        })
    else:
        sentiment, polarity = analyzer.textblob_sentiment(text)
        return jsonify({
            'sentiment': sentiment,
            'polarity': polarity
        })

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple texts."""
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    results = analyzer.analyze_batch(texts)
    return jsonify(results)

def create_sample_data():
    """Create sample training data."""
    sample_texts = [
        "I love this product! It's amazing!",
        "This is the worst thing I've ever bought.",
        "It's okay, nothing special.",
        "Absolutely fantastic! Highly recommend!",
        "Terrible quality, waste of money.",
        "Pretty good, satisfied with purchase.",
        "Outstanding service and quality!",
        "Not worth the price, disappointed.",
        "Average product, meets expectations.",
        "Excellent! Will buy again!"
    ]
    
    sample_labels = [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'positive', 'negative', 'neutral', 'positive'
    ]
    
    return sample_texts, sample_labels

def main():
    """Main execution function."""
    print("NLP Sentiment Analyzer")
    print("=" * 30)
    
    # Train with sample data
    texts, labels = create_sample_data()
    accuracy = analyzer.train_custom_model(texts, labels)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    
    # Save model
    analyzer.save_model()
    print("Model saved successfully!")
    
    # Test analysis
    test_texts = [
        "I'm so happy with this purchase!",
        "This product is terrible.",
        "It's an okay product."
    ]
    
    print("\nTest Analysis:")
    results = analyzer.analyze_batch(test_texts)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")
        print("-" * 40)
    
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()

