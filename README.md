# NLP Sentiment Analyzer

[English](#english) | [Português](#português)

## English

### Overview
Advanced sentiment analysis system with multiple models and web interface. Features TextBlob integration, custom machine learning models, and a professional Flask web application for real-time sentiment analysis.

### Features
- **Multiple Models**: TextBlob and custom ML models
- **Web Interface**: Professional Flask application
- **Batch Processing**: Analyze multiple texts simultaneously
- **Text Preprocessing**: Advanced cleaning and tokenization
- **Model Training**: Train custom models with your data
- **Real-time Analysis**: Instant sentiment prediction
- **REST API**: JSON endpoints for integration

### Technologies Used
- **Python 3.8+**
- **Flask**: Web framework
- **TextBlob**: Natural language processing
- **NLTK**: Text processing toolkit
- **Scikit-learn**: Machine learning
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/galafis/NLP-Sentiment-Analyzer.git
cd NLP-Sentiment-Analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python sentiment_analyzer.py
```

4. Open your browser to `http://localhost:5000`

### Usage

#### Web Interface
- **Single Text Analysis**: Enter text and get instant sentiment analysis
- **Batch Analysis**: Analyze multiple texts at once
- **Real-time Results**: Immediate feedback with confidence scores

#### API Endpoints

**Single Text Analysis**
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

**Batch Analysis**
```bash
curl -X POST http://localhost:5000/analyze_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible service", "It's okay"]}'
```

#### Python API
```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Single analysis
sentiment, confidence = analyzer.textblob_sentiment("I love this!")
print(f"Sentiment: {sentiment}, Confidence: {confidence}")

# Batch analysis
texts = ["Great!", "Bad!", "Okay"]
results = analyzer.analyze_batch(texts)
for result in results:
    print(f"{result['text']}: {result['sentiment']}")
```

### Features

#### Text Preprocessing
- URL and mention removal
- Special character cleaning
- Stopword filtering
- Tokenization

#### Sentiment Models
- **TextBlob**: Rule-based sentiment analysis
- **Custom ML**: Trained logistic regression model
- **Confidence Scores**: Probability estimates

#### Web Interface
- Clean, responsive design
- Tabbed interface for different analysis types
- Real-time results display
- Error handling

### Model Training
Train custom models with your own data:

```python
texts = ["I love this!", "This is terrible!", "It's okay"]
labels = ["positive", "negative", "neutral"]

accuracy = analyzer.train_custom_model(texts, labels)
analyzer.save_model("my_model.pkl")
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Português

### Visão Geral
Sistema avançado de análise de sentimentos com múltiplos modelos e interface web. Apresenta integração com TextBlob, modelos de machine learning personalizados e aplicação Flask profissional para análise de sentimentos em tempo real.

### Funcionalidades
- **Múltiplos Modelos**: TextBlob e modelos ML personalizados
- **Interface Web**: Aplicação Flask profissional
- **Processamento em Lote**: Analise múltiplos textos simultaneamente
- **Pré-processamento**: Limpeza avançada e tokenização
- **Treinamento de Modelo**: Treine modelos personalizados com seus dados
- **Análise em Tempo Real**: Predição instantânea de sentimentos
- **API REST**: Endpoints JSON para integração

### Tecnologias Utilizadas
- **Python 3.8+**
- **Flask**: Framework web
- **TextBlob**: Processamento de linguagem natural
- **NLTK**: Toolkit de processamento de texto
- **Scikit-learn**: Machine learning
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/galafis/NLP-Sentiment-Analyzer.git
cd NLP-Sentiment-Analyzer
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
python sentiment_analyzer.py
```

4. Abra seu navegador em `http://localhost:5000`

### Uso

#### Interface Web
- **Análise de Texto Único**: Digite texto e obtenha análise instantânea
- **Análise em Lote**: Analise múltiplos textos de uma vez
- **Resultados em Tempo Real**: Feedback imediato com scores de confiança

#### Endpoints da API

**Análise de Texto Único**
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Eu amo este produto!"}'
```

**Análise em Lote**
```bash
curl -X POST http://localhost:5000/analyze_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Ótimo produto!", "Serviço terrível", "Está ok"]}'
```

#### API Python
```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Análise única
sentiment, confidence = analyzer.textblob_sentiment("Eu amo isso!")
print(f"Sentimento: {sentiment}, Confiança: {confidence}")

# Análise em lote
texts = ["Ótimo!", "Ruim!", "Ok"]
results = analyzer.analyze_batch(texts)
for result in results:
    print(f"{result['text']}: {result['sentiment']}")
```

### Funcionalidades

#### Pré-processamento de Texto
- Remoção de URLs e menções
- Limpeza de caracteres especiais
- Filtragem de stopwords
- Tokenização

#### Modelos de Sentimento
- **TextBlob**: Análise baseada em regras
- **ML Personalizado**: Modelo de regressão logística treinado
- **Scores de Confiança**: Estimativas de probabilidade

#### Interface Web
- Design limpo e responsivo
- Interface com abas para diferentes tipos de análise
- Exibição de resultados em tempo real
- Tratamento de erros

### Treinamento de Modelo
Treine modelos personalizados com seus próprios dados:

```python
texts = ["Eu amo isso!", "Isso é terrível!", "Está ok"]
labels = ["positive", "negative", "neutral"]

accuracy = analyzer.train_custom_model(texts, labels)
analyzer.save_model("meu_modelo.pkl")
```

### Contribuindo
1. Faça um fork do repositório
2. Crie uma branch de feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adicionar nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

### Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

