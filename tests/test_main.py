"""
Unit tests for NLP-Sentiment-Analyzer
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analyzer import SentimentAnalyzer, create_sample_data


@pytest.fixture
def analyzer():
    """Create a fresh SentimentAnalyzer instance."""
    return SentimentAnalyzer()


@pytest.fixture
def trained_analyzer(analyzer):
    """Create a trained SentimentAnalyzer instance."""
    texts, labels = create_sample_data()
    analyzer.train_custom_model(texts, labels)
    return analyzer


class TestPreprocessText:
    """Tests for preprocess_text()."""

    def test_lowercases_text(self, analyzer):
        result = analyzer.preprocess_text("HELLO WORLD TODAY")
        assert result == result.lower()

    def test_removes_urls(self, analyzer):
        result = analyzer.preprocess_text("visit https://example.com today")
        assert "https" not in result
        assert "example" not in result

    def test_removes_mentions_and_hashtags(self, analyzer):
        result = analyzer.preprocess_text("hello @user and #topic stuff")
        assert "@user" not in result
        assert "#topic" not in result

    def test_removes_special_characters(self, analyzer):
        result = analyzer.preprocess_text("price is $100! great deal!!!")
        assert "$" not in result
        assert "!" not in result

    def test_removes_stopwords(self, analyzer):
        result = analyzer.preprocess_text("this is a very good product")
        # 'this', 'is', 'a' are stopwords and should be removed
        words = result.split()
        assert "this" not in words
        assert "is" not in words

    def test_removes_short_words(self, analyzer):
        result = analyzer.preprocess_text("I am so ok and it is an amazing thing")
        words = result.split()
        for word in words:
            assert len(word) > 2


class TestTextBlobSentiment:
    """Tests for textblob_sentiment()."""

    def test_returns_tuple(self, analyzer):
        result = analyzer.textblob_sentiment("I love this product")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_positive_text(self, analyzer):
        sentiment, polarity = analyzer.textblob_sentiment("I love this, it is amazing and wonderful!")
        assert sentiment == "positive"
        assert polarity > 0.1

    def test_negative_text(self, analyzer):
        sentiment, polarity = analyzer.textblob_sentiment("This is terrible, awful and horrible!")
        assert sentiment == "negative"
        assert polarity < -0.1

    def test_polarity_range(self, analyzer):
        _, polarity = analyzer.textblob_sentiment("Some neutral text about things")
        assert -1.0 <= polarity <= 1.0


class TestTrainCustomModel:
    """Tests for train_custom_model()."""

    def test_training_sets_trained_flag(self, analyzer):
        assert analyzer.trained is False
        texts, labels = create_sample_data()
        analyzer.train_custom_model(texts, labels)
        assert analyzer.trained is True

    def test_training_returns_accuracy(self, analyzer):
        texts, labels = create_sample_data()
        accuracy = analyzer.train_custom_model(texts, labels)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestPredictSentiment:
    """Tests for predict_sentiment() after training."""

    def test_returns_tuple(self, trained_analyzer):
        result = trained_analyzer.predict_sentiment("I love this product")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_valid_label(self, trained_analyzer):
        sentiment, _ = trained_analyzer.predict_sentiment("This is great!")
        assert sentiment in ("positive", "negative", "neutral")

    def test_returns_probability(self, trained_analyzer):
        _, probability = trained_analyzer.predict_sentiment("Wonderful experience")
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0

    def test_falls_back_to_textblob_when_untrained(self, analyzer):
        assert analyzer.trained is False
        result = analyzer.predict_sentiment("I love this")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestAnalyzeBatch:
    """Tests for analyze_batch()."""

    def test_returns_list(self, analyzer):
        texts = ["Good product", "Bad product", "Okay product"]
        results = analyzer.analyze_batch(texts)
        assert isinstance(results, list)

    def test_returns_correct_length(self, analyzer):
        texts = ["Text one", "Text two", "Text three", "Text four"]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 4

    def test_result_has_expected_keys(self, analyzer):
        texts = ["I love this"]
        results = analyzer.analyze_batch(texts)
        assert "text" in results[0]
        assert "sentiment" in results[0]
        assert "confidence" in results[0]

    def test_batch_with_trained_model(self, trained_analyzer):
        texts = ["Great!", "Terrible!", "Fine."]
        results = trained_analyzer.analyze_batch(texts)
        assert len(results) == 3
        for result in results:
            assert result["sentiment"] in ("positive", "negative", "neutral")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
