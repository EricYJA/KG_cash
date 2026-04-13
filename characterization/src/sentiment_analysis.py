# characterization/src/sentiment_analysis.py
import numpy as np
import logging
from transformers import pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize a SOTA BERT/RoBERTa-based sentiment pipeline
# Using a model fine-tuned for negative, neutral, and positive sentiment
sentiment_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,  # Returns probabilities for all classes
    device=-1  # Use 0 if you have a GPU available
)


def get_sentiment_vector(text):
    """
    Generates a sentiment vector for a given text using a RoBERTa model.
    Returns a vector of probabilities: [Negative, Neutral, Positive]
    """
    # The pipeline with top_k=None returns all scores
    # Example output: [{'label': 'negative', 'score': 0.8}, {'label': 'neutral', 'score': 0.15}, ...]
    try:
        results = sentiment_pipeline(text)[0]
        scores = {res['label']: res['score'] for res in results}

        neg = scores.get('negative', 0.0)
        neu = scores.get('neutral', 0.0)
        pos = scores.get('positive', 0.0)

        return np.array([neg, neu, pos])
    except Exception as e:
        # Fallback to zero-vector if processing fails
        return np.array([0.0, 0.0, 0.0])


def extract_questions_from_dataset(dataset, split='train'):
    """Extracts a flat list of actual string questions from the dataset."""
    questions = []
    for item in dataset[split]:
        # Adapt this extraction based on the actual schema of your dataset dictionaries
        q = item.get('question') or item.get('utterance') or item.get('RawQuestion')
        if q:
            questions.append(str(q).strip())
    return questions
