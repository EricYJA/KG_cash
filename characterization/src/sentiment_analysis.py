# characterization/src/sentiment_analysis.py
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_sentiment_vector(text):
    """
    Generates a sentence embedding for a given text using all-MiniLM-L6-v2.
    """
    try:
        return model.encode(text)
    except Exception as e:
        # Fallback to zero-vector if processing fails
        return np.zeros(384)


def extract_questions_from_dataset(dataset, split='train'):
    """Extracts a flat list of actual string questions from the dataset."""
    questions = []
    for item in dataset[split]:
        # Adapt this extraction based on the actual schema of your dataset dictionaries
        q = item.get('question') or item.get('utterance') or item.get('RawQuestion')
        if q:
            questions.append(str(q).strip())
    return questions
