import re
import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from emotion_tool.config import EMOTIONS, POSITIVE_THRESH, NEGATIVE_THRESH

sia = SentimentIntensityAnalyzer()

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s.!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def score_emotions(text):
    """Score each emotion category by keyword hit rate."""
    words  = set(word_tokenize(clean(text)))
    scores = {}
    for emotion, keywords in EMOTIONS.items():
        hits         = len(words & set(keywords))
        scores[emotion] = round(hits / max(len(words), 1), 4)
    return scores

def score_sentence_sentiment(text):
    """Return VADER compound score per sentence."""
    sentences = sent_tokenize(text)
    results   = []
    for sent in sentences:
        score = sia.polarity_scores(sent)
        results.append({
            "sentence":  sent,
            "compound":  score["compound"],
            "pos":       score["pos"],
            "neg":       score["neg"],
            "neu":       score["neu"],
        })
    return results

def dominant_emotion(emotion_scores):
    """Return the emotion with the highest score."""
    if not any(emotion_scores.values()):
        return "neutral"
    return max(emotion_scores, key=emotion_scores.get)

def analyze(text):
    """Full analysis of a user text — returns a rich dict."""
    cleaned   = clean(text)
    vader     = sia.polarity_scores(cleaned)
    emotions  = score_emotions(text)
    sentences = score_sentence_sentiment(text)
    dominant  = dominant_emotion(emotions)

    if vader["compound"] >= POSITIVE_THRESH:
        overall_mood = "positive"
    elif vader["compound"] <= NEGATIVE_THRESH:
        overall_mood = "negative"
    else:
        overall_mood = "neutral"

    return {
        "original":     text,
        "cleaned":      cleaned,
        "word_count":   len(cleaned.split()),
        "vader":        vader,
        "overall_mood": overall_mood,
        "emotions":     emotions,
        "dominant":     dominant,
        "sentences":    sentences,
    }