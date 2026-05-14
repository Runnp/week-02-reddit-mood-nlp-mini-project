import numpy as np
from preprocess import clean_text

class NLPEngine:
    def __init__(self, sk_model=None, lstm_model=None, tokenizer_data=None):
        self.sk = sk_model
        self.lstm = lstm_model
        self.tokenizer_data = tokenizer_data

    # ─────────────────────────────
    # TEXT PIPELINE
    # ─────────────────────────────
    def preprocess(self, text):
        return clean_text(text)

    # ─────────────────────────────
    # SKLEARN PREDICTION
    # ─────────────────────────────
    def predict_sklearn(self, text):
        if self.sk is None:
            return None

        cleaned = self.preprocess(text)

        vec = self.sk["vectorizer"].transform([cleaned])
        pred = self.sk["model"].predict(vec)[0]
        proba = self.sk["model"].predict_proba(vec)[0]

        return {
            "label": self.sk["classes"][pred],
            "confidence": dict(zip(self.sk["classes"], proba))
        }

    # ─────────────────────────────
    # LSTM PREDICTION
    # ─────────────────────────────
    def predict_lstm(self, text):
        if self.lstm is None or self.tokenizer_data is None:
            return None

        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            return None

        cleaned = self.preprocess(text)

        seq = self.tokenizer_data["tokenizer"].texts_to_sequences([cleaned])
        pad = pad_sequences(
            seq,
            maxlen=self.tokenizer_data["max_len"],
            padding="post",
            truncating="post"
        )

        prob = self.lstm.predict(pad, verbose=0)[0]
        pred = np.argmax(prob)

        return {
            "label": self.tokenizer_data["classes"][pred],
            "confidence": dict(zip(self.tokenizer_data["classes"], prob))
        }

    # ─────────────────────────────
    # COMBINED PREDICTION
    # ─────────────────────────────
    def predict(self, text):
        sk_res = self.predict_sklearn(text)
        lstm_res = self.predict_lstm(text)

        return {
            "sklearn": sk_res,
            "lstm": lstm_res,
            "agreement": (
                sk_res is not None and lstm_res is not None
                and sk_res["label"] == lstm_res["label"]
            )
        }