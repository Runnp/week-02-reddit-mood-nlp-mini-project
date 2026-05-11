# Emotion Tool — configuration

# Emotion categories and their keyword signals
EMOTIONS = {
    "sadness": [
        "sad", "cry", "tears", "hopeless", "empty", "alone", "lost",
        "grief", "sorrow", "miserable", "depressed", "heartbroken",
        "worthless", "despair", "numb", "hollow", "broken", "dark",
        "suffer", "pain", "hurt", "exhausted", "tired", "drained",
    ],
    "happiness": [
        "happy", "joy", "excited", "grateful", "love", "wonderful",
        "amazing", "fantastic", "blessed", "proud", "thrilled", "elated",
        "cheerful", "delighted", "smile", "laugh", "celebrate", "win",
        "success", "great", "awesome", "glad", "content", "peaceful",
    ],
    "anxiety": [
        "anxious", "worry", "scared", "fear", "panic", "nervous",
        "stressed", "overwhelmed", "dread", "terrified", "uneasy",
        "restless", "tense", "apprehensive", "paranoid", "helpless",
    ],
    "anger": [
        "angry", "furious", "rage", "hate", "frustrated", "annoyed",
        "bitter", "resentful", "mad", "livid", "outraged", "disgusted",
        "hostile", "irritated", "enraged",
    ],
    "hope": [
        "hope", "better", "improve", "recover", "future", "forward",
        "progress", "healing", "change", "grow", "try", "believe",
        "possible", "will", "someday", "eventually",
    ],
}

# Color per emotion
EMOTION_COLORS = {
    "sadness":   "#6B9FD4",
    "happiness": "#7BC67E",
    "anxiety":   "#F4A261",
    "anger":     "#E07070",
    "hope":      "#C3A6E8",
}

# VADER thresholds
POSITIVE_THRESH =  0.05
NEGATIVE_THRESH = -0.05