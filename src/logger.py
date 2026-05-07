import logging
import os
from datetime import datetime

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "project.log")

def get_logger(name="reddit_nlp"):
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # file handler — saves everything to logs/project.log
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # console handler — shows INFO and above in terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def log_step(step, message, level="info"):
    logger = get_logger()
    msg    = f"[{step}] {message}"
    getattr(logger, level)(msg)

def log_dataframe(df, label="DataFrame"):
    logger = get_logger()
    logger.info(f"{label} — shape: {df.shape}")
    logger.info(f"{label} — columns: {list(df.columns)}")
    if "subreddit" in df.columns:
        logger.info(f"{label} — subreddits: {df['subreddit'].value_counts().to_dict()}")