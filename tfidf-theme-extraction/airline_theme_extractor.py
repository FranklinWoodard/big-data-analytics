"""
airline_theme_extractor.py
CS469 Big Data Analytics — Unit 2 Individual Project
Author: Franklin Woodard
Description: Extracts themes from airline tweets using TF-IDF.
             Outputs original dataset with a 'top_themes' column appended.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "Tweets.csv"
OUTPUT_CSV = "Tweets_with_themes.csv"
TOP_N      = 3        # themes to extract per tweet
MAX_TERMS  = 5000     # vocabulary size for TF-IDF

# ── Preprocessing ─────────────────────────────────────────────────────────────
STOP_WORDS = [
    # Standard English stopwords
    "i","me","my","we","our","you","your","he","she","it","its","they","them",
    "what","which","who","this","that","these","those","am","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","not","no","nor",
    "so","yet","both","either","neither","just","than","too","very","also",
    "but","and","or","if","then","because","as","until","while","of","at",
    "by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","once","here","there","when","where",
    "why","how","all","each","every","both","more","most","other","some","such",
    "a","an","the","s","re","ve","ll","d","t","m",
    # Airline-tweet-specific noise
    "virginamerica","united","usairways","southwestair","americanair",
    "jetblue","united","flight","airline","plane","amp","http","https",
    "co","rt","via","get","got","im","dont","cant","didnt","ive",
    "thank","thanks","please","hi","hey","yes","no","lol","one","us",
    "go","going","back","still","make","made","time","now","today","need",
]

def clean(text):
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)          # remove @mentions
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)     # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8")
print(f"  {len(df):,} tweets loaded")

df["_clean_text"] = df["text"].apply(clean)

# Drop rows where cleaning left nothing
valid_mask = df["_clean_text"].str.strip().str.len() > 0
df_valid   = df[valid_mask].copy()
print(f"  {len(df_valid):,} tweets after cleaning")

# ── TF-IDF ────────────────────────────────────────────────────────────────────
print("Fitting TF-IDF model...")
vectorizer = TfidfVectorizer(
    max_features=MAX_TERMS,
    stop_words=STOP_WORDS,
    ngram_range=(1, 2),      # unigrams + bigrams catch phrases like "long delay"
    min_df=2,                # ignore terms appearing in fewer than 2 tweets
    sublinear_tf=True,       # dampen high-frequency term counts
)

tfidf_matrix = vectorizer.fit_transform(df_valid["_clean_text"])
feature_names = vectorizer.get_feature_names_out()

# ── Extract top N themes per tweet ────────────────────────────────────────────
print(f"Extracting top {TOP_N} themes per tweet...")

def get_themes(row_vector):
    scores = row_vector.toarray().flatten()
    top_idx = scores.argsort()[::-1][:TOP_N]
    themes  = [feature_names[i] for i in top_idx if scores[i] > 0]
    return " | ".join(themes) if themes else "general"

themes = []
for i in range(tfidf_matrix.shape[0]):
    themes.append(get_themes(tfidf_matrix[i]))

df_valid["top_themes"] = themes

# Re-merge back to original df (rows that were empty get "no_text")
df = df.merge(
    df_valid[["tweet_id", "top_themes"]],
    on="tweet_id",
    how="left"
)
df["top_themes"] = df["top_themes"].fillna("no_text")

# ── Save ──────────────────────────────────────────────────────────────────────
df.drop(columns=["_clean_text"], errors="ignore", inplace=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nDone. Output written to: {OUTPUT_CSV}")

# ── Quick summary ─────────────────────────────────────────────────────────────
print("\n── Sample output (10 rows) ──────────────────────────────────────────")
sample = df[["airline_sentiment", "airline", "text", "top_themes"]].head(10)
for _, row in sample.iterrows():
    print(f"  [{row['airline_sentiment']:8s}] {row['airline']:15s} | {row['top_themes']}")
    print(f"           Tweet: {str(row['text'])[:80]}")
    print()

# Theme frequency across full dataset
print("── Most common theme terms across all tweets ────────────────────────")
from collections import Counter
all_terms = []
for t in df["top_themes"]:
    all_terms.extend([x.strip() for x in t.split("|")])
top_20 = Counter(all_terms).most_common(20)
for term, count in top_20:
    print(f"  {count:5d}  {term}")
