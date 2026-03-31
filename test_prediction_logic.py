
import re
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS, SARCASM_MARKERS, flip_sentiment

def test_keyword_detection(text):
    print(f"\nTesting: '{text}'")
    # Clean words for matching (logic from app.py)
    words_raw = text.lower().split()
    words_clean = [re.sub(r'[^a-z0-9]', '', w) for w in words_raw]
    words_clean = [w for w in words_clean if w]
    
    found_pos = [w for w in words_clean if w in POSITIVE_KEYWORDS]
    found_neg = [w for w in words_clean if w in NEGATIVE_KEYWORDS]
    found_sarc = [w for w in words_clean if w in SARCASM_MARKERS]
    
    print(f"Cleaned words: {words_clean}")
    print(f"Positive: {found_pos}")
    print(f"Negative: {found_neg}")
    print(f"Sarcasm: {found_sarc}")

def test_sarcasm_flip():
    print("\nTesting Sarcasm Flip:")
    print(f"Positive + Sarcastic -> {flip_sentiment('Positive')}")
    print(f"Negative + Sarcastic -> {flip_sentiment('Negative')}")
    print(f"Neutral + Sarcastic -> {flip_sentiment('Neutral')}")

if __name__ == "__main__":
    test_keyword_detection("Oh genius! This stock is going to the moon! 🚀")
    test_keyword_detection("Just brilliant... my account is red.")
    test_sarcasm_flip()
