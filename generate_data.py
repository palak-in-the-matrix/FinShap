"""
==============================================================================
generate_data.py — Synthetic Dataset Generator
==============================================================================
Generates a realistic financial text dataset simulating Reddit WallStreetBets
posts (1,000,000 rows) when the original CSV is unavailable.

Columns: id, title, score, author, author_flair_text, removed_by,
         total_awards_received, awarders, created_utc, full_link,
         num_comments, over_18

Author: Research Implementation
==============================================================================
"""

import os
import random
import string
import time
import numpy as np
import pandas as pd

# ============================================================================
# TEMPLATE CONFIGURATIONS
# ============================================================================

TICKERS = [
    'GME', 'AMC', 'TSLA', 'PLTR', 'BB', 'NOK', 'AAPL', 'MSFT',
    'NVDA', 'AMD', 'SPY', 'QQQ', 'AMZN', 'GOOGL', 'META', 'NFLX',
    'BABA', 'NIO', 'WISH', 'CLOV', 'SOFI', 'RIVN', 'LCID', 'COIN',
    'SQ', 'PYPL', 'DIS', 'BA', 'F', 'GE', 'SNAP', 'UBER', 'LYFT',
]

# Positive templates (bullish sentiment)
POSITIVE_TEMPLATES = [
    "{ticker} to the moon! 🚀🚀🚀",
    "Just bought more {ticker} calls, this is going to squeeze",
    "{ticker} gains today were incredible, diamond hands paying off",
    "Bull case for {ticker}: revenue up, earnings beat expectations",
    "{ticker} breaking out, massive rally incoming",
    "My {ticker} calls are printing money, best investment ever",
    "{ticker} is the most undervalued stock in the market right now",
    "Holding {ticker} forever, this company is amazing",
    "{ticker} surge after earnings! Great growth ahead",
    "YOLO into {ticker}, tendies incoming!",
    "Just made $10k on {ticker} options, green all day",
    "{ticker} strong support level, buy the dip",
    "Why {ticker} will outperform the market this year",
    "Long {ticker} - excellent fundamentals and growth",
    "{ticker} is going to make us all rich",
    "Buying more {ticker} shares, the future is bright",
    "My portfolio is up 50% thanks to {ticker}",
    "{ticker} rocket ship, hold tight everyone!",
    "Best earnings report ever for {ticker}, upgrade coming",
    "{ticker} winning, love this stock so much",
]

# Negative templates (bearish sentiment)
NEGATIVE_TEMPLATES = [
    "{ticker} is crashing, sell everything now",
    "Lost all my money on {ticker} puts, worst day ever",
    "{ticker} is going bankrupt, total scam",
    "Bear case for {ticker}: declining revenue, debt crisis",
    "My {ticker} bags are so heavy, down 80%",
    "{ticker} dump after terrible earnings report",
    "Fear and panic selling in {ticker}, market crash incoming",
    "{ticker} puts printing, this stock is dead",
    "Sold all my {ticker}, this company is a fraud",
    "{ticker} margin call wiped out my account, I'm broke",
    "Why {ticker} is overvalued and headed to zero",
    "{ticker} bleeding red all week, worst investment",
    "Recession fears tanking {ticker}, short this garbage",
    "Lost everything on {ticker}, should have sold earlier",
    "{ticker} falling off a cliff, bearish pattern forming",
    "Terrible news for {ticker}, downgrade expected",
    "{ticker} bubble about to pop, weak fundamentals",
    "My {ticker} position is worthless now",
    "Inflation destroying {ticker} stock price, panic mode",
    "{ticker} is in a death spiral, get out while you can",
]

# Neutral templates
NEUTRAL_TEMPLATES = [
    "What do you think about {ticker}?",
    "Anyone else looking at {ticker} today?",
    "{ticker} daily discussion thread",
    "Need advice on {ticker} position",
    "{ticker} earnings report coming next week",
    "What's the play for {ticker} this month?",
    "{ticker} chart analysis - thoughts?",
    "How is everyone playing {ticker} options?",
    "New to trading {ticker}, where to start?",
    "{ticker} volume is interesting today",
    "Comparing {ticker} to other stocks in the sector",
    "Any news on {ticker} recent developments?",
    "{ticker} price prediction for next quarter",
    "Should I hold or sell my {ticker} shares?",
    "What's happening with {ticker} after hours?",
    "Does anyone have DD on {ticker}?",
    "{ticker} technical analysis for beginners",
    "Market update: {ticker} trading sideways",
    "Looking at options chain for {ticker}",
    "How does {ticker} compare to the index?",
]

# Sarcastic templates (appear positive/negative but mean opposite)
SARCASTIC_TEMPLATES = [
    "Yeah {ticker} to the moon, what could possibly go wrong lol",
    "Great job buying {ticker} at the top, genius move",
    "Obviously {ticker} is the greatest investment ever, surely nothing bad will happen",
    "{ticker} down 50% but at least we have diamond hands right?",
    "Congratulations to everyone who bought {ticker} at ATH, brilliant",
    "Wow {ticker} only lost 30% today, amazing performance lmao",
    "Totally didn't see {ticker} crash coming, absolutely shocking",
    "Sure {ticker} will recover, just like my marriage haha",
    "{ticker} is definitely not a bubble, clearly a safe investment",
    "Nothing to see here, {ticker} is perfectly fine losing billions",
    "Smart money is definitely buying {ticker} at this price right?",
    "Brilliant strategy losing money on {ticker} puts, genius level trading",
    "Yeah right, {ticker} will definitely beat earnings this time surely",
    "Another amazing day for {ticker} bag holders, love to see it lol",
    "Incredible how {ticker} keeps finding new lows, truly impressive",
]

# General financial discussion
GENERAL_TEMPLATES = [
    "Market is looking interesting today",
    "Anyone else worried about inflation numbers?",
    "Fed meeting tomorrow, what's your strategy?",
    "Portfolio update for the week",
    "New to investing, any tips for beginners?",
    "Rate my portfolio: tech heavy with some ETFs",
    "Why I'm moving to index funds this year",
    "Options expiry Friday, what are you playing?",
    "Market crash or just a correction?",
    "Best books on investing and trading?",
    "Is the stock market still worth it?",
    "Weekly earnings thread - major reports coming",
    "How do you handle losing trades emotionally?",
    "Day trading tips that actually work",
    "Long term investing vs short term trading debate",
]


def generate_dataset(n_rows=1000000, output_path='r_wallstreetbets_big.csv',
                     seed=42, chunksize=100000):
    """
    Generate a synthetic WallStreetBets-style dataset.
    
    Args:
        n_rows: Number of rows to generate  
        output_path: Output CSV file path
        seed: Random seed for reproducibility
        chunksize: Rows per generation chunk
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"[INFO] Generating synthetic dataset: {n_rows:,} rows")
    start_time = time.time()
    
    # Generate in chunks to manage memory
    first_chunk = True
    generated = 0
    
    while generated < n_rows:
        batch_size = min(chunksize, n_rows - generated)
        
        rows = []
        for i in range(batch_size):
            idx = generated + i
            
            # Select template category with realistic distribution
            r = random.random()
            if r < 0.30:
                template = random.choice(POSITIVE_TEMPLATES)
            elif r < 0.55:
                template = random.choice(NEGATIVE_TEMPLATES)
            elif r < 0.75:
                template = random.choice(NEUTRAL_TEMPLATES)
            elif r < 0.90:
                template = random.choice(SARCASTIC_TEMPLATES)
            else:
                template = random.choice(GENERAL_TEMPLATES)
            
            ticker = random.choice(TICKERS)
            title = template.format(ticker=ticker)
            
            # Add some natural variation
            if random.random() < 0.1:
                title = title.upper()
            if random.random() < 0.15:
                title += " " + random.choice(["🚀", "💎", "🙌", "📈", "📉", "😂", "🤡", "💀"])
            
            # Generate other columns
            row = {
                'id': ''.join(random.choices(string.ascii_lowercase + string.digits, k=6)),
                'title': title,
                'score': max(0, int(np.random.exponential(50))),
                'author': f"user_{random.randint(1, 100000)}",
                'author_flair_text': random.choice([None, 'Retard', 'Ape', 'Bull', 'Bear', 'Diamond Hands', '']),
                'removed_by': random.choice([None, None, None, None, 'moderator', 'automod']),
                'total_awards_received': max(0, int(np.random.exponential(1))),
                'awarders': '[]',
                'created_utc': 1612137600 + random.randint(0, 31536000),  # 2021-2022
                'full_link': f"https://reddit.com/r/wallstreetbets/comments/{''.join(random.choices(string.ascii_lowercase, k=6))}",
                'num_comments': max(0, int(np.random.exponential(20))),
                'over_18': random.random() < 0.05,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if first_chunk:
            df.to_csv(output_path, index=False, mode='w', encoding='utf-8')
            first_chunk = False
        else:
            df.to_csv(output_path, index=False, mode='a', header=False, encoding='utf-8')
        
        generated += batch_size
        elapsed = time.time() - start_time
        print(f"  Generated {generated:,}/{n_rows:,} rows ({generated/n_rows*100:.0f}%) - {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n[DONE] Dataset generated successfully!")
    print(f"  File: {output_path}")
    print(f"  Rows: {n_rows:,}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Time: {total_time:.1f}s")


if __name__ == '__main__':
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'r_wallstreetbets_big.csv')
    generate_dataset(n_rows=1000000, output_path=output)
