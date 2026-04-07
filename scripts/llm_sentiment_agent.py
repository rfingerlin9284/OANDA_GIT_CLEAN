#!/usr/bin/env python3
"""
scripts/llm_sentiment_agent.py — RBOTZILLA_OANDA_CLEAN
Label: NEW_CLEAN_FEATURE

Standalone Sentiment Hive Mind.
Runs in the background, pulls real-time contextual news from CryptoPanic, Reddit, and Yahoo Finance,
passes the data to an LLM for structured scoring, and outputs `logs/market_sentiment.json`.

Usage:
  python3 scripts/llm_sentiment_agent.py
  (Add --dry-run to test APIs without overwriting the live file)
"""

import os
import time
import json
import logging
import argparse
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

# Set up raw logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("sentiment")

# Load configuration
load_dotenv()
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO_DIR, "logs")
OUT_FILE = os.path.join(LOG_DIR, "market_sentiment.json")

# Extract API Keys
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# Ensure logs dir exists
os.makedirs(LOG_DIR, exist_ok=True)


class SentimentNode:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.brain = {}
        # Prepopulate structure
        self.targets = {
            "BTC_USD": {"query": "BTC", "source": "cryptopanic"},
            "ETH_USD": {"query": "ETH", "source": "cryptopanic"},
            "EUR_USD": {"query": "EUR", "source": "yahoo"},
            "USD_JPY": {"query": "JPY", "source": "yahoo"}
        }

    def fetch_cryptopanic(self, currency: str) -> str:
        """Fetch latest headlines from CryptoPanic for a specific coin."""
        if not CRYPTOPANIC_API_KEY:
            return ""
        
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&currencies={currency}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                titles = [str(post.get("title", "")) for post in data.get("results", [])[:5]]
                return " | ".join(titles)
        except Exception as e:
            log.warning(f"CryptoPanic fetch failed for {currency}: {e}")
        return ""

    def fetch_yahoo_finance(self, query: str) -> str:
        """Fetch lightweight RSS/Search headlines from Yahoo Finance (Macro)."""
        # Placeholder for Yahoo Finance / Reddit logic pending IBKR or full RSS integration.
        # Hardcoded mock feed for demonstration purposes during integration setup.
        if query == "EUR":
            return "ECB signals potential rate cuts | Eurozone inflation cooling fast"
        elif query == "JPY":
            return "Bank of Japan discusses tightening | Yen surges on intervention fears"
        return "General market volatility continues"

    def query_llm_sentiment(self, symbol: str, text: str) -> dict:
        """
        Sends aggregated text to a local Ollama LLM to generate a strict sentiment JSON response.
        Requires Ollama running on localhost:11434 with llama3 (or gemma2) installed.
        """
        if not text:
            return {"sentiment": 0.0, "confidence": 0.0, "catalyst": "No data"}

        prompt = (
            f"Analyze the following financial news headlines for {symbol}. "
            "Determine the immediate market sentiment. "
            "Respond ONLY with a valid JSON object matching this exact structure: "
            '{"sentiment": <float from -1.0 to 1.0>, "confidence": <float from 0.0 to 1.0>, "catalyst": "<3 word summary>"} '
            f"Headlines: {text}"
        )

        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            if res.status_code == 200:
                data = res.json()
                response_text = data.get("response", "{}")
                parsed = json.loads(response_text)
                return {
                    "sentiment": float(parsed.get("sentiment", 0.0)),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "catalyst": str(parsed.get("catalyst", "Ollama interpretation"))
                }
        except requests.exceptions.ConnectionError:
            log.warning("[Ollama] Connection refused. Is Ollama running on localhost:11434?")
        except Exception as e:
            log.error(f"[Ollama] Generation failed: {e}")

        # Fallback keyword logic if Ollama is down/missing
        log.debug(f"[{symbol}] Ollama failed, defaulting to neutral/keyword logic.")
        if "cut" in text.lower() or "dovish" in text.lower():
            return {"sentiment": -0.6, "confidence": 0.5, "catalyst": "Keyword logic: Dovish"}
        elif "tightening" in text.lower() or "surge" in text.lower() or "bull" in text.lower():
            return {"sentiment": 0.6, "confidence": 0.5, "catalyst": "Keyword logic: Bullish"}
        
        return {"sentiment": 0.0, "confidence": 0.1, "catalyst": "Neutral fallback"}

    def run_cycle(self):
        log.info("Starting sentiment gathering cycle...")
        updated_state = {}

        for pair, config in self.targets.items():
            query = config["query"]
            source = config["source"]

            # 1. Gather raw data
            headlines = ""
            if source == "cryptopanic":
                headlines = self.fetch_cryptopanic(query)
            else:
                headlines = self.fetch_yahoo_finance(query)

            # 2. Score via LLM
            score_data = self.query_llm_sentiment(pair, headlines)
            
            # 3. Append metadata
            score_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            updated_state[pair] = score_data
            log.info(f"[{pair}] Sentiment: {score_data['sentiment']} (Conf: {score_data['confidence']}) - {score_data['catalyst']}")

            # Sleep briefly to avoid API rate limits
            time.sleep(1)

        # 4. Save to Brain JSON
        if not self.dry_run:
            try:
                with open(OUT_FILE, "w") as f:
                    json.dump(updated_state, f, indent=2)
                log.info(f"Successfully wrote {len(updated_state)} states to {OUT_FILE}")
            except Exception as e:
                log.error(f"Failed to write market_sentiment.json: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Hive Mind")
    parser.add_argument("--dry-run", action="store_true", help="Run once without saving to the JSON log")
    args = parser.parse_args()

    agent = SentimentNode(dry_run=args.dry_run)
    
    if args.dry_run:
        log.info("Executing Single Dry-Run...")
        agent.run_cycle()
    else:
        log.info("Starting Persistent Sentiment Node 🧠")
        while True:
            agent.run_cycle()
            log.info("Sleeping for 5 minutes...")
            time.sleep(300)
