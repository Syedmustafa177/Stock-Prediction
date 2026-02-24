import re
import requests
import feedparser
import yfinance as yf
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class Newsanalysis:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'en-US,en;q=0.9',
        }

    def _sentiment_label(self, score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        return "Neutral"

    def _apply_sentiment(self, df):
        """Add compound score and sentiment label columns to a news DataFrame."""
        df["compound"] = df["title"].apply(
            lambda t: self.vader.polarity_scores(t)["compound"]
        )
        df["sentiment"] = df["compound"].apply(self._sentiment_label)
        return df

    # ------------------------------------------------------------------
    # News scraping methods (in priority order)
    # ------------------------------------------------------------------

    def _fetch_news_yfinance(self, ticker):
        """Primary: fetch news via yfinance (no scraping, very reliable)."""
        try:
            stock = yf.Ticker(ticker)
            items = stock.news
            if not items:
                return None
            parsed = []
            for item in items:
                title = item.get("title", "")
                pub_ts = item.get("providerPublishTime", 0)
                date = (
                    datetime.fromtimestamp(pub_ts).date()
                    if pub_ts
                    else datetime.now().date()
                )
                parsed.append([ticker, date, title])
            df = pd.DataFrame(parsed, columns=["ticker", "date", "title"])
            cutoff = datetime.now().date() - timedelta(days=30)
            return df[df["date"] >= cutoff]
        except Exception as e:
            print(f"yfinance news fetch failed: {e}")
            return None

    def _fetch_news_yahoo_rss(self, ticker):
        """Secondary: Yahoo Finance RSS feed."""
        try:
            url = (
                f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                f"?s={ticker}&region=US&lang=en-US"
            )
            feed = feedparser.parse(url)
            parsed = []
            for entry in feed.entries:
                title = entry.get("title", "")
                pub = entry.get("published_parsed")
                date = datetime(*pub[:3]).date() if pub else datetime.now().date()
                parsed.append([ticker, date, title])
            if not parsed:
                return None
            df = pd.DataFrame(parsed, columns=["ticker", "date", "title"])
            cutoff = datetime.now().date() - timedelta(days=30)
            return df[df["date"] >= cutoff]
        except Exception as e:
            print(f"Yahoo RSS fetch failed: {e}")
            return None

    def _fetch_news_google_rss(self, ticker, company_name=""):
        """Tertiary: Google News RSS feed."""
        try:
            query = f"{company_name} stock" if company_name else f"{ticker} stock"
            url = (
                f"https://news.google.com/rss/search"
                f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            feed = feedparser.parse(url)
            parsed = []
            for entry in feed.entries:
                title = entry.get("title", "")
                pub = entry.get("published_parsed")
                date = datetime(*pub[:3]).date() if pub else datetime.now().date()
                parsed.append([ticker, date, title])
            if not parsed:
                return None
            df = pd.DataFrame(parsed, columns=["ticker", "date", "title"])
            cutoff = datetime.now().date() - timedelta(days=30)
            return df[df["date"] >= cutoff]
        except Exception as e:
            print(f"Google News RSS fetch failed: {e}")
            return None

    def _fetch_news_finviz(self, ticker):
        """Fallback: legacy FinViz scraping (may be blocked by bot detection)."""
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            req = Request(url=url, headers={"user-agent": "my-app"})
            response = urlopen(req)
            html = BeautifulSoup(response, "html5lib")
            news_table = html.find(id="news-table")
            if not news_table:
                return None
            parsed = []
            date = datetime.now().strftime("%Y-%m-%d")
            for row in news_table.findAll("tr"):
                if row.a is None:
                    continue
                title = row.a.text
                date_data = row.td.text.split()
                if len(date_data) == 1:
                    # Single element means only a time is present;
                    # this row shares the date from the previous row.
                    pass
                else:
                    date = date_data[0]
                parsed.append([ticker, date, title])
            if not parsed:
                return None
            df = pd.DataFrame(parsed, columns=["ticker", "date", "title"])
            df["date"] = df["date"].apply(
                lambda x: datetime.now().strftime("%Y-%m-%d") if x == "Today" else x
            )
            df["date"] = pd.to_datetime(df["date"]).dt.date
            cutoff = datetime.now().date() - timedelta(days=30)
            return df[df["date"] >= cutoff]
        except Exception as e:
            print(f"FinViz fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def newsa(self, ticker):
        """
        Fetch market news and run VADER sentiment analysis.
        Tries multiple sources in order: yfinance → Yahoo RSS →
        Google News RSS → FinViz (legacy).
        Returns a DataFrame with columns [date, title, sentiment]
        or a human-readable error string.
        """
        company_name = ""
        try:
            company_name = yf.Ticker(ticker).info.get("longName", "")
        except Exception:
            pass

        df = self._fetch_news_yfinance(ticker)
        if df is None or df.empty:
            df = self._fetch_news_yahoo_rss(ticker)
        if df is None or df.empty:
            df = self._fetch_news_google_rss(ticker, company_name)
        if df is None or df.empty:
            df = self._fetch_news_finviz(ticker)

        if df is None or df.empty:
            return "🙇 Could not fetch news data for this stock."

        df = self._apply_sentiment(df)
        return df[["date", "title", "sentiment"]]

    def get_employer_sentiment(self, ticker):
        """
        Retrieve employee / employer sentiment data.

        Uses two sources:
        1. Glassdoor public page scraping (best-effort).
        2. yfinance company info metrics as a reliable fallback.

        Returns a dict with keys:
            company, employees, sector, industry,
            glassdoor_rating (optional), sentiment, source
        """
        result = {}
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            result["company"] = info.get("longName", ticker)
            result["employees"] = info.get("fullTimeEmployees", "N/A")
            result["sector"] = info.get("sector", "N/A")
            result["industry"] = info.get("industry", "N/A")
        except Exception as e:
            print(f"yfinance info fetch failed: {e}")
            result.setdefault("company", ticker)

        # Attempt Glassdoor scrape ----------------------------------------
        try:
            company_slug = result.get("company", ticker).split()[0].lower()
            gd_url = (
                f"https://www.glassdoor.com/Reviews/"
                f"{company_slug}-reviews-SRCH_KE0,{len(company_slug)}.htm"
            )
            resp = requests.get(gd_url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(resp.content, "html.parser")
            rating_el = (
                soup.find("div", {"data-test": "rating-info"})
                or soup.find("span", class_="ratingNumber")
                or soup.find("div", class_="ratingNumber")
            )
            if rating_el:
                raw = rating_el.get_text(strip=True)
                # Glassdoor ratings are always in "X.X" format (0–5 scale).
                # Only attempt conversion when the text looks like a decimal number.
                if re.match(r'^\d(\.\d)?$', raw):
                    rating = float(raw)
                else:
                    raise ValueError(f"Unexpected rating format: {raw!r}")
                result["glassdoor_rating"] = rating
                result["sentiment"] = (
                    "Positive" if rating >= 3.5
                    else ("Neutral" if rating >= 2.5 else "Negative")
                )
                result["source"] = "Glassdoor"
        except Exception as e:
            print(f"Glassdoor scrape failed: {e}")

        if "sentiment" not in result:
            result["source"] = "yfinance"
            result["sentiment"] = "N/A (review site unavailable)"

        return result

    def get_future_plans(self, ticker):
        """
        Gather forward-looking company data from yfinance.

        Returns a dict with:
            company, business_summary, sector, industry,
            forward_pe, revenue_growth, earnings_growth,
            target_mean_price, recommendation,
            earnings_dates (DataFrame or None),
            analyst_recommendations (DataFrame or None)
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            data = {
                "company": info.get("longName", ticker),
                "business_summary": info.get("longBusinessSummary", ""),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "revenue_growth": info.get("revenueGrowth", "N/A"),
                "earnings_growth": info.get("earningsGrowth", "N/A"),
                "target_mean_price": info.get("targetMeanPrice", "N/A"),
                "recommendation": info.get("recommendationKey", "N/A"),
                "earnings_dates": None,
                "analyst_recommendations": None,
            }
            try:
                cal = stock.calendar
                if cal is not None and not cal.empty:
                    data["earnings_dates"] = cal
            except Exception:
                pass
            try:
                recs = stock.recommendations
                if recs is not None and not recs.empty:
                    data["analyst_recommendations"] = recs.tail(5)
            except Exception:
                pass
            return data
        except Exception as e:
            print(f"Future plans fetch failed: {e}")
            return None
