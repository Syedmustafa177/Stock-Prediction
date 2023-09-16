from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Newsanalysis:
    def __init__(self):
        # Initialization code goes here
        pass

    def newsa(self, user_input):
        fin_viz_url = f"https://finviz.com/quote.ashx?t={user_input}"

        news_tables = {}

        url = fin_viz_url

        req = Request(url=url, headers={'user-agent': 'my-app'})

        try:
            response = urlopen(req)
        except HTTPError as e:
            print(f"HTTP Error: {e.code} - {e.reason}")
            return "🙇 Could'nt fetch NEWS data for this STOCK."

        html = BeautifulSoup(response, "html5lib")
        news_table = html.find(id="news-table")
        news_tables[user_input] = news_table

        parsed_data = []

        for row in news_table.findAll('tr'):
            title = row.a.text
            date_data = row.td.text.split()

            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]

            parsed_data.append([user_input, date, time, title])

        df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

        vader = SentimentIntensityAnalyzer()

        f = lambda title: vader.polarity_scores(title)["compound"]
        df["compound"] = df["title"].apply(f)
        df["date"] = df["date"].apply(lambda x: datetime.now().strftime("%Y-%m-%d") if x == "Today" else x)
        df["date"] = pd.to_datetime(df.date).dt.date

        # Filter news articles from the past 30 days
        current_date = datetime.now().date()
        past_30_days = current_date - timedelta(days=30)
        df = df[df['date'] >= past_30_days]

        sentiment_label = []

        for compound_score in df["compound"]:
            if compound_score > 0.05:
                sentiment_label.append("Positive")
            elif compound_score < -0.05:
                sentiment_label.append("Negative")
            else:
                sentiment_label.append("Neutral")

        df["sentiment"] = sentiment_label

        plt.figure(figsize=(10, 8))

        mean_df = df.groupby(['ticker', 'date']).mean()
        mean_df = mean_df.unstack()
        mean_df = mean_df.xs("compound", axis="columns").transpose()
        mean_df.plot(kind="bar")
        # plt.show()

        # print(mean_df)
        return df[["date","title", "sentiment"]]




# news = Newsanalysis()
# user_input = "WIT"
# result = news.newsa(user_input)
# if result is not None:
#     print(result)
# print(news)
