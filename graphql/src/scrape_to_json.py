import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import sys

class ScrapeLesswrong:
  
    """ 
    Simple pyhton class to scrape lesswrong data during a certain time.
    For changing the dates, go to get_and_save_articles(). 
    To change the result variables, visit https://www.lesswrong.com/graphiql 
    """
  
    def __init__(self, forum):
        self.platform = forum
        try:
            if forum == 'lw':
                self.url = "https://www.lesswrong.com/graphql"
                self.headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Referer": "https://www.lesswrong.com/",
                    "Origin": "https://www.lesswrong.com"
                }
            elif forum == 'af':
                self.url = "https://www.alignmentforum.org/graphql"
                self.headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Referer": "https://www.alignmentforum.org/",
                    "Origin": "https://www.alignmentforum.org"
                }
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")
        
        self.query = """
        query ($after: Date, $before: Date, $limit: Int) {
            posts(input: {
            terms: {
                view: "new",
                limit: $limit,
                meta: false,
                after: $after,
                before: $before
            }
            }) {
            results {
                _id
                title
                slug
                pageUrl
                postedAt
                baseScore
                voteCount
                commentCount
                meta
                question
                url
                htmlBody
                user {
                username
                slug
                displayName
                }
            }
            }
        }
        """

    def get_and_save_articles(self):
        start_date = datetime(2016, 1, 1)
        end_date = datetime(2025, 12, 31)

        all_results = []

        while start_date < end_date:
            
            # calculate month range
            next_month = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            after = start_date.isoformat() + "Z"
            before = next_month.isoformat() + "Z"
            
            variables = {"after": after, "before": before, "limit": 10000}  # 5000 per request
            response = requests.post(self.url, json={"query": self.query, "variables": variables}, headers=self.headers)
            data = response.json()
            results = data.get("data", {}).get("posts", {}).get("results", [])
            
            # Save JSON for this month
            filename = f"{start_date.year}-{start_date.month:02}.json"
            filepath = f"graphql/data/{self.platform}/json/{start_date.year}/" + filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved {len(results)} posts to {filepath}")

            # small delay to avoid rate limiting
            time.sleep(1)

            start_date = next_month

def main(platform):
    scraper = ScrapeLesswrong(forum=platform)
    scraper.get_and_save_articles()

if __name__ == "__main__":
    main(sys.argv[1])