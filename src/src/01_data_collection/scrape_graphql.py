import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

class ScrapeLesswrong:
  
    """ 
    Simple python class to scrape lesswrong data during a certain time.
    For changing the dates, go to get_and_save_articles(). 
    To change the result variables, visit https://www.lesswrong.com/graphiql 
    """
  
    def __init__(self, platform):
        try:
            if platform in ['lw', 'af']:
                self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")
        try:
            if platform == 'lw':
                self.url = "https://www.lesswrong.com/graphql"
                self.headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Referer": "https://www.lesswrong.com/",
                    "Origin": "https://www.lesswrong.com"
                }
            elif platform == 'af':
                self.url = "https://www.alignmentforum.org/graphql"
                self.headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://www.alignmentforum.org/",
                    "Origin": "https://www.alignmentforum.org",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                }
            else:
                raise ValueError("FORUM variable has to be 'lw' or 'af'")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        self.session = requests.Session()
        
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
        start_date = datetime(2009, 1, 1) if self.platform == 'lesswrong' else datetime(2015, 1, 1)
        end_date = datetime(2025, 6, 30)
        last_month = start_date
        all_results = []
        files_opened = 0

        print(f"\n{'='*60}")
        print(f"Processing year {start_date.year}...")
        print(f"{'='*60}")

        while start_date < end_date:
            
            # calculate month range
            next_month = (start_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            after = start_date.isoformat() + "Z"
            before = next_month.isoformat() + "Z"

            if last_month.year != start_date.year:
                print()
                print(f"\n{'='*60}")
                print(f"Processing year {next_month.year}...")
                print(f"{'='*60}")
            
            variables = {"after": after, "before": before, "limit": 1000}  # 10000 per request
            response = self.session.post(self.url, json={"query": self.query, "variables": variables}, headers=self.headers)
            
            # Check for request errors
            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code} for {start_date.strftime('%Y-%m')}")
                start_date = next_month
                continue
                
            data = response.json()
            results = data.get("data", {}).get("posts", {}).get("results", [])
            
            # Create directory structure if it doesn't exist
            year_dir = f"src/raw_data/{self.platform}/json/{start_date.year}"
            os.makedirs(year_dir, exist_ok=True)
            
            # Save JSON for this month
            filename = f"{start_date.year}-{start_date.month:02}.json"
            filepath = os.path.join(year_dir, filename)

            all_results.append(results)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                files_opened += 1

            print(f"✅ Saved {len(results)} posts to {filepath}")

            # small delay to avoid rate limiting
            time.sleep(1)

            last_month = start_date
            start_date = next_month

        return len(all_results), files_opened

def main(platform):
    scraper = ScrapeLesswrong(platform=platform)
    n_posts, n_files = scraper.get_and_save_articles()
    print(f"\n{'='*60}")
    print(f"✓ COMPLETE: Retrieved {n_posts} papers total")
    print(f"✓ Saved across {len(n_files)} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <forum>")
        print("Where <forum> is 'lw' or 'af'")
        sys.exit(1)
    main(sys.argv[1])