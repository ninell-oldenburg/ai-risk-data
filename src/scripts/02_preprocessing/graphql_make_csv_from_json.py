import os
import json
import pandas as pd
import sys

class LesswrongJsonToCsv:
    def __init__(self, platform):
        self.input_base = f"graphql/data/{platform}/json" 
        self.output_base = f"graphql/data/{platform}/csv"
        self.total_posts = 0

    def transform(self):
        # walk through all year folders
        for year in range(2015, 2025): 
            year_folder = os.path.join(self.input_base, str(year))
            if not os.path.exists(year_folder):
                continue

            # check output year folder exists
            output_year_folder = os.path.join(self.output_base, str(year))
            os.makedirs(output_year_folder, exist_ok=True)

            subtotal_posts = 0

            # all monthly JSON files (e.g. 2025-01.json)
            for filename in sorted(os.listdir(year_folder)):
                if not filename.endswith(".json"):
                    continue

                filepath = os.path.join(year_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        posts = json.load(f)  # this should be a list
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse {filepath}: {e}")
                        continue

                if not isinstance(posts, list):
                    print(f"⚠️ Unexpected structure in {filepath}, skipping...")
                    continue

                if not posts:
                    print(f"⚠️ No posts in {filepath}")
                    continue

                # flatten json into data frame
                df = pd.json_normalize(posts)

                # Remove 'user' and 'url' columns if they exist
                columns_to_remove = ['user', 'url']
                df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

                # output csv path
                csv_filename = filename.replace(".json", ".csv")
                output_path = os.path.join(output_year_folder, csv_filename)

                # Save
                df.to_csv(output_path, index=False, encoding="utf-8")
                subtotal_posts += len(df)
                print(f"✅ Saved {output_path} ({len(df)} posts)")

            self.total_posts += subtotal_posts

        print(f'Total Posts: {self.total_posts}')

def main(platform):
    transformer = LesswrongJsonToCsv(platform)
    transformer.transform()

if __name__ == "__main__":
    main(sys.argv[1])