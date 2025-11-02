import os
import json
import pandas as pd
import sys
from datasketch import MinHash, MinHashLSH

class LesswrongJsonToCsv:
    def __init__(self, platform, deduplicate=True, threshold=0.85):
        try:
            if platform in ['lw', 'af']:
                self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")
        self.input_base = f"src/raw_data/{self.platform}/json/" 
        self.output_base = f"src/processed_data/{self.platform}/01_cleaned_csv"
        self.total_posts = 0

        # near-duplicate detection
        self.deduplicate = deduplicate
        self.threshold = threshold
        if deduplicate:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
            self.minhashes = {}

    def _get_minhash(self, text):
        """Compute a MinHash signature for a given text."""
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        return m
    
    def _clean_title(self, title):
        """Clean title by removing line breaks and surrounding quotes."""
        if pd.isna(title):
            return ""
        
        title = str(title)
        # Remove line breaks
        title = title.replace('\n', ' ').replace('\r', ' ')
        # Remove multiple spaces
        title = ' '.join(title.split())
        # Remove surrounding quotes
        title = title.strip('"').strip("'")
        
        return title.strip()

    def _clean_body(self, body):
        """Clean body text."""
        if pd.isna(body):
            return ""
        return str(body)

    def transform(self):
        years = sorted([
            int(name) for name in os.listdir(self.input_base)
            if os.path.isdir(os.path.join(self.input_base, name)) and name.isdigit()
        ])
        for year in years:
            year_folder = os.path.join(self.input_base, str(year))
            if not os.path.exists(year_folder):
                continue

            output_year_folder = os.path.join(self.output_base, str(year))
            os.makedirs(output_year_folder, exist_ok=True)

            subtotal_posts = 0
            for filename in sorted(os.listdir(year_folder)):
                if not filename.endswith(".json"):
                    continue

                filepath = os.path.join(year_folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        posts = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse {filepath}: {e}")
                        continue

                if not isinstance(posts, list):
                    print(f"⚠️ Unexpected structure in {filepath}, skipping...")
                    continue

                df = pd.json_normalize(posts)
                columns_to_remove = ['user', 'url']
                df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

                df['title'] = df['title'].apply(self._clean_title)

                # Near duplicate removal
                if self.deduplicate and 'title' in df.columns and 'htmlBody' in df.columns:
                    unique_rows = []
                    for idx, row in df.iterrows():
                        text = f"{row['title']} {row['htmlBody']}"
                        mh = self._get_minhash(text)
                        dup = self.lsh.query(mh)
                        if not dup:
                            key = f"{year}-{filename}-{idx}"
                            self.lsh.insert(key, mh)
                            self.minhashes[key] = mh
                            unique_rows.append(row)
                    df = pd.DataFrame(unique_rows)
                    print(f"Deduplicated to {len(df)} unique posts in {filename}")

                if len(df) == 0:
                    continue

                csv_filename = filename.replace(".json", ".csv")
                output_path = os.path.join(output_year_folder, csv_filename)
                df.to_csv(output_path, index=False, encoding="utf-8")
                subtotal_posts += len(df)
                print(f"✅ Saved {output_path} ({len(df)} posts)")

            self.total_posts += subtotal_posts

        print(f'Total Posts: {self.total_posts}')

def main(platform):
    transformer = LesswrongJsonToCsv(platform, deduplicate=True)
    transformer.transform()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python graphql_01_make_csv_wo_dups.py <forum>")
        print("Where <forum> is 'lw' (LessWrong) or 'af' (Alignment Forum)")
        sys.exit(1)
        
    main(sys.argv[1])