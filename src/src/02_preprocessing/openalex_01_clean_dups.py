import os
import pandas as pd
from datasketch import MinHash, MinHashLSH

class OpenAlexDeduplicator:
    def __init__(self, threshold=0.85):
        self.input_folder = "src/raw_data/openalex/csv"
        self.output_folder = "src/processed_data/openalex/01_dedup"
        os.makedirs(self.output_folder, exist_ok=True)
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.minhashes = {}

    def _get_minhash(self, text):
        m = MinHash(num_perm=128)
        for token in str(text).split():
            m.update(token.encode("utf8"))
        return m

    def deduplicate_file(self, filepath):
        df = pd.read_csv(filepath)
        if "title" not in df.columns:
            print(f"⚠️  Skipping {filepath}: missing 'title' column.")
            return None

        unique_rows = []
        for i, row in df.iterrows():
            text = f"{row['title']} {row.get('abstract', '')}"
            mh = self._get_minhash(text)
            dup = self.lsh.query(mh)
            if not dup:
                key = f"{os.path.basename(filepath)}-{i}"
                self.lsh.insert(key, mh)
                self.minhashes[key] = mh
                unique_rows.append(row)

        print(f"🧹 {filepath}: reduced {len(df)} → {len(unique_rows)} unique rows")
        return pd.DataFrame(unique_rows)

    def run(self):
        all_csvs = [f for f in os.listdir(self.input_folder) if f.endswith(".csv")]
        for csv_file in all_csvs:
            input_path = os.path.join(self.input_folder, csv_file)
            df_unique = self.deduplicate_file(input_path)
            if df_unique is not None and not df_unique.empty:
                output_path = os.path.join(self.output_folder, csv_file)
                df_unique.to_csv(output_path, index=False, encoding="utf-8")
                print(f"✅ Saved deduplicated file: {output_path}")

if __name__ == "__main__":
    deduper = OpenAlexDeduplicator()
    deduper.run()
