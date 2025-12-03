import os
import pandas as pd
from datasketch import MinHash, MinHashLSH

class OpenAlexDeduplicator:
    def __init__(self, threshold=0.85):
        self.input_folder = "src/raw_data/openalex/csv/"
        self.output_folder = "src/processed_data/openalex/01_dedup/"
        os.makedirs(self.output_folder, exist_ok=True)
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.minhashes = {}
        self.total_original = 0
        self.total_unique = 0

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

        self.total_original += len(df)
        
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

        self.total_unique += len(unique_rows)
        
        print(f"🧹 {filepath}: reduced {len(df)} → {len(unique_rows)} unique rows")
        return pd.DataFrame(unique_rows)

    def run(self):
        all_csvs = [
            os.path.join(root, f)
            for root, _, files in os.walk(self.input_folder)
            for f in files if f.endswith(".csv")
        ]

        if not all_csvs:
            print("⚠️ No CSV files found — check your input folder structure.")
            return

        for input_path in all_csvs:
            rel_path = os.path.relpath(input_path, self.input_folder)
            output_path = os.path.join(self.output_folder, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            df_unique = self.deduplicate_file(input_path)
            if df_unique is not None and not df_unique.empty:
                df_unique.to_csv(output_path, index=False, encoding="utf-8")
                print(f"✅ Saved deduplicated file: {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("📊 DEDUPLICATION SUMMARY")
        print("="*60)
        print(f"Total papers (original):        {self.total_original:,}")
        print(f"Total papers (after dedup):     {self.total_unique:,}")
        print(f"Duplicates removed:             {self.total_original - self.total_unique:,}")
        print(f"Retention rate:                 {(self.total_unique/self.total_original*100):.2f}%")
        print("="*60)


if __name__ == "__main__":
    deduper = OpenAlexDeduplicator()
    deduper.run()