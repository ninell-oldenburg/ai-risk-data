import pandas as pd
import sys
import json
import os

class TopicsToCsv:
    def _load_topic_labels(self, forum):
        """Load topic labels from JSON file."""
        json_path = 'src/metadata/topic_labels.json'
        try:
            with open(json_path, 'r') as f:
                all_labels = json.load(f)
            
            # Get labels for current platform
            if forum not in all_labels:
                raise ValueError(f"Platform '{forum}' not found in topic_labels.json")
            
            platform_labels = all_labels[forum]
            
            # Convert string keys to integers
            return {int(k): v for k, v in platform_labels.items()}
        
        except FileNotFoundError:
            print(f"ERROR: Could not find {json_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"ERROR: Invalid JSON in {json_path}")
            sys.exit(1)

    def append_topics_to_csv(self):
        """
        Append topic columns (cluster index + label) to LW CSV files 
        using clustering results.
        """
        updated_files = 0
        total_matches = 0

        matches_lw = 0
        matches_af = 0

        forums = ['lesswrong','alignment_forum']
        for forum in forums:
            clustering_result = f'src/metadata/clustering_results/{forum}/bertopic_results.csv'

            print(f"Loading clustering results from {clustering_result}...")
            results_df = pd.read_csv(clustering_result)
            print(f"Loaded {len(results_df)} clustered {forum} posts")

            # human-readable topic labels using loaded JSON
            topic_labels = self._load_topic_labels(forum)
            results_df["topic_label"] = results_df["topic"].map(topic_labels)

            for file_path, group in results_df.groupby("file"):
                print(f"\nProcessing {file_path}...")

                try:
                    # just the filename and build correct path
                    filename = os.path.basename(file_path)  # gets just "2025-01.csv"
                    correct_file_path = f'{file_path.split()[0]}'

                    # original file with correct path
                    df = pd.read_csv(correct_file_path)
                    original_count = len(df)

                    merge_data = group[["_id", "topic", "topic_label"]]
                    merged = df.merge(merge_data, how="left", on="_id")

                    # rename cluster column
                    merged.rename(columns={"topic": "topic_cluster_id"}, inplace=True)
                    merged["topic_cluster_id"] = merged["topic_cluster_id"].fillna(-1).astype(int)

                    output_path = file_path.replace('02_with_links_and_gender', '03_with_topics')
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    merged.to_csv(output_path, index=False)

                    matches = (merged["topic_cluster_id"] != -1).sum()
                    updated_files += 1
                    total_matches += matches

                    if forum == 'lesswrong':
                        matches_lw += 1  
                    else: 
                        matches_af += 1

                    print(f"  SUCCESS: {matches}/{original_count} posts matched")

                except Exception as e:
                    print(f'Error processing {file_path}: {e}')
                    print(f'Error type: {type(e).__name__}')

        print()
        print(f'TOTAL MATCHES: {total_matches}')
        print(f'LESSWRONG MATCHES: {matches_lw}')
        print(f'ALIGNMENT FORUM MATCHES: {matches_af}')

def main():
    converter = TopicsToCsv()
    converter.append_topics_to_csv()

if __name__ == "__main__":
    main()