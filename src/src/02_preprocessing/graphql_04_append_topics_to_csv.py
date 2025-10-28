import pandas as pd
import sys

# define manually!
CLUSTER_TOPICS = {
        0: 'Rationality: Probability & Bayesian Reasoning', 
        1: 'AI: Research Agendas & Infrastructure',
        2: 'Community: Meetups & Events',
    }

class TopicsToCsv:
    def __init__(self, platform, ntopics):
        self.ntopics = ntopics
        try:
            if platform in ['lw', 'af']:
                self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")
        self.input_base = f'src/processed_data/data/{self.platform}/02_with_links_and_gender/'
        self.output_base = f'src/processed_data/{self.platform}/03_with_topics/'


    def append_topics_to_csv(self):
        """
        Append topic columns (cluster index + label) to LW CSV files 
        using clustering results.
        """
        clustering_results_csv = f'src/metadata/clustering_results/{self.platform}/lda_{self.ntopics}.csv'

        print(f"Loading clustering results from {clustering_results_csv}...")
        results_df = pd.read_csv(clustering_results_csv)
        print(f"Loaded {len(results_df)} clustered posts")

        updated_files = 0
        total_matches = 0

        # human-readable topic labels
        results_df["topic_label"] = results_df["dominant_topic"].map(CLUSTER_TOPICS)

        for file_path, group in results_df.groupby("file"):
            print(f"\nProcessing {file_path}...")

            try:
                # just the filename and build correct path
                import os
                filename = os.path.basename(file_path)  # gets just "2025-01.csv"
                correct_file_path = self.input_base + f'{file_path.split(f"src/graphql/data/{self.platform}/csv_cleaned/")[1]}'

                # original file with correct path
                df = pd.read_csv(correct_file_path)
                original_count = len(df)

                merge_data = group[["_id", "dominant_topic", "topic_label"]]
                merged = df.merge(merge_data, how="left", on="_id")

                # rename cluster column
                merged.rename(columns={"dominant_topic": "topic_cluster_id"}, inplace=True)

                merged["topic_cluster_id"] = merged["topic_cluster_id"].fillna(-1).astype(int)
                merged["topic_label"] = merged["topic_label"].fillna("Misc: No Topic")

                output_path = self.output_base + f'{file_path.split("csv_cleaned/")[1]}'
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                merged.to_csv(output_path, index=False)

                matches = (merged["topic_cluster_id"] != -1).sum()
                updated_files += 1
                total_matches += matches

                print(f"  SUCCESS: {matches}/{original_count} posts matched")

            except Exception as e:
                print(f'Error processing {file_path}: {e}')
                print(f'Error type: {type(e).__name__}')

        print()
        print(f'TOTAL MATCHES: {total_matches}')

def main(platform, topics):
    converter = TopicsToCsv(platform=platform, ntopics=topics)
    converter.append_topics_to_csv()

if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage: python graphql_04_append_topics_to_csv.py <forum>")
        print("Where <forum> is 'lw' or 'af'")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])