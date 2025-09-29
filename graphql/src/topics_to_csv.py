import pandas as pd
import sys

# define manually!
CLUSTER_TOPICS = {
        0: 'Rationality: Probability & Bayesian Reasoning', 
        1: 'AI: Research Agendas & Infrastructure',
        2: 'Community: Meetups & Events',
        3: 'AI: Agency, Control & Power',
        4: 'Economics: Markets & Economic Behavior',
        5: 'Rationality: Productivity & Wellbeing',
        6: 'AI: Research Communications & Postings',
        7: 'AI: Alignment Theory & Strategy',
        8: 'Community: LessWrong Meta & Community',
        9: 'Rationality: Science & Culture',
        10: 'Rationality: Pandemic Epistemics & COVID',
        11: 'AI: AGI & Global Catastrophic Risk',
        12: 'AI: AI Progress & Scaling',
        13: 'Economics: Game Theory & Decision Theory',
        14: 'Rationality: Knowledge Systems & Concepts',
        15: 'AI: Emergent Capabilities & Misalignment',
        16: 'Philosophy: Mind, Consciousness & Evolution',
        17: 'AI: Companies, Governance & Public Impact',
        18: 'Philosophy: Morality & Ethics',
        19: 'AI: Mechanistic Interpretability',
        20: 'Community: Informal & Exploratory Discussion',
        21: 'AI: Optimization, Mesa-Optimization & RL',
        22: 'AI: Value Alignment',
        23: 'AI: Core ML & Neural Networks',
        24: 'Rationality: Mathematics & Formal Reasoning'
    }

class TopicsToCsv:
    def __init__(self, platform, ntopics):
        self.ntopics = ntopics
        self.platform = platform

    def append_topics_to_csv(self):
        """
        Append topic columns (cluster index + label) to LW CSV files 
        using clustering results.
        """
        clustering_results_csv = f'graphql/topics/{self.platform}/lda_{self.ntopics}.csv'

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
                correct_file_path = f'graphql/data/{self.platform}/csv_cleaned/{file_path.split(f"graphql/data/{self.platform}/csv_cleaned/")[1]}'  # gets "csv_cleaned/2025/2025-01.csv"
                
                # original file with correct path
                df = pd.read_csv(correct_file_path)
                original_count = len(df)

                merge_data = group[["_id", "dominant_topic", "topic_label"]]
                merged = df.merge(merge_data, how="left", on="_id")

                # rename cluster column
                merged.rename(columns={"dominant_topic": "topic_cluster_id"}, inplace=True)

                merged["topic_cluster_id"] = merged["topic_cluster_id"].fillna(-1).astype(int)
                merged["topic_label"] = merged["topic_label"].fillna("Misc: No Topic")

                # save back with proper filename
                output_path = f'graphql/data/{self.platform}/csv_cleanedwithtopic/{file_path.split("csv_cleaned/")[1]}'
                
                # check output directory exists
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
    if not len(sys.argv == 3):
        print("Usage: python script.py <forum>")
        print("Where <forum> is 'lw' or 'af'")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])