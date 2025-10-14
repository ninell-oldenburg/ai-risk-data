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
        24: 'Rationality: Mathematics & Formal Reasoning',
        25: 'Rationality: Probability & Bayesian Reasoning', 
        26: 'AI: Research Agendas & Infrastructure',
        27: 'Community: Meetups & Events',
        28: 'AI: Agency, Control & Power',
        29: 'Economics: Markets & Economic Behavior',
        30: 'Rationality: Productivity & Wellbeing',
        31: 'AI: Research Communications & Postings',
        32: 'AI: Alignment Theory & Strategy',
        33: 'Community: LessWrong Meta & Community',
        34: 'Rationality: Science & Culture',
        35: 'Rationality: Pandemic Epistemics & COVID',
        36: 'AI: AGI & Global Catastrophic Risk',
        37: 'AI: AI Progress & Scaling',
        38: 'Economics: Game Theory & Decision Theory',
        39: 'Rationality: Knowledge Systems & Concepts',
        40: 'AI: Emergent Capabilities & Misalignment',
        41: 'Philosophy: Mind, Consciousness & Evolution',
        42: 'AI: Companies, Governance & Public Impact',
        43: 'Philosophy: Morality & Ethics',
        44: 'AI: Mechanistic Interpretability',
        45: 'Community: Informal & Exploratory Discussion',
        46: 'AI: Optimization, Mesa-Optimization & RL',
        47: 'AI: Value Alignment',
        48: 'AI: Core ML & Neural Networks',
        49: 'Rationality: Mathematics & Formal Reasoning',
        50: 'Community: Informal & Exploratory Discussion',
        51: 'AI: Optimization, Mesa-Optimization & RL',
        52: 'AI: Value Alignment',
        53: 'AI: Core ML & Neural Networks',
        54: 'Rationality: Mathematics & Formal Reasoning',
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