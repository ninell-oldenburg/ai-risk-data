import pandas as pd

# define manually!
CLUSTER_TOPICS = {
        0: 'AI: AGI & Superintelligence',
        1: 'AI: AI Safety & Risk Management',
        2: 'AI: Alignment & Control',
        3: 'Rationality: Productivity & Bias', 
        4: 'Economics: Markets & Economic Behavior',
        5: 'Misc: Geopolitics & International Relations',
        6: 'Rationality: Psychiatry & Mind',
        7: 'AI: Neural Networks & Interpretability', 
        8: 'AI: Research Communications & Postings',
        9: 'Rationality: Mental Models & Conceptual Frameworks',
        10: 'Rationality: Argumentation & Belief Formation',
        11: 'AI: Sparse Autoencoders & Feature Analysis',
        12: 'Community: Personal Narratives & Stories',
        13: 'Community: LessWrong Meta & Community',
        14: 'AI: Reinforcement Learning',
        15: 'AI: Optimization & Mesa-Optimization',
        16: 'Misc: COVID & Health Research',
        17: 'AI: AI Industry & Companies',
        18: 'Philosophy: Evolution & Long-term Future',
        19: 'Economics: Complex Systems & Social Decision-making',
        20: 'Philosophy: Physics & Cosmology',
        21: 'AI: Large Language Models & Applications',
        22: 'Community: Digital Media & Online Platforms',
        23: 'AI: ML Research & Academic Papers',
        24: 'Rationality: Forecasting & Predictions',
        25: 'Community: Conversational/Informal Discussion',
        26: 'Psychology: Social Psychology & Relationships',
        27: 'AI: Alignment Theory', 
        28: 'Philosophy: Moral Philosophy',
        29: 'Community: Effective Altruism Organization & Funding',
        30: 'AI: AI Progress & Scaling Laws',
        31: 'Rationality: Probability & Bayesian Reasoning', 
        32: 'Economics: Decision Theory & Utility Maximization',
        33: 'Economics: Game Theory & Mathematical Proofs',
        34: 'Misc: Software & Technology',
        35: 'Psychology: Education & Learning',
    }

class TopicsToCsv:
    def __init__(self):
        pass

    def append_topics_to_csv(self):
        """
        Append topic columns (cluster index + label) to LW CSV files 
        using clustering results.
        """
        clustering_results_csv = 'x-risk-data/topics/lda_results_36.csv'

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
                correct_file_path = f'x-risk-data/data/lw_csv_cleaned/{file_path.split("x-risk-data/data/lw_csv_cleaned/")[1]}'  # gets "lw_csv_cleaned/2025/2025-01.csv"
                
                # original file with correct path
                df = pd.read_csv(correct_file_path)
                original_count = len(df)

                merge_data = group[["title", "dominant_topic", "topic_label"]]
                merged = df.merge(merge_data, how="left", on="title")

                # rename cluster column
                merged.rename(columns={"dominant_topic": "topic_cluster_id"}, inplace=True)

                merged["topic_cluster_id"] = merged["topic_cluster_id"].fillna(-1).astype(int)
                merged["topic_label"] = merged["topic_label"].fillna("Misc: No Topic")
                
                mask = (merged["topic_label"] == "Misc: No Topic") & (
                    merged["cleaned_htmlBody"].str[:20].str.lower().str.contains("meetup", na=False)
                    | merged["title"].str.lower().str.contains("meetup", na=False)
                )
                # meetups are defined rule-based
                merged.loc[mask, "topic_label"] = "Community: Meetups & Events"
                merged.loc[mask, "topic_cluster_id"] = -2 

                # save back with proper filename
                output_path = f'x-risk-data/data/lw_csv_cleaned_topic/{file_path.split("lw_csv_cleaned/")[1]}'
                
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

def main():
    converter = TopicsToCsv()
    converter.append_topics_to_csv()

if __name__ == "__main__":
    main()