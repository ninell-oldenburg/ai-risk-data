import pandas as pd

# define manually!
CLUSTER_TOPICS = {
        0: 'ai: agi',
        1: 'ai: safety',
        2: 'ai: model capabilities',
        3: 'rationality: mental health & productivity', 
        4: 'economics: macroeconomics',
        5: 'economics: nuclear preparedness',
        6: 'philosophy: consciousness',
        7: 'ai: neural network circuits', 
        8: 'ai: alignment',
        9: 'rationality: abstract world problems',
        10: 'rationality: concrete world problems',
        11: 'ai: mech interp',
        12: 'community: personal stories',
        13: 'community: lesswrong auditing',
        14: 'ai: reinforcement learning',
        15: 'ai: model training & optimization',
        16: 'bio: covid',
        17: 'ai: tech giants',
        18: 'bio: evolution',
        19: 'economics: decision theory',
        20: 'philosophy: simulations',
        21: 'ai: model reasoning',
        22: 'community: media and media threads',
        23: 'ai: research',
        24: 'rationality: forecasting',
        25: 'misc: miscelleanous',
        26: 'rationality: social problems',
        27: 'ai: alignment problems', 
        28: 'philosophy: moral philosophy',
        29: 'community: effective altruism',
        30: 'ai: scaling',
        31: 'ai: probability theory', 
        32: 'economics: decision theory',
        33: 'economics: game theory',
        34: 'ai: data & information',
        35: 'rationality: parenting',
    }

class TopicsToCsv:
    def __init__(self):
        pass

    def append_topics_to_csv():
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
                merged["topic_label"] = merged["topic_label"].fillna("misc: no topic")
                
                mask = (merged["topic_label"] == "No Topic") & (
                    merged["cleaned_htmlBody"].str[:50].str.lower().str.contains("meetup", na=False)
                    | merged["title"].str.lower().str.contains("meetup", na=False)
                )
                # meetups are defined rule-based
                merged.loc[mask, "topic_label"] = "community: meetup"
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