import src.scrape_to_json as scrape
import src.json_to_csv as to_csv
import src.extract_links_and_gender as lng
import src.topic_clust as tc
import src.topics_to_csv as ttc
import sys

class Main:
    def __init__(self):
        self.scrape = scrape.main
        self.to_csv = to_csv.main
        self.lng = lng.main
        self.tc = tc.main
        self.ttc = ttc.main

def main(forum):
    processor = Main()
    processor.scrape(forum)
    processor.to_csv(forum)
    processor.lng(forum)
    ntopics = processor.tc(forum, test=False, optimal_topics=25, type_cluster = 'lda')
    processor.ttc(forum, ntopics)

if __name__ == '__main__':
    # Check for exactly 2 arguments (script name + forum)
    if len(sys.argv) != 2:
        print("==================================================")
        print("==== USAGE: 'python main.py FORUM'            ====")
        print("==== [FORUM]: forum you want to download      ====")
        print("====          == lw  for lesswrong.com OR     ====")
        print("====          == af  for alignmentforum.org   ====")
        print("==================================================")
        sys.exit(1)
    else:
        # Use the actual forum argument, not the script name
        main(sys.argv[1])

bashpip install -r requirements.txt
python src/src/01_data_collection/scrape_graphql.py lw
python src/src/01_data_collection/scrape_graphql.py af
python src/src/01_data_collection/scrape_openalex.py

python src/src/02_preprocessing/graphql_01_make_csv_wo_dups.py lw
python src/src/02_preprocessing/graphql_01_make_csv_wo_dups.py af
python src/src/02_preprocessing/graphql_02_extract_links_predict_gender.py lw
python src/src/02_preprocessing/graphql_02_extract_links_predict_gender.py af

# The topic analysis below has several options.
# Additional to the forum (obligatory, lw or af), you can define
# <TEST (optional)> (bool, whether to run a test to find the optimal # of topics)
# <OPTIMAL_TOPICS (optional)>, the number of optimal topics if you want to test a specific number
# (in that case it is recommended to disable the test bool), and
# <TYPE_CLUSTER (optional)>, which clustering/analysis to run: K-Means or LDA (default)
python src/src/02_preprocessing/graphql_03_run_topic_clustering.py lw
python src/src/02_preprocessing/graphql_03_run_topic_clustering.py af

# Take a break here and look at the topics clusters in 
# src/metadata/clustering_results/<method_ntopics>_summary.txt 
# and name the topics accordingly in graphql_04_append_topics_to_csv.py
# for both lesswrong and alignment forum. Then proceed
python src/src/02_preprocessing/graphql_04_append_topics_to_csv.py lw
python src/src/02_preprocessing/graphql_04_append_topics_to_csv.py af

python src/src/02_preprocessing/openalex_01_clean_dups.py
python src/src/02_preprocessing/openalex_02_predict_gender.py

python src/src/03_graph_construction/build_graph_tables.py