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