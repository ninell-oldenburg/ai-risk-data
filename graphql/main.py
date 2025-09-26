import src.scrape_to_json as scrape
import src.json_to_csv as to_csv
import src.extract_links_and_gender as lng
import src.topic_clust as tc
import src.topics_to_csv as ttc
import src.arxiv_to_doi as atd
import sys

class Main:
    def __init__(self):
        self.scrape() = scrape.main()
        self.to_csv() = to_csv.main()
        self.lng() = lng.main()
        self.tc() = tc.main()
        self.ttc() = ttc.main()
        self.atd() = atd.main()

def main(forum):
    main = Main()
    main.scrape(forum)
    main.to_csv(forum)
    main.lng(forum)
    ntopics = main.tc(forum)
    main.ttc(forum, ntopics)
    main.atd()

if __name__ == '__main__':

    if len(sys.argv) != 1:
        print("==================================================")
        print("==== USAGE: 'python graphiql.py/main.py FORUM ====")
        print("==== [FORUM]: forum you want to download      ====")
        print("====          == lw  for lesswrong.com OR     ====")
        print("====          == af  for alignmentforum.org   ====")
        print("==================================================")
    else:
        main(sys.argv[0])
