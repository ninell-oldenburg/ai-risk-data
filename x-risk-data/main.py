import src.scrape_lw_to_json as scrape
import src.lw_json_to_csv as to_csv
import src.extract_links_and_gender as lng
import src.topic_clust as tc
import src.topics_to_csv as ttc

scrape.main()
to_csv.main()
lng.main()
tc.main()
ttc.main()