# AI Risk Research Network Data

This is a data set for citation and demographic analysis of the discourse around the risks of AI.

## Overview

The data set contains three sources: 1) LessWrong (founded in 2009), an online forum of the rationalist community where they, amongst other things, talk about AI risk, 2) the Alignment Forum (founded 2015), a very similar but a lot smaller forum only dedicated to AI, and 3) academic papers that are available under the topic "Ethics and Social Impacts of AI" on the data base openalex.org.

## Data Set Description

All data lies in `data/`, divided in nodes (all forum posts, all forum authors, all academic papers) and edges (edges for forum posts, edges for academic papers).

### Time Period
- LessWrong: 2015-2024
- Alignment Forum: 2015-2024
- OpenAlex Works: 2015-2024

### Data Collection
- LessWrong/AF posts scraped via GraphQL API on October 15, 2025
- OpenAlex data retrieved via API on October 15, 2025
- Gender inference performed using `nomquamgender`, `chegender`, and manual username mapping

### Contents

**Node Tables** (N nodes):
- `data/nodes/nodes_posts.csv` - Forum posts (N=15,234)
- `data/nodes/nodes_authors.csv` - Forum users (N=3,421)
- `data/nodes/nodes_openalex_works.csv` - Academic papers (N=8,932)
- `data/nodes/nodes_openalex_authors.csv` - Academic researchers (N=12,456)

**Edge Tables** (N edges):
- `data/edges/edges_post_citations.csv` - Post→Post/Paper citations (N=45,678)
- `data/edges/edges_openalex_citations.csv` - Paper→Paper (N=34,567)

### Details

Find detailed descriptions in `docs/`, i.e. 

- `docs/methodology.md` - Details about duplicate detection, gender prediction, and link extraction
- `docs/data_dictionary.csv` - Complete column definitions
- `docs/data_collection.json` - Collection metadata
- `docs/processing_log.txt` - Processing history

## Usage

### Loading Data
```python
import pandas as pd

forum_posts = pd.read_csv('data/nodes/nodes_posts.csv')
forum_authors = pd.read_csv('data/nodes/nodes_authors.csv')
academic_papers = pd.read_csv('data/nodes/nodes_papers.csv')
forum_citations = pd.read_csv('data/edges/edges_post_citations.csv')
academic_citations = pd.read_csv('data/edges/edges_openalex_citations.csv')
```

### Reproducibility
To reproduce this dataset from scratch (runs apprx. 2 hours on intel chip):
```
bashpip install -r requirements.txt
python src/src/main.py
```

or 
```
bashpip install -r requirements.txt
# Scrape data and save them to src/raw_data/
python src/src/01_data_collection/scrape_graphql.py lw
python src/src/01_data_collection/scrape_graphql.py af
python src/src/01_data_collection/scrape_openalex.py

# Make CSV of JSON and run near dup detection
python src/src/02_preprocessing/graphql_01_make_csv_wo_dups.py lw
python src/src/02_preprocessing/graphql_01_make_csv_wo_dups.py af

# Extract links and infer gender
python src/src/02_preprocessing/graphql_02_extract_links_predict_gender.py lw
python src/src/02_preprocessing/graphql_02_extract_links_predict_gender.py af

# Run topic analysis with several options.
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

# Run dup detection for academix papers
python src/src/02_preprocessing/openalex_01_clean_dups.py
# Infer author genders for academix papers
python src/src/02_preprocessing/openalex_02_predict_gender.py

# Construct final nodes and edges tables
python src/src/03_graph_construction/build_graph_tables.py
```

See docs/methodology.md for detailed pipeline documentation. To visualize the ouutputs, use `notebooks/visualize.ipynb`.

**Known Limitations:** Gender classification contains inherent uncertainties (see `docs/methodology.md`). Near-duplicate detection may not catch all duplicates.

## Metadata

**Version:** 1.0 
**Last Updated:** October 2025
**Citation:** *forthcoming*
**Python Version:** 3.10.8

## License
GNU GENERAL PUBLIC LICENSE V. 3

## Contact
ninelloldenburg@gmail.com