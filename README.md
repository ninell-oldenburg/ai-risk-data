# AI Risk Research Network Data

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10.8](https://img.shields.io/badge/python-3.10.8-blue.svg)](https://www.python.org/downloads/)

## Abstract

We present a dataset integrating academic literature and online discourse on AI ethics and safety, spanning 2000--2025. The dataset includes 201,481 papers from the academic publication database OpenAlex, 44,792 posts from the rationalist forum LessWrong, of which 28.5% are about AI Ethics and Safety, and 4,224 posts from the AI Alignment Forum, which only contains posts about the target topic. For academic papers, we provide OpenAlex metadata, citation networks, and author information; for forum posts, we provide cleaned text, extracted citations, engagement metrics, and BERTopic-based topic assignments. All entries include inferred author gender (F1=0.944 for forum, 0.977 for OpenAlex authors) to enable demographic analysis, though with noted limitations. Data are structured as nodes and edges, enabling analysis of citation patterns, knowledge flow, and collaboration across venues. Validation against 65 canonical papers achieved 96.7% coverage. The dataset supports longitudinal, bibliometric, and demographic analysis of AI ethics and safety research spanning academic and community venues. All data, code, and documentation are openly available.

---

## Contents
- [Abstract](#abstract)
- [Overview](#overview)
- [Data Record](#data-record)
- [Usage](#usage)
- [Technical Validation](#technical-validation)
- [Metadata](#metadata)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

The data set contains three sources: 

1) [LessWrong](https://www.lesswrong.com). An online forum of the rationalist community where, amongst other things, AI risk is discussed. Founded in 2009.
2) [Alignment Forum](https://www.alignmentforum.org). A very similar but a lot smaller forum only dedicated to AI. Founded in 2015.
3) Academic papers that are available under the topic "[Ethics and Social Impacts of AI](https://api.openalex.org/T10883)" on the citation graph data base [OpenAlex](https://openalex.org/).

---

## Data Record

All data lies in `data/`, divided in nodes and edges.

### Time Period
- LessWrong: 2015-2025
- Alignment Forum: 2015-2025
- OpenAlex Works: 2015-2025

### Data Collection
- LessWrong/AF posts scraped via [GraphQL API](https://www.lesswrong.com/graphiql) on December 3rd, 2025
- OpenAlex data retrieved via [API](https://docs.openalex.org/) on December 3rd, 2025
- Gender inference performed using [`nomquamgender`](https://github.com/ianvanbuskirk/nomquamgender) (Van Buskirk et al., 2023), [`chgender`](https://pypi.org/project/chgender/) (Zhou, 2016), and manual username mapping

### Contents

**Node Tables** (N nodes):
- `data/nodes/nodes_forum_posts.csv` - Forum posts (N=49,016) (~442,5 MB)
- `data/nodes/nodes_forum_authors.csv` - Forum users (N=6,198) (~1,3 MB)
- `data/nodes/nodes_openalex_works.csv` - Academic papers (N=201,481) (~34,5 MB)
- `data/nodes/nodes_openalex_authors.csv` - Academic researchers (N=365,946) (~38,7 MB)

**Edge Tables** (N edges):
- `data/edges/edges_post_to_post.csv` - Post→Post citations (N=9,415) (~340 KB)
- `data/edges/edges_post_to_openalex.csv` - Post→Paper citations (N=406) (~31 KB)
- `data/edges/edges_openalex_authorship.csv` - Author→Paper (N=640,486) (~35,9 MB)
- `data/edges/edges_openalex_to_openalex.csv` - Paper→Paper (N=581,852) (~38,4 MB)

**Total dataset size:** ~591,6 MB

### Details

Find detailed descriptions in `docs/`, i.e. 

- `docs/data_dictionary.csv` - Complete column definitions
- `docs/data_collection.json` - Metadata for the data collection

---

## Usage

### Loading Data
```python
import pandas as pd

forum_posts = pd.read_csv('data/nodes/nodes_forum_posts.csv')
forum_authors = pd.read_csv('data/nodes/nodes_forum_authors.csv')
academic_papers = pd.read_csv('data/nodes/nodes_openalex_works.csv')
forum_citations = pd.read_csv('data/edges/edges_post_to_post.csv')
academic_citations = pd.read_csv('data/edges/edges_openalex_citations.csv')
```

## Reproducibility
To reproduce this dataset from scratch (runs apprx. 5-9 hours on intel chip):
```bash
pip install -r requirements.txt
python src/src/main.py
```

or 
```bash
pip install -r requirements.txt
# Scrape data and save them to src/raw_data/
python src/src/01_data_collection/graphql_00_scrape.py lw
python src/src/01_data_collection/graphql_00_scrape.py af
python src/src/01_data_collection/openalex_00_scrape.py

# Make CSV of JSON and run near dup detection
python src/src/02_preprocessing/graphql_01_make_csv_wo_dups.py lw
python src/src/02_preprocessing/graphql_01_make_csv_wo_dups.py af

# Extract links and infer gender
python src/src/02_preprocessing/graphql_02_extract_links_predict_gender.py lw
python src/src/02_preprocessing/graphql_02_extract_links_predict_gender.py af

# Run topic analysis.
# The hyperparameters are already defined in src/metadata/config_topic_modeling.json
# You can also run a sweep yourself by uncommenting the sweep instructions in the code.
python src/src/02_preprocessing/graphql_03_run_topic_clustering.py lw
python src/src/02_preprocessing/graphql_03_run_topic_clustering.py af

# Take a break here and look at the topics clusters in 
# src/metadata/clustering_results/<method_ntopics>_summary.txt 
# and name the topics accordingly in: src/metadata/topic_labels.json
# Then proceed:
python src/src/02_preprocessing/graphql_04_append_topics_to_csv.py lw
python src/src/02_preprocessing/graphql_04_append_topics_to_csv.py af

# Run dup detection for academic papers
python src/src/02_preprocessing/openalex_01_clean_dups.py
# Infer author genders for academic papers
python src/src/02_preprocessing/openalex_02_predict_gender.py

# Construct final nodes and edges tables
python src/src/03_graph_construction/build_graph_tables.py
```

---

## Technical Validation

Please see the paper for details on how we validated the data.

## Metadata

**Version:** 1.0 

**Last Updated:** December 2025

**Citation:** *forthcoming*

### Contact
Anonymized for review.

---

## Acknowledgements
Anonymized for review.

---

## References

- Blei, D., Ng, A., & Jordan, M. (2001). Latent dirichlet allocation. Advances in neural information processing systems, 14.
- Van Buskirk, I; Clauset, A., and Larremore, D. B. (2023). An Open-Source Cultural Consensus Approach to Name-Based Gender Classification. Proceedings of the International AAAI Conference on Web and Social Media, Volumne 17, pages 866--877. Github. https://github.com/ianvanbuskirk/nomquamgender
- Zhou, J. (2016). chgender (Version 0.0.2) [Python package]. PyPI. https://pypi.org/project/chgender/