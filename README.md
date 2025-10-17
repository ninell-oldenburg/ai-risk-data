# AI Risk Research Network Data

## Overview

This is a data set for citation demographic analysis of the discourse around the risks of AI. The data set spans the years January 2015 to June 2025. It contains three sources: 1) LessWrong (founded in 2009), an online forum of the rationalist community where they, amongst other things, talk about AI risk, 2) the Alignment Forum, a very similar but a lot smaller forum only dedicated to AI, and 3) academic papers that are available under the topic "Etics and Social Impacts of AI" on the data base openalex.org.

## Data and Metadata

All data lies in `data/`, divided in nodes (all forum posts, all forum authors, all academic papers) and edges (edges for forum posts, edges for academic papers).

## Dataset Description

### Time Period
- LessWrong: January 2015 - September 2024
- Alignment Forum: June 2017 - September 2024  
- OpenAlex Works: 2015-2024 (cited works only)

### Data Collection
- LessWrong/AF posts scraped via GraphQL API on October 15, 2024
- OpenAlex data retrieved via API on October 12, 2024
- Gender inference performed using [method] with X% coverage

### Contents

**Node Tables** (N nodes):
- `nodes/nodes_posts.csv` - Forum posts (N=15,234)
- `nodes/nodes_authors.csv` - Forum users (N=3,421)
- `nodes/nodes_openalex_works.csv` - Academic papers (N=8,932)
- `nodes/nodes_openalex_authors.csv` - Academic researchers (N=12,456)

**Edge Tables** (N edges):
- `edges/edges_post_citations.csv` - Post→Post citations (N=45,678)
- `edges/edges_post_openalex.csv` - Post→Paper citations (N=23,456)
- `edges/edges_openalex_authorship.csv` - Author→Paper (N=34,567)

**Metadata**:
- `metadata/data_dictionary.csv` - Column definitions
- `metadata/data_collection.json` - Collection metadata
- `metadata/processing_log.txt` - Processing history

## Data Dictionary
See `metadata/data_dictionary.csv` for complete column definitions.

### Key Fields
- **post_id**: Unique identifier format abc123XYZ
- **source**: 'lw' (LessWrong) or 'af' (Alignment Forum)
- **author_gender_inferred**: Inferred using [method], values: male/female/unknown
- **topic_id**: From BERTopic clustering (20 topics)

## Quality & Limitations

### Excluded Data
- Comments (not included)
- Draft posts
- Deleted or private posts

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

## Reproducibility
To reproduce this dataset from scratch:
```
bashpip install -r requirements.txt
python src/src/main.py
```

or 
```
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
```

See docs/methodology.md for detailed pipeline documentation.

All source code to generate the data lies in `src/src/`. Use `main.py` to run the whole script, or find the single files numbered in running order. In detail, these first scrape the data from either LessWrong, the Alignment Forum (smae script), or OpenAlex and saves the results in `src/raw_data`. It then preprocesses the data according to its origin and saves the intermediate data outputs in `src/processed_data`. More specifically, this means that it runs 1) a near duplicate detection, 2) extracts links for forum posts, 3) infers gender from username or firstname, and 4) runs a topic clustering analysis for forum posts. All steps are described below in detail. Lastly, the final nodes and edges tables are built in `src/src/03_graph_table_construction/`. To visualize the ouutputs, use `notebooks/visualize.ipynb`.

## Gender Classification

We classified the gender in different ways. Primarily, we started using the first name by generating a list of common first names and classifying the usernames on whether they contain a female or male first name. This works mostly well as many LessWrong users have a firstname-lastname (or firstnamelastname) style username. For this, we checked with the list of common names from long to short names to prevent one name being classified as different gender when it contains another gender's name, e.g. Pauline (female) as Paul (male) when paul would be checked first.

We then checked all users that were not catched by this inital classification. Especially, we checked all authors who wrote five or more posts for identifiers of their gender. This took different ways: some users refer ot their personal website, blog, or twitter account where we could see their name; others referred to themselves in their description as "male"/"guy" or "female"/"girl" or said that they don't identify with genders, etc. Lastly, we checked in the comments that this user was mentioned for pronouns and took this pronoun if the user themselves didn't complain about it in a later comment. The reasoning here is that either the commenting user and the user under question have met in real life and so know of each other's gender or the assumed it and the user under question accepted this assumption.

Users who deleted their account are also classified as "unknown". Users with ambiguously gendered names such as Logan or Kim unless they have more than five posts and self-identified gender noted anywhere.

We are well aware that this classification is still noisy from different aspects. People may change their gender and still have their old username. Different cultures may use a female or multi-gender name as male one and vice versa. People may have been referred to by others using the wrong gender assumptions and didn't raise the issue for several reasons. However, we think these are reasonable levels of noise and accept them in sake for this classification.

## Links

We extracted all links in the posts unless it is a linkpost (a post that only contains one link to another website). You can use the links for e.g. a citation graph analysis.

## Table Columns

The forum post nodes tables now contain.

- `_id`:    LessWrong article ID
- `title`:  Title
- `slug`:   Slug, i.e. the-article's-title-in-short e.g. to use in a link
- `pageUrl`:    Url which is https://www.lesswrong.com/posts/*_id*/*slug*
- `postedAt`:   Date of posting
- `baseScore`:  LessWrong score (users can upvote and downvote articles)
- `voteCount`:  How many users upvoted and downvoted and article
- `commentCount`:   Number of comments
- `meta`:   If it is a meta post (bool)
- `question`:  If it is a question-type (bool)
- `htmlBody`:   The original HTML
- `user.username`:  Author username (internal but showed in case there is no display name)
- `user.slug`:  Author username slug
- `user.displayName`:   Authro display name to be shown in the website (optional)
- `is_linkpost`:    If this post is a link post
- `extracted_links`:    List of extracted links, first one is skipped if it's a linkpost
- `cleaned_htmlBody`:   Pure text without any HTML tags
- `user_gender`:    Classified user gender (see more under Gender Classification)
- `topic_cluster_id`:   ID of the identified *dominant* topic
- `topic_label`:    Label of the topic (we manually assigned the labels to the topics)

**Version:** 1.0  
**Last Updated:** October 2025 
**Citation:** *forthcoming*

## License
GNU GENERAL PUBLIC LICENSE V. 3

## Contact
ninelloldenburg@gmail.com