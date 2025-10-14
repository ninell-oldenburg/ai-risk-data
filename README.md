# LessWrong 10 Years Dataset

This is a LessWrong Dataset for the 10 years from 2016 to 2025. I contains all articles and metadata about those articles.
You can use our code to generate the dataset locally (and tweak the metadata) using the `main.py` file. You can visualize
the generated data with the notebook provided, `visualize.ipynb`.

## Data and Metadata

You can find the csv data in `data/lw_csv_cleaned_topic/` which means that the json files were converted into CSVs, cleaned (i.e. gender classified and text and links extracted from the HTML) and a topic clustering has been performed and also appended to the CSVs. All other folders in `data/` are one of the preceeding steps to reach this final CSV.

In each CSV, you can now find:

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
- `user.username`:  Username (internal but showed in case there is no display name)
- `user.slug`:  Username lowercase and with hyphens instead of whitespace
- `user.displayName`:   Display name to be shown in the website (optional)
- `is_linkpost`:    If this post starts with "This is a linkpost" (bool)
- `extracted_links`:    List of extracted links, first one is skipped if it's a linkpost
- `cleaned_htmlBody`:   Pure text without any HTML tags
- `user_gender`:    Classified user gender (see more under Gender Classification)
- `topic_cluster_id`:   ID of the identified *dominant* topic (-2 to 35)
- `topic_label`:    Label of the topic (we manually assigned the labels to the topics)

## Gender Classification

We classified the gender in different ways. Primarily, we started using the first name by generating a list of common first names and classifying the usernames on whether they contain a female or male first name. This works mostly well as many LessWrong users have a firstname-lastname (or firstnamelastname) style username. For this, we checked with the list of common names from long to short names to prevent one name being classified as different gender when it contains another gender's name, e.g. Pauline (female) as Paul (male) when paul would be checked first.

We then checked all users that were not catched by this inital classification. Especially, we checked all authors who wrote five or more posts for identifiers of their gender. This took different ways: some users refer ot their personal website, blog, or twitter account where we could see their name; others referred to themselves in their description as "male"/"guy" or "female"/"girl" or said that they don't identify with genders, etc. Lastly, we checked in the comments that this user was mentioned for pronouns and took this pronoun if the user themselves didn't complain about it in a later comment. The reasoning here is that either the commenting user and the user under question have met in real life and so know of each other's gender or the assumed it and the user under question accepted this assumption.

Users who deleted their account are also classified as "unknown". Users with ambiguously gendered names such as Logan or Kim unless they have more than five posts and self-identified gender noted anywhere.

We are well aware that this classification is still noisy from different aspects. People may change their gender and still have their old username. Different cultures may use a female or multi-gender name as male one and vice versa. People may have been referred to by others using the wrong gender assumptions and didn't raise the issue for several reasons. However, we think these are reasonable levels of noise and accept them in sake for this classification.

## Links

We extracted all links in the posts unless it is a linkpost (a post that only contains one link to another website). You can use the links for e.g. a citation graph analysis.

# AI Risk Research Network Data

**Version:** 1.0  
**Last Updated:** October 2024  
**Citation:** [Your paper citation]

## Overview
This dataset contains a network analysis of AI risk research communities (LessWrong, Alignment Forum) and their citations to academic literature.

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

### Known Issues
- Gender inference coverage: 78% of authors
- Some historical posts may have incomplete text
- Citation extraction limited to hyperlinks in post body

### Excluded Data
- Comments (not included)
- Draft posts
- Deleted or private posts

## Usage

### Loading Data
```python
import pandas as pd

posts = pd.read_csv('data/nodes/nodes_posts.csv')
citations = pd.read_csv('data/edges/edges_post_citations.csv')

Reproducibility
To reproduce this dataset from scratch:
bashpip install -r requirements.txt
python src/01_data_collection/scrape_lesswrong.py
python src/02_preprocessing/clean_and_enrich.py
python src/03_graph_construction/build_graph_tables.py
See docs/methodology.md for detailed pipeline documentation.
License
[Your license]
Contact
[Your email]
Acknowledgments

LessWrong/Alignment Forum for data access
OpenAlex for bibliometric data