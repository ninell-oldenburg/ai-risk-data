import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import glob
from collections import Counter
import nomquamgender as nqg
import json
import sys
import kagglehub
from pathlib import Path
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtractLinksAndGender:
    def __init__(self, platform):
        try:
            if platform in ['lw', 'af']:
                self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")

        with open("src/metadata/graphql_usernames.json", "r", encoding="utf-8") as f:
            names_data = json.load(f)

        self.MALE_USERNAMES = names_data["MALE_USERNAMES"]
        self.FEMALE_USERNAMES = names_data["FEMALE_USERNAMES"]
        self.GENDER_TERMS = {'male': 'gm', 'female': 'gf'}
        self.nqgmodel = nqg.NBGC()
        self.nqgmodel.threshold = .2
        self.arxiv_data = {}
        self.arxiv_pattern = re.compile(r'arxiv\.org/(?:abs|pdf|ps|html|format)/([a-z\-]+/\d+|\d+\.\d+)', re.IGNORECASE)
        self.doi_pattern = re.compile(r'10\.\d{4,}/[^\s,;|\]}\)"\'\><\n]+', re.IGNORECASE)
    
    def clean_html(self, html_content: str) -> str:
        """Extract plain text from HTML, removing all tags and styling"""
        if pd.isna(html_content):
            return ''
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # plain text, strip extra whitespace
            text = soup.get_text()
            # clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            return ''
        
    def extract_links_from_html(self, html_content: str) -> List[str]:
        """Extract all links from HTML content"""
        if pd.isna(html_content):
            return []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # skip empty, anchor-only, or javascript links
                if href and not href.startswith('#') and not href.startswith('javascript:'):
                    # clean up relative URLs if needed
                    if href.startswith('//'):
                        href = 'https:' + href
                    elif href.startswith('/'):
                        if self.platform == 'lesswrong':
                            href = 'https://lesswrong.com' + href
                        elif self.platform == 'alignment_forum':
                            href = 'https://alignmentforum.org' + href
                    links.append(href)
            
            return links
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []
    
    def download_arxiv_dataset(self) -> str:
        logger.info("Downloading arXiv dataset from Kaggle...")
        path = kagglehub.dataset_download("Cornell-University/arxiv")
        logger.info(f"Dataset downloaded to: {path}")
        return path
    
    def load_arxiv_data(self, dataset_path: str):
        logger.info("Loading arXiv dataset...")
        
        # find JSON file
        dataset_dir = Path(dataset_path)
        json_files = list(dataset_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError("No JSON file found in the arXiv dataset")
        
        json_file = json_files[0]  # assume the first JSON file is the main dataset
        logger.info(f"Loading data from: {json_file}")
        
        # load and index the arXiv data by ID
        with open(json_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    paper = json.loads(line.strip())
                    paper_id = paper.get('id', '').strip()
                    if paper_id:
                        self.arxiv_data[paper_id] = paper
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                
                if line_num % 100000 == 0:
                    logger.info(f"Loaded {line_num} entries...")
        
        logger.info(f"Loaded {len(self.arxiv_data)} arXiv papers")
    
    def extract_arxiv_ids(self, text: str) -> List[str]:
        """Extract arXiv IDs from text"""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        matches = self.arxiv_pattern.findall(text)
        # remove version numbers (e.g., v1, v2) to match the base ID
        clean_ids = [match.split('v')[0] for match in matches]
        return list(set(clean_ids))
    
    def clean_doi(self, doi):
        """
        Comprehensive DOI cleaning for matching.
        Handles all the weird edge cases we've found.
        """
        if not doi or pd.isna(doi):
            return None
        
        doi = str(doi).strip()
        
        # 1. remove URL prefixes
        doi = doi.replace('https://doi.org/', '')
        doi = doi.replace('http://doi.org/', '')
        doi = doi.replace('https://dx.doi.org/', '')
        doi = doi.replace('http://dx.doi.org/', '')
        doi = doi.replace('doi:', '')

        # 2. remove fragments and query parameters
        if '#' in doi:
            doi = doi.split('#')[0]
        if '?' in doi:
            doi = doi.split('?')[0]
        
        # 3. remove HTML entities and special characters
        doi = doi.replace('&amp;', '').replace('&amp', '')
        # normalize en-dashes and em-dashes to regular hyphens
        doi = doi.replace('–', '-').replace('—', '-')
        doi = re.sub(r'%[0-9a-f]{2}', '', doi, flags=re.IGNORECASE)  # URL-encoded chars
        # incomplete parenthetical patterns
        doi = re.sub(r'\(\d{2}\)$', '', doi)
        doi = re.sub(r'[†‌—]', '', doi)  # special unicode
        doi = re.sub(r'&type=.*$', '', doi)
        doi = re.sub(r'&.*$', '', doi)
        # sequences of dashes (en-dash, em-dash, regular dash)
        doi = re.sub(r'\.?[-—–]{2,}$', '', doi)
        # trailing spaces
        doi = doi.strip()
        # zero-width and other invisible unicode characters
        doi = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060\ufeff]', '', doi)
        
        # 4. remove file extensions and path suffixes
        doi = re.sub(r'\.pdf.*$', '', doi)
        doi = re.sub(r'\.(full\.pdf|pdf\.full).*$', '', doi)
        doi = re.sub(r'/abstract$', '', doi)
        doi = re.sub(r'/pdf$', '', doi)
        doi = re.sub(r'/epdf$', '', doi)
        doi = re.sub(r'/full$', '', doi)
        doi = re.sub(r'/issuetoc$', '', doi)
        doi = re.sub(r'/full/html$', '', doi)
        doi = re.sub(r'/meta$', '', doi)
        doi = re.sub(r'/tables/\d+$', '', doi)
        doi = re.sub(r'/suppl_file/.*$', '', doi)
        doi = re.sub(r'/full\.pdf$', '', doi)
        
        # 5. remove version indicators
        doi = re.sub(r'v\d+\.full.*$', '', doi)  # v1.full.pdf etc
        doi = re.sub(r'v\d+\.full$', '', doi)    # v1.full
        doi = re.sub(r'v\d+$', '', doi)          # v1
        
        # 6. remove bracket and parenthesis artifacts
        doi = re.sub(r'\[[^\]]*$', '', doi)  # incomplete brackets
        doi = re.sub(r'\([^)]*$', '', doi)   # incomplete parentheses
        if doi.endswith('('):
            doi = doi[:-1]
        
        # 7. remove duplicate acprof, acrefore, acref segments
        if '/acrefore/' in doi or '/acref/' in doi:
            match = re.match(r'(10\.1093/acr(?:efore|ef)/\d+(?:\.\d+)*)', doi)
            if match:
                doi = match.group(1)
        
        # 8. remove caret suffixes
        doi = re.sub(r'\.\^.*$', '', doi)
        doi = re.sub(r'\^.*$', '', doi)
        
        # 9. remove trailing text patterns
        # multiple hyphenated words
        doi = re.sub(r'/[a-z]+-[a-z]+-[a-z]+-.*$', '', doi, flags=re.IGNORECASE)
        # single hyphenated fragment
        doi = re.sub(r'/[a-z]+-[a-z]+-?$', '', doi, flags=re.IGNORECASE)
        # word suffixes (gödel, kraft, etc.)
        doi = re.sub(r'\.?[a-zàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿłćśźżğ]{4,}$', '', doi, flags=re.IGNORECASE)
        # text concatenated after digit endings (no separator)
        doi = re.sub(r'(\d)[a-z]{4,}(-[a-z]+)*$', r'\1', doi, flags=re.IGNORECASE)
        # trailing lone hyphens
        doi = doi.rstrip('-')
        # footnote markers
        doi = re.sub(r'\.footnotes\*\d+$', '', doi)
        # possessive markers and contractions at the end
        doi = re.sub(r"(what's|that's|it's|what’s|that’s|it’s|[a-z]+'s?)$", '', doi, flags=re.IGNORECASE)
        # .cross- and similar suffix patterns
        doi = re.sub(r'\.cross-$', '', doi)
        # 2-3 letter word fragments at the end (but not legitimate suffixes like .x or .e1234)
        doi = re.sub(r'\.([a-z]{2,3})$', lambda m: '' if not any(c.isdigit() for c in m.group(1)) else m.group(0), doi, flags=re.IGNORECASE)
        # article title paths (pattern: /numbers/long-title-text)
        doi = re.sub(r'/\d+/[a-z][a-z-]+-[a-z-]+$', '', doi, flags=re.IGNORECASE)
        # 2-3 letter author initials concatenated after numbers
        doi = re.sub(r'(\d)([a-z]{2,3})$', r'\1', doi, flags=re.IGNORECASE)
        
        # 10. clean last segment after final slash
        parts = doi.split('/')
        if len(parts) >= 2:
            last_part = parts[-1]
            last_part = re.sub(r'[a-z]{3,}$', '', last_part, flags=re.IGNORECASE)
            parts[-1] = last_part
            doi = '/'.join(parts)
        
        # 11. remove trailing special characters
        doi = re.sub(r'[\.&]+$', '', doi)  # ampersands and periods
        # trailing underscores (multiple)
        doi = re.sub(r'_+$', '', doi)
        # /full one more time before final cleanup
        doi = re.sub(r'/full$', '', doi)
        doi = doi.rstrip("/.,;:!?_+'\"‘")
        
        # 12. final version check
        doi = re.sub(r'v\d+$', '', doi)
        doi = re.sub(r'/full$', '', doi)
        doi = re.sub(r'v\d+\.full$', '', doi)
        
        # 13. lowercase
        doi = doi.lower()
        
        # 14. validate completeness
        if doi.endswith('-'):
            return None
        
        if '/' in doi:
            parts = doi.split('/')
            if len(parts[-1]) < 2:
                return None
            
        # filter out DOIs that are just the prefix
        if '/' not in doi or doi.split('/')[-1] == '':
            return None
        
        # filter out DOIs where the last part looks incomplete
        if '/' in doi:
            last_part = doi.split('/')[-1]
            if len(last_part) <= 2 or last_part.endswith('-'):
                return None
        
        return doi.strip() if doi else None
        
    def extract_direct_dois(self, text: str) -> List[str]:
        """Extract DOIs directly from text."""
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        matches = self.doi_pattern.findall(text)
        cleaned_dois = []
        for doi in matches:
            cleaned_doi = self.clean_doi(doi)
            if cleaned_doi and re.match(r'^10\.\d{4,}/.+', cleaned_doi):
                cleaned_dois.append(cleaned_doi)
        
        return list(set(cleaned_dois)) 
    
    def extract_all_dois(self, text: str) -> List[str]:
        """Extract all DOIs from text: both direct DOIs and DOIs from arXiv links."""
        all_dois = []
        
        direct_dois = self.extract_direct_dois(text)
        all_dois.extend(direct_dois)
        
        arxiv_ids = self.extract_arxiv_ids(text)
        for arxiv_id in arxiv_ids:
            doi = self.get_doi_for_arxiv_id(arxiv_id)
            if doi:
                all_dois.append(doi)

        return list(set(all_dois))
        
    def get_doi_for_arxiv_id(self, arxiv_id: str) -> Optional[str]:
        """Get DOI for a given arXiv ID."""
        paper = self.arxiv_data.get(arxiv_id)
        if paper:
            doi = paper.get('doi', '')
            return doi if doi else None
        return None
        
    def extract_gender_from_username(self, username: str, display_name: str = None) -> str:
        """Attempt to extract gender from username and display name"""
        if pd.isna(username):
            return 'nan'
        
        text_to_analyze = str(username).lower()
        if pd.notna(display_name):
            text_to_analyze += ' ' + str(display_name).lower()
        
        all_usernames = list(self.FEMALE_USERNAMES) + list(self.MALE_USERNAMES)
        for name in all_usernames:
            if len(name) > 3 and name in text_to_analyze:
                if name in self.FEMALE_USERNAMES:
                    return 'gf'
                elif name in self.MALE_USERNAMES:
                    return 'gm'
        
        # try display name first as it often contains the real name
        if pd.notna(display_name):
            split_displayname = self._split_username(str(display_name))
            gender = self.nqgmodel.classify(split_displayname)
            if gender[0] != '-':
                return gender[0]

        # then try username
        split_username = self._split_username(str(username))
        gender = self.nqgmodel.classify(split_username)
        if gender[0] != '-':
            return gender[0]
        
        return '-'
    
    def _clean_title(self, title):
        """Clean title by removing line breaks and surrounding quotes."""
        if pd.isna(title):
            return ""
        
        title = str(title)
        # remove line breaks and tabs
        title = title.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # remove multiple spaces
        title = ' '.join(title.split())
        # remove surrounding quotes (both single and double)
        while title and title[0] in ('"', "'") and title[-1] in ('"', "'"):
            title = title[1:-1].strip()
        
        return title.strip()

    def _split_username(self, username: str) -> str:
        """Split username into individual components and return as lowercase string"""
        username = re.sub(r'^md\.\s*', '', username, flags=re.IGNORECASE)
        username = re.sub(r'\d+', ' ', username)

        parts = re.split(r'[_\-\.\s\@]+', username)
        
        # camelCase
        expanded_parts = []
        for part in parts:
            if not part:  # skip empty strings
                continue
            # split camelCase
            camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
            # consecutive capitals followed by lowercase
            camel_split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', camel_split)
            expanded_parts.append(camel_split)
        
        # lowercase and clean up extra spaces
        result = ' '.join(expanded_parts).lower().strip()
        # clean up any multiple spaces
        result = re.sub(r'\s+', ' ', result)
        
        return result if result else username.lower()
    
    def process_csv_file(self, in_filepath: str, out_filepath: str) -> None:
        """Process a single CSV file and add extracted links column"""
        try:
            df = pd.read_csv(in_filepath)
            
            # add new columns for link information
            df['extracted_links'] = ''
            df['cleaned_htmlBody'] = ''
            df['user_gender'] = ''
            df['extracted_dois'] = ''
            usrs = Counter()
            gender_dist = Counter()
            
            for idx, row in df.iterrows():
                df.at[idx, 'title'] = self._clean_title(row.get('title'))
                
                # get the HTML content
                html_content = row.get('htmlBody')
                
                # extract cleaned text
                cleaned_text = self.clean_html(html_content)
                df.at[idx, 'cleaned_htmlBody'] = cleaned_text
                
                # extract gender from username
                gender = self.extract_gender_from_username(row.get('user.username'), row.get('user.displayName'))
                df.at[idx, 'user_gender'] = gender
                gender_dist[gender] += 1
                usrs[row.get('user.username')] += 1
                
                # Extract links from HTML
                html_links = self.extract_links_from_html(html_content)
                
                # store links as JSON list
                df.at[idx, 'extracted_links'] = json.dumps(html_links) if html_links else '[]'
                
                # Extract DOIs from the original HTML content
                combined_text = str(html_content) + ' ' + cleaned_text
                dois = self.extract_all_dois(combined_text)
                
                # store DOIs as JSON list
                df.at[idx, 'extracted_dois'] = json.dumps(dois) if dois else '[]'

            # remove the htmlBody column
            df = df.drop(columns=['htmlBody'], errors='ignore')

            # save to output file
            df.to_csv(out_filepath, index=False)
            
            # return counts for summary
            posts_with_links = (df['extracted_links'] != '[]').sum()
            posts_with_dois = (df['extracted_dois'] != '[]').sum()
            
            return posts_with_links, posts_with_dois, usrs, gender_dist
        
        except Exception as e:
            print(f"Error processing {in_filepath}: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0, Counter(), Counter()
        
def main(forum):
    extractor = ExtractLinksAndGender(platform=forum)
    
    try:
        arxiv_path = extractor.download_arxiv_dataset()
        extractor.load_arxiv_data(arxiv_path)
    except Exception as e:
        logger.warning(f"Could not load arXiv dataset: {e}")
        logger.warning("Continuing without arXiv->DOI mapping")
    
    base_path_in = f"src/processed_data/{extractor.platform}/01_cleaned_csv"
    base_path_out = f"src/processed_data/{extractor.platform}/02_with_links_and_gender"
    total_posts_with_links = 0
    total_posts_with_dois = 0
    files_processed = 0
    usrs = Counter()
    gender_dist = Counter()

    csv_file_pairs = []
    years = sorted([
        int(name) for name in os.listdir(base_path_in)
        if os.path.isdir(os.path.join(base_path_in, name)) and name.isdigit()
    ])
    for year in years:
        input_dir = os.path.join(base_path_in, str(year))
        output_dir = os.path.join(base_path_out, str(year))
        
        # ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        year_input_files = glob.glob(os.path.join(input_dir, "*.csv"))
        
        for input_file in year_input_files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, filename)
            csv_file_pairs.append((input_file, output_file))

    print(f"Found {len(csv_file_pairs)} CSV files to process")

    for i, (csv_file_in, csv_file_out) in enumerate(sorted(csv_file_pairs)):
        print(f"Processing {i+1}/{len(csv_file_pairs)}: {csv_file_in} -> {csv_file_out}")
        posts_with_links, posts_with_dois, file_usrs, gender_dict = extractor.process_csv_file(csv_file_in, csv_file_out)
        usrs += file_usrs
        gender_dist += gender_dict
        
        total_posts_with_links += posts_with_links
        total_posts_with_dois += posts_with_dois
        files_processed += 1
        print(f"✅ Updated with {posts_with_links} posts with links, {posts_with_dois} posts with DOIs")

    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print(f"\nCompleted!")
    print(f"Files processed: {files_processed}/{len(csv_file_pairs)}")
    print(f"Total posts with extracted links: {total_posts_with_links}")
    print(f"Total posts with extracted DOIs: {total_posts_with_dois}")
    print(f"\nEach CSV file now has these new columns added:")
    print(f"  - 'extracted_links': All links as JSON list")
    print(f"  - 'extracted_dois': DOIs as JSON list")
    print(f"  - 'cleaned_htmlBody': Plain text with all HTML removed")
    print(f"  - 'user_gender': Predicted gender")
    print(f"\nThe 'htmlBody' column has been removed.")

    print()

    user_gender_map = {}
    for username, count in usrs.items():
        gender = extractor.extract_gender_from_username(username)
        user_gender_map[username] = gender
    
    user_dist = Counter(user_gender_map.values())

    print(f'Total users: {len(usrs)}')
    for key, value in user_dist.items():
        print(f'{key}: {value}')

    print()

    print('Inferred gender distribution of posts:')
    for key, value in gender_dist.items():
        print(f'{key}: {value}')
     
    """
    # UNCOMMENT THIS TO SEE SINGLE NAME - GENDER MAPPING
    for key, value in user_gender_map.items():
        print(f'{key}: {value}')
    """

if __name__ == "__main__":    
    if len(sys.argv) != 2:
        print("Usage: python graphql_02_extract_links_predict_gender.py <forum>")
        print("Where <forum> is 'lw' (LessWrong) or 'af' (Alignment Forum)")
        sys.exit(1)
        
    main(sys.argv[1])