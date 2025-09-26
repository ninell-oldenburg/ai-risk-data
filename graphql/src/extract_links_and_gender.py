import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import glob
from typing import List, Optional
from collections import Counter
import src.names as names
import nomquamgender as nqg
import json
import sys
import kagglehub
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtractLinksAndGender:
    def __init__(self, platform):
        try:
            if platform in ['lw', 'af']:
                self.platform = platform
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")

        self.MALE_NAMES = names.MALE_NAMES
        self.FEMALE_NAMES = names.FEMALE_NAMES
        self.nqgmodel = nqg.NBGC()
        self.arxiv_data = {}
        self.arxiv_pattern = re.compile(r'arxiv\.org/(?:abs/|pdf/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)', re.IGNORECASE)
        self.doi_pattern = re.compile(r'(?:doi:|doi\.org/|dx\.doi\.org/)([0-9]{2}\.[0-9]{4,}/[^\s,;)]+)', re.IGNORECASE)

    def is_linkpost(self, row) -> bool:
        """Check if a post is marked as a linkpost by looking for 'this is a linkpost' at the beginning"""
        html_content = row.get('htmlBody')
        if pd.isna(html_content):
            return False
        
        try:
            # parse HTML and get text content
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text().strip().lower()
            
            # check if it starts with "this is a linkpost"
            return ('this is a linkpost') in text_content
        
        except Exception as e:
            print(f"Error checking linkpost status: {e}")
            return False
    
    def clean_html(self, html_content: str) -> str:
        """Extract plain text from HTML, removing all tags and styling"""
        if pd.isna(html_content):
            return ''
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # plain text, strip extra whitespace
            text = soup.get_text()
            # clean up whitespace (replace multiple spaces/newlines with single spaces)
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
            
            # all anchor tags with href attributes
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # skip empty, anchor-only, or javascript links
                if href and not href.startswith('#') and not href.startswith('javascript:'):
                    # clean up relative URLs if needed
                    if href.startswith('//'):
                        href = 'https:' + href
                    elif href.startswith('/'):
                        if self.platform == 'lw':
                            href = 'https://lesswrong.com' + href
                        elif self.platform == 'af':
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
        
        # Find the JSON file in the dataset
        dataset_dir = Path(dataset_path)
        json_files = list(dataset_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError("No JSON file found in the arXiv dataset")
        
        json_file = json_files[0]  # Assume the first JSON file is the main dataset
        logger.info(f"Loading data from: {json_file}")
        
        # Load and index the arXiv data by ID
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
                
                # Log progress every 100k entries
                if line_num % 100000 == 0:
                    logger.info(f"Loaded {line_num} entries...")
        
        logger.info(f"Loaded {len(self.arxiv_data)} arXiv papers")
    
    def extract_arxiv_ids(self, text: str) -> List[str]:
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        matches = self.arxiv_pattern.findall(text)
        # remove version numbers (e.g., v1, v2) to match the base ID
        clean_ids = [match.split('v')[0] for match in matches]
        return list(set(clean_ids))
    
    def extract_direct_dois(self, text: str) -> List[str]:
        """
        Extract DOIs directly from text.
        
        Args:
            text (str): Text potentially containing DOI links
            
        Returns:
            List[str]: List of extracted DOIs
        """
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        matches = self.doi_pattern.findall(text)
        clean_dois = []
        for doi in matches:
            doi = re.sub(r'[.,;:)\]}>"\'\s]+$', '', doi)
            if doi:
                clean_dois.append(doi)
        
        return list(set(clean_dois)) 
    
    def extract_all_dois(self, text: str) -> List[str]:
        """
        Extract all DOIs from text: both direct DOIs and DOIs from arXiv links.
        
        Args:
            text (str): Text potentially containing links
            
        Returns:
            List[str]: List of all extracted DOIs
        """
        all_dois = []
        
        # Extract direct DOIs
        direct_dois = self.extract_direct_dois(text)
        all_dois.extend(direct_dois)
        
        # Extract DOIs from arXiv links
        arxiv_ids = self.extract_arxiv_ids(text)
        for arxiv_id in arxiv_ids:
            doi = self.get_doi_for_arxiv_id(arxiv_id)
            if doi:
                all_dois.append(doi)

        return list(set(all_dois))
        
    def get_doi_for_arxiv_id(self, arxiv_id: str) -> Optional[str]:
        """
        Get DOI for a given arXiv ID.
        
        Args:
            arxiv_id (str): arXiv paper ID
            
        Returns:
            Optional[str]: DOI if found, None otherwise
        """
        paper = self.arxiv_data.get(arxiv_id)
        if paper:
            doi = paper.get('doi', '')
            return doi if doi else None
        return None
        
    def extract_gender_from_username(self, username: str, display_name: str = None) -> str:
        """Attempt to extract gender from username and display name"""
        if pd.isna(username):
            return 'unknown'
        
        # First, try to extract individual components from username
        username_parts = self._split_username(str(username))
        
        # Check each part against name lists
        for part in username_parts:
            gender = self._check_name_in_lists(part.lower())
            if gender != 'unknown':
                return gender
        
        # If no parts matched, try display name parts
        if pd.notna(display_name):
            display_parts = self._split_username(str(display_name))
            for part in display_parts:
                gender = self._check_name_in_lists(part.lower())
                if gender != 'unknown':
                    return gender
        
        # If still no match, fall back to original method (substring search)
        text_to_analyze = str(username).lower()
        if pd.notna(display_name):
            text_to_analyze += ' ' + str(display_name).lower()
        
        # Sort by length (longest first) to prevent shorter names matching within longer ones
        all_names = list(self.FEMALE_NAMES) + list(self.MALE_NAMES)
        all_names_sorted = sorted(all_names, key=len, reverse=True)
        
        for name in all_names_sorted:
            if len(name) > 3 and name in text_to_analyze:
                if name in self.FEMALE_NAMES:
                    return 'female'
                elif name in self.MALE_NAMES:
                    return 'male'
        
        return 'unknown'

    def _split_username(self, username: str) -> list:
        """Split username into individual components using various methods"""
        parts = []
        
        # Method 1: Split by common separators
        separated_parts = re.split(r'[_\-\.\s]+', username)
        parts.extend([part for part in separated_parts if len(part) > 1])
        
        # Method 2: Split camelCase
        camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', username)
        parts.extend([part for part in camel_parts if len(part) > 1])
        
        # Method 3: Split by numbers (keep the text parts)
        number_split = re.split(r'\d+', username)
        parts.extend([part for part in number_split if len(part) > 1])
        
        # Remove duplicates and filter out very short parts
        unique_parts = list(set(parts))
        filtered_parts = [part for part in unique_parts if len(part) >= 2]
        
        # If no good parts found, return the original username
        if not filtered_parts:
            filtered_parts = [username]
        
        return filtered_parts

    def _check_name_in_lists(self, name: str) -> str:
        """Check if a name exists in the gender name lists"""
        if name in self.FEMALE_NAMES:
            return 'female'
        elif name in self.MALE_NAMES:
            return 'male'
        else:
            return 'unknown'
    
    def process_csv_file(self, in_filepath: str, out_filepath: str) -> None:
        """Process a single CSV file and add extracted links column"""
        try:
            df = pd.read_csv(in_filepath)
            
            # add new columns for link information
            df['is_linkpost'] = df.apply(self.is_linkpost, axis=1)
            df['extracted_links'] = ''
            df['cleaned_htmlBody'] = ''
            df['user_gender'] = ''
            unknown_names = Counter()
            usrs = Counter()
            
            for idx, row in df.iterrows():
                # clean the HTML content to plain text
                cleaned_text = self.clean_html(row.get('htmlBody'))
                df.at[idx, 'cleaned_htmlBody'] = cleaned_text
                
                # extract gender from username
                gender = self.extract_gender_from_username(row.get('user.username'), row.get('user.displayName'))
                df.at[idx, 'user_gender'] = gender
                unknown_names[gender] += 1
                usrs[row.get('user.username')] += 1
                
                # extract links from htmlBody
                html_links = self.extract_links_from_html(row.get('htmlBody'))
                
                # if it's a linkpost, skip the first link
                if row['is_linkpost'] and html_links:
                    df.at[idx, 'extracted_links'] = '; '.join(html_links[1:])  # Skip first link
                else:
                    df.at[idx, 'extracted_links'] = '; '.join(html_links)

            df['extracted_dois'] = df['extracted_links'].apply(self.extract_all_dois)
            
            # save back to the same file
            df.to_csv(out_filepath, index=False)
            
            # return counts for summary
            posts_with_links = (df['extracted_links'] != '').sum()
            linkposts = df['is_linkpost'].sum()
            
            return posts_with_links, linkposts, unknown_names, usrs
        
        except Exception as e:
            print(f"Error processing {in_filepath}: {e}")
            return 0, 0, 0, 0
        
def main(forum):
    extractor = ExtractLinksAndGender(platform=forum)
    base_path_in = f"graphql/data/{extractor.platform}/csv"
    base_path_out = f"graphql/data/{extractor.platform}/csv_cleaned"
    total_posts_with_links = 0
    total_linkposts = 0
    files_processed = 0
    unknown_names = Counter()
    usrs = Counter()

    csv_file_pairs = []
    for year in range(2016, 2026):
        input_dir = os.path.join(base_path_in, str(year))
        output_dir = os.path.join(base_path_out, str(year))
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find actual input files for this year
        year_input_files = glob.glob(os.path.join(input_dir, "*.csv"))
        
        for input_file in year_input_files:
            # Extract filename and create corresponding output path
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, filename)
            csv_file_pairs.append((input_file, output_file))

    print(f"Found {len(csv_file_pairs)} CSV files to process")

    # Process each pair
    for i, (csv_file_in, csv_file_out) in enumerate(sorted(csv_file_pairs)):
        print(f"Processing {i+1}/{len(csv_file_pairs)}: {csv_file_in} -> {csv_file_out}")
        posts_with_links, linkposts, unknowns, file_usrs = extractor.process_csv_file(csv_file_in, csv_file_out)
        unknown_names += unknowns
        usrs += file_usrs
        
        total_posts_with_links += posts_with_links
        total_linkposts += linkposts
        files_processed += 1
        print(f"✅ Updated with {posts_with_links} posts with links, {linkposts} linkposts")

    print(f"\nCompleted!")
    print(f"Files processed: {files_processed}/{len(csv_file_pairs)}")
    print(f"Total posts with extracted links: {total_posts_with_links}")
    print(f"Total linkposts identified: {total_linkposts}")
    print(f"\nEach CSV file now has these new columns added:")
    print(f"  - 'is_linkpost': Boolean for linkpost detection")
    print(f"  - 'extracted_links': Citation links (semicolon-separated)")
    print(f"  - 'cleaned_htmlBody': Plain text with all HTML removed")

    print(f'Total users: {len(usrs)}')
    print(f'Unknown names: {len(unknown_names)}')
    for key, value in unknown_names.items():
        print(f'{key}: {value}')

    """
    # USE THIS TO PRINT OUT UNKNOWN USERNAMES
    for key, value in unknown_names.items():
        if value > 4:
            print(key)
    """

if __name__ == "__main__":    
    main(sys.argv[1])