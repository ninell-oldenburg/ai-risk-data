import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import glob
from typing import List
from collections import Counter
import names

class ExtractLinksAndGender:
    def __init__(self):
        self.MALE_NAMES = names.MALE_NAMES
        self.FEMALE_NAMES = names.FEMALE_NAMES

    def is_linkpost(row) -> bool:
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
    
    def clean_html(html_content: str) -> str:
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
        
    def extract_links_from_html(html_content: str) -> List[str]:
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
                        href = 'https://lesswrong.com' + href
                    links.append(href)
            
            return links
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return []
        
    def extract_gender_from_username(self, username: str, display_name: str = None) -> str:
        """Attempt to extract gender from username and display name"""
        if pd.isna(username):
            return 'unknown'
        
        # combine username and display name for analysis
        text_to_analyze = str(username).lower()
        if pd.notna(display_name):
            text_to_analyze += ' ' + str(display_name).lower()
        
        # all names and sort by length (longest first)
        # this prevents one name being classified as different gender 
        # when it contains another gender's name, e.g.  pauline as paul
        all_names = list(self.FEMALE_NAMES) + list(self.MALE_NAMES)
        all_names_sorted = sorted(all_names, key=len, reverse=True)
        
        for name in all_names_sorted:
            if name in text_to_analyze:
                if name in self.FEMALE_NAMES:
                    return 'female'
                else: 
                    return 'male'
        
        return 'unknown'
    
    def process_csv_file(self, in_filepath: str, out_filepath: str) -> None:
        """Process a single CSV file and add extracted links column"""
        try:
            df = pd.read_csv(in_filepath)

            print(out_filepath)
            
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
            
            # save back to the same file
            df.to_csv(out_filepath, index=False)
            
            # return counts for summary
            posts_with_links = (df['extracted_links'] != '').sum()
            linkposts = df['is_linkpost'].sum()
            
            return posts_with_links, linkposts, unknown_names, usrs
        
        except Exception as e:
            print(f"Error processing {in_filepath}: {e}")
            return 0, 0, 0, 0
        
def main():
    extractor = ExtractLinksAndGender()
    base_path_in = "../data/lw_csv"
    base_path_out = "../data/lw_csv_cleaned"
    total_posts_with_links = 0
    total_linkposts = 0
    files_processed = 0
    unknown_names = Counter()
    usrs = Counter()

    # Find all CSV files matching the pattern
    csv_files_in = []
    csv_files_out = []
    for year in range(2016, 2026):  # 2016 to 2025
        year_path_in = os.path.join(os.path.join(base_path_in, str(year)), "*.csv")
        csv_files_in.extend(glob.glob(year_path_in))

        for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            file_path_out = os.path.join(os.path.join(base_path_out, str(year)), f"{year}-{month}.csv")
            csv_files_out.append(file_path_out)

    print(f"Found {len(csv_files_in)} CSV files to process")

    # Process each file and update it in place
    for i, csv_file_in in enumerate(sorted(csv_files_in)):
        print(f"Processing {i+1}/{len(csv_files_in)}: {csv_file_in}")
        posts_with_links, linkposts, unknowns, file_usrs = extractor.process_csv_file(csv_file_in, csv_files_out[i])
        unknown_names += unknowns
        usrs += file_usrs
        
        total_posts_with_links += posts_with_links
        total_linkposts += linkposts
        files_processed += 1
        print(f"  ✓ Updated with {posts_with_links} posts with links, {linkposts} linkposts")

    print(f"\nCompleted!")
    print(f"Files processed: {files_processed}/{len(csv_files_out)}")
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
    main()