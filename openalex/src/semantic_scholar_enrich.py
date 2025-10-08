import pandas as pd
import requests
import time
from typing import Optional, Dict
import json
import glob
import os

class SemanticScholarEnricher:
    """Enriches OpenAlex papers with Semantic Scholar abstracts and full text links."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize the enricher.
        
        Args:
            rate_limit_delay: Delay between API calls in seconds (default 1.0)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
    def get_paper_by_doi(self, doi: str) -> Optional[Dict]:
        """Fetch paper data from Semantic Scholar using DOI."""
        try:
            url = f"{self.BASE_URL}/DOI:{doi}"
            params = {
                'fields': 'title,abstract,openAccessPdf,isOpenAccess,externalIds'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                print(f"Error for DOI {doi}: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Exception for DOI {doi}: {str(e)}")
            return None
    
    def get_paper_by_title(self, title: str) -> Optional[Dict]:
        """Search for paper by title on Semantic Scholar."""
        try:
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': title,
                'limit': 1,
                'fields': 'title,abstract,openAccessPdf,isOpenAccess,externalIds'
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    return data['data'][0]
            return None
                
        except Exception as e:
            print(f"Exception searching for title '{title[:50]}...': {str(e)}")
            return None
    
    def enrich_paper(self, row: pd.Series, doi_col: str = 'doi', 
                     title_col: str = 'title') -> Dict:
        """
        Enrich a single paper row with Semantic Scholar data.
        
        Args:
            row: Pandas Series representing a row from the CSV
            doi_col: Name of the DOI column
            title_col: Name of the title column
            
        Returns:
            Dictionary with enriched data
        """
        result = {
            's2_abstract': None,
            's2_pdf_url': None,
            's2_is_open_access': None,
            's2_found_by': None
        }
        
        # Try DOI first
        if doi_col in row and pd.notna(row[doi_col]) and row[doi_col]:
            paper_data = self.get_paper_by_doi(row[doi_col])
            if paper_data:
                result['s2_found_by'] = 'doi'
                result['s2_abstract'] = paper_data.get('abstract')
                result['s2_is_open_access'] = paper_data.get('isOpenAccess')
                
                if paper_data.get('openAccessPdf'):
                    result['s2_pdf_url'] = paper_data['openAccessPdf'].get('url')
                
                return result
        
        # Fall back to title search
        if title_col in row and pd.notna(row[title_col]) and row[title_col]:
            paper_data = self.get_paper_by_title(row[title_col])
            if paper_data:
                result['s2_found_by'] = 'title'
                result['s2_abstract'] = paper_data.get('abstract')
                result['s2_is_open_access'] = paper_data.get('isOpenAccess')
                
                if paper_data.get('openAccessPdf'):
                    result['s2_pdf_url'] = paper_data['openAccessPdf'].get('url')
        
        return result
    
    def enrich_csv(self, input_csv: str, output_csv: str, 
                   doi_col: str = 'doi', title_col: str = 'title',
                   start_row: int = 0, max_rows: Optional[int] = None):
        """
        Enrich an entire CSV file with Semantic Scholar data.
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output CSV file
            doi_col: Name of the DOI column
            title_col: Name of the title column
            start_row: Row to start from (useful for resuming)
            max_rows: Maximum number of rows to process (None for all)
        """
        print(f"Loading {input_csv}...")
        df = pd.read_csv(input_csv)
        
        print(f"Found {len(df)} papers")
        
        # Initialize new columns if they don't exist
        for col in ['s2_abstract', 's2_pdf_url', 's2_is_open_access', 's2_found_by']:
            if col not in df.columns:
                df[col] = None
        
        # Determine range to process
        end_row = min(start_row + max_rows, len(df)) if max_rows else len(df)
        
        print(f"Processing rows {start_row} to {end_row}...")
        
        for idx in range(start_row, end_row):
            if idx % 10 == 0:
                print(f"Progress: {idx}/{end_row} ({100*idx/end_row:.1f}%)")
                # Save progress periodically
                df.to_csv(output_csv, index=False)
            
            row = df.iloc[idx]
            enriched = self.enrich_paper(row, doi_col, title_col)
            
            for key, value in enriched.items():
                df.at[idx, key] = value
        
        print(f"Saving enriched data to {output_csv}...")
        df.to_csv(output_csv, index=False)
        
        # Print summary statistics
        found_count = df['s2_found_by'].notna().sum()
        abstract_count = df['s2_abstract'].notna().sum()
        pdf_count = df['s2_pdf_url'].notna().sum()
        
        print(f"\nEnrichment complete!")
        print(f"Papers found on Semantic Scholar: {found_count}/{len(df)}")
        print(f"Abstracts found: {abstract_count}")
        print(f"PDF links found: {pdf_count}")


# Example usage
if __name__ == "__main__":
    enricher = SemanticScholarEnricher(rate_limit_delay=1.0)

    root_dir = "openalex/data/csv/"

    csv_files = []
    for year in range(2015, 2025):
        year_dir = os.path.join(root_dir, str(year))

        year_input_files = glob.glob(os.path.join(year_dir, "*.csv"))
    
    for file in year_input_files:
        enricher.enrich_csv(
            input_csv=file,
            output_csv=file,
            doi_col='doi',
            title_col='title',
            start_row=0,  # Start from beginning
            max_rows=None  # Process all rows (set to a number to test first)
        )
    
    # To resume from row 100 if interrupted:
    # enricher.enrich_csv('openalex_papers.csv', 'openalex_enriched.csv', start_row=100)