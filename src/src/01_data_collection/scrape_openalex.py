import os
import time
from datetime import datetime
from typing import List, Dict
import requests
import pandas as pd
import glob
import matplotlib.pyplot as plt
from collections import Counter

TOPIC_IDS = {
    "Ethics & Social Impacts of AI": "T10883",
    "Adversarial Robustness in ML": "T11689",
    "Explainable AI": "T12026"
}

class AIScholarshipAnalyzer:
    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.headers = {'User-Agent': f'mailto:{email}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_top_concepts_by_topic(self, start_year=2000, end_year=2025, top_n=10):
        for topic_name, topic_id in TOPIC_IDS.items():
            print(f"\nFetching papers for topic: {topic_name} ({topic_id})")
            
            all_papers = []
            for year in range(start_year, end_year + 1):
                # Fetch papers for the year
                year_counts = self._fetch_year(topic_id, year)
                
                # Load the CSVs created for this year
                for filepath in year_counts.keys():
                    df = pd.read_csv(filepath)
                    all_papers.append(df)
            
            if not all_papers:
                print(f"⚠️ No papers found for {topic_name}")
                continue
            
            df_all = pd.concat(all_papers, ignore_index=True)
            
            # Flatten all concepts
            concept_list = []
            for concepts_str in df_all['concepts'].dropna():
                concept_list.extend([c.strip() for c in concepts_str.split(';') if c.strip()])
            
            counter = Counter(concept_list)
            top_concepts = counter.most_common(top_n)
            print(f"Top {top_n} concepts for {topic_name}:")
            for concept, count in top_concepts:
                print(f"  {concept}: {count}")
            
            # Plot
            concepts, counts = zip(*top_concepts)
            plt.figure(figsize=(10,6))
            plt.barh(concepts[::-1], counts[::-1], color='skyblue')
            plt.title(f"Top {top_n} Concepts in {topic_name}")
            plt.xlabel("Frequency")
            plt.tight_layout()
            plt.show()
        
    def get_and_save_articles(self, topic_id: str = "T10883", 
                            start_date: datetime = datetime(2000, 1, 1), 
                            end_date: datetime = datetime(2025, 6, 30)) -> Dict[str, int]:
        """
        Fetch ALL papers by iterating year-by-year (or month-by-month if needed).
        This bypasses the 10,000 result limit.
        
        Returns:
            Dictionary with statistics about saved papers
        """
        total_papers = 0
        saved_counts = {}
        
        for year in range(start_date.year, end_date.year + 1):
            if year == end_date.year:
                last_month = end_date.month
            else:
                last_month = 12
            print(f"\n{'='*60}")
            print(f"Processing year {year}...")
            print(f"{'='*60}")
            
            # First, check how many papers this year has
            count = self._get_paper_count(topic_id, year)
            print(f"Found {count} papers for {year}")
            
            if count > 10000:
                # If more than 10k, iterate by month
                print(f"⚠️  Year {year} has {count} papers (>10k limit). Splitting by month...")
                year_counts = self._fetch_year_by_month(topic_id, year, max_month=last_month)
            else:
                # Otherwise fetch the whole year at once
                year_counts = self._fetch_year(topic_id, year)
            
            # Update totals
            for filepath, count in year_counts.items():
                saved_counts[filepath] = saved_counts.get(filepath, 0) + count
                total_papers += count
        
        print(f"\n{'='*60}")
        print(f"✓ COMPLETE: Retrieved {total_papers} papers total")
        print(f"✓ Saved across {len(saved_counts)} files")
        print(f"{'='*60}")
        
        return saved_counts
    
    def _get_paper_count(self, topic_id: str, year: int) -> int:
        """Get count of papers for a specific year"""
        url = f"{self.base_url}/works"
        params = {
            'filter': f'publication_year:{year},type:article,topics.id:{topic_id}',
            'per-page': 1
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('meta', {}).get('count', 0)
        except requests.exceptions.RequestException as e:
            print(f"Error getting count for {year}: {e}")
            return 0
    
    def _fetch_year(self, topic_id: str, year: int) -> Dict[str, int]:
        """Fetch all papers for a single year"""
        papers_by_month = {}
        page = 1
        per_page = 200
        
        url = f"{self.base_url}/works"
        
        while True:
            params = {
                'filter': f'publication_year:{year},type:article,topics.id:{topic_id}',
                'per-page': per_page,
                'page': page,
                'select': 'id,doi,title,publication_year,publication_date,type,cited_by_count,concepts,authorships,topics,referenced_works,abstract_inverted_index,cited_by_api_url,counts_by_year'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                batch = data.get('results', [])
                
                if not batch:
                    break
                
                # Group by month
                for paper in batch:
                    pub_date = paper.get('publication_date')
                    if pub_date and len(pub_date) >= 7:
                        year_month = pub_date[:7]
                    else:
                        year_month = f"{year}-01"
                    
                    if year_month not in papers_by_month:
                        papers_by_month[year_month] = []
                    papers_by_month[year_month].append(paper)
                
                print(f"  Page {page}: {len(batch)} papers retrieved...")
                
                if len(batch) < per_page:
                    break
                    
                page += 1
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page} for year {year}: {e}")
                break
        
        return self._save_papers_by_month(papers_by_month)
    
    def _fetch_year_by_month(self, topic_id: str, year: int, max_month: int = 12) -> Dict[str, int]:
        """Fetch papers month-by-month for years with >10k papers"""
        all_saved_counts = {}
        
        for month in range(1, max_month + 1):
            month_str = f"{month:02d}"
            year_month = f"{year}-{month_str}"
            
            # Check count for this month
            count = self._get_month_count(topic_id, year, month)
            print(f"  {year_month}: {count} papers")
            
            if count == 0:
                continue
            
            if count > 10000:
                print(f"    ⚠️  Month {year_month} has {count} papers (>10k). This is unusual!")
                print(f"    ⚠️  You may need to split by day or filter further.")
            
            papers = self._fetch_month(topic_id, year, month)
            saved_counts = self._save_papers_by_month({year_month: papers})
            
            for filepath, file_count in saved_counts.items():
                all_saved_counts[filepath] = all_saved_counts.get(filepath, 0) + file_count
            
            time.sleep(0.2)  # Extra politeness between months
        
        return all_saved_counts
    
    def _get_month_count(self, topic_id: str, year: int, month: int) -> int:
        """Get count of papers for a specific month"""
        url = f"{self.base_url}/works"
        month_str = f"{month:02d}"
        params = {
            'filter': f'publication_date:{year}-{month_str},type:article,topics.id:{topic_id}',
            'per-page': 1
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('meta', {}).get('count', 0)
        except requests.exceptions.RequestException as e:
            print(f"Error getting count for {year}-{month_str}: {e}")
            return 0
    
    def _fetch_month(self, topic_id: str, year: int, month: int) -> List[Dict]:
        """Fetch all papers for a single month"""
        papers = []
        page = 1
        per_page = 200
        month_str = f"{month:02d}"
        
        url = f"{self.base_url}/works"
        
        while True:
            params = {
                'filter': f'publication_date:{year}-{month_str},type:article,topics.id:{topic_id}',
                'per-page': per_page,
                'page': page,
                'select': 'id,title,publication_year,publication_date,cited_by_count,concepts,authorships,referenced_works'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                batch = data.get('results', [])
                
                if not batch:
                    break
                
                papers.extend(batch)
                
                if len(batch) < per_page:
                    break
                    
                page += 1
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page} for {year}-{month_str}: {e}")
                break
        
        return papers
    
    def _save_papers_by_month(self, papers_by_month: Dict[str, List[Dict]]) -> Dict[str, int]:
        saved_counts = {}
        
        for year_month, papers in papers_by_month.items():
            if not papers:
                continue
                
            year = year_month.split('-')[0]
            year_dir = f"src/raw_data/openalex/csv/{year}"
            os.makedirs(year_dir, exist_ok=True)
            filepath = os.path.join(year_dir, f"{year_month}.csv")
            
            # Flatten papers
            flattened_papers = []
            for paper in papers:
                flat_paper = {
                    'id': paper.get('id'),
                    'doi': paper.get('doi'),
                    'title': paper.get('title'),
                    'publication_year': paper.get('publication_year'),
                    'publication_date': paper.get('publication_date'),
                    'type': paper.get('type'),
                    'cited_by_count': paper.get('cited_by_count'),
                    'abstract_inverted_index': str(paper.get('abstract_inverted_index', '')),
                    'num_authors': len(paper.get('authorships', [])),
                    'author_names': '; '.join([
                        auth.get('author', {}).get('display_name', '') 
                        for auth in paper.get('authorships', [])
                    ]),
                    'num_concepts': len(paper.get('concepts', [])),
                    'concepts': '; '.join([
                        concept.get('display_name', '') 
                        for concept in paper.get('concepts', [])
                    ]),
                    'num_topics': len(paper.get('topics', [])), 
                    'topics': '; '.join([
                        topic.get('display_name', '') 
                        for topic in paper.get('topics', [])
                    ]),  # ADD THIS
                    'num_references': len(paper.get('referenced_works', [])),
                    'referenced_works': '; '.join(paper.get('referenced_works', [])) if paper.get('referenced_works') else ''
                }
                flattened_papers.append(flat_paper)

            df = pd.DataFrame(flattened_papers)
            
            # Batch fetch ALL DOIs for this month at once
            #print(f"  Fetching DOIs for {len(df)} papers...")
            #df['referenced_dois'] = self.batch_fetch_dois_for_month(df['referenced_works'])

            df.to_csv(filepath, index=False, encoding='utf-8')
            saved_counts[filepath] = len(df)
            print(f"✅ Saved {len(df)} papers to {filepath}")

        return saved_counts

    """def batch_fetch_dois_for_month(self, referenced_works_series: pd.Series) -> pd.Series:
        # Fetch DOIs for all papers in a month efficiently.
        # Collect all unique work IDs
        all_work_ids = set()
        for ref_str in referenced_works_series:
            if pd.isna(ref_str) or ref_str is None or not str(ref_str).strip():
                continue
            works_clean = str(ref_str).replace('[','').replace(']','').replace("'",'').replace('"','')
            work_ids = [w.strip().split('/')[-1] for w in works_clean.replace(';',',').split(',') if w.strip()]
            all_work_ids.update(work_ids)
        
        print(f"  Found {len(all_work_ids)} unique referenced works to fetch DOIs for...")
        
        # Fetch DOIs in batches
        doi_map = self.fetch_dois_batch(list(all_work_ids))
        
        # Map back to each paper
        result = []
        for ref_str in referenced_works_series:
            if pd.isna(ref_str) or ref_str is None or not str(ref_str).strip():
                result.append('')
                continue
            
            works_clean = str(ref_str).replace('[','').replace(']','').replace("'",'').replace('"','')
            work_ids = [w.strip().split('/')[-1] for w in works_clean.replace(';',',').split(',') if w.strip()]
            dois = [doi_map.get(wid, '') for wid in work_ids]
            result.append('; '.join(filter(None, dois)))
        
        return pd.Series(result)

    def fetch_dois_batch(self, work_ids: List[str]) -> Dict[str, str]:
        # Fetch DOIs for a list of work IDs with rate limit handling.
        doi_map = {}
        batch_size = 50
        
        for i in range(0, len(work_ids), batch_size):
            batch = work_ids[i:i+batch_size]
            
            for attempt in range(3):
                try:
                    filter_string = '|'.join(batch)
                    url = f"{self.base_url}/works"
                    params = {
                        'filter': f'openalex_id:{filter_string}',
                        'select': 'id,doi',
                        'per-page': 200
                    }
                    response = self.session.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    for work in data.get('results', []):
                        work_id = work.get('id', '').split('/')[-1]
                        doi = work.get('doi', '')
                        # Only process DOI if it's not None or empty
                        if doi:
                            doi = doi.replace('https://doi.org/', '')
                        doi_map[work_id] = doi
                    break
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        wait_time = (2 ** attempt) * 2
                        print(f"    Rate limit. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        break
                except Exception as e:
                    print(f"    Error: {e}")
                    break
            
            time.sleep(0.5)  # Longer delay between batches
        
        return doi_map
    
    def extract_dois_from_references(self, referenced_works: str) -> str:
        # Extract DOIs from OpenAlex referenced works.
        import time
        
        if pd.isna(referenced_works) or not referenced_works.strip():
            return ''
        
        # Clean string and extract work IDs
        works_clean = referenced_works.replace('[','').replace(']','').replace("'",'').replace('"','')
        work_ids = [w.strip().split('/')[-1] for w in works_clean.replace(';',',').split(',') if w.strip()]
        
        if not work_ids:
            return ''
        
        dois = []
        batch_size = 50  # OpenAlex batch size
        
        for i in range(0, len(work_ids), batch_size):
            batch = work_ids[i:i+batch_size]
            try:
                # Query OpenAlex for this batch
                filter_string = '|'.join(batch)
                url = "https://api.openalex.org/works"
                params = {
                    'filter': f'openalex_id:{filter_string}',
                    'select': 'id,doi',
                    'per-page': 200
                }
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Map OpenAlex ID -> DOI
                doi_map = {}
                for work in data.get('results', []):
                    work_id = work.get('id', '').split('/')[-1]
                    doi = work.get('doi', '')
                    if doi:
                        doi = doi.replace('https://doi.org/', '')
                    doi_map[work_id] = doi
                
                # Keep order same as batch
                batch_dois = [doi_map.get(wid, '') for wid in batch]
                dois.extend(batch_dois)
                
            except Exception as e:
                print(f"    Warning: Error fetching DOI batch: {e}")
                # Fill with empty strings for this batch to preserve order
                dois.extend([''] * len(batch))
            
            time.sleep(0.1)  # Polite delay
        
        # Return semicolon-separated DOIs
        return '; '.join(filter(None, dois))"""

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your email
    analyzer = AIScholarshipAnalyzer("ninelloldenburg@gmail.com") 
    analyzer.get_top_concepts_by_topic()
    #analyzer.get_and_save_articles()