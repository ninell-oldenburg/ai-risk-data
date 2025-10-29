import os
import time
from datetime import datetime
from typing import List, Dict, Set
import requests
import pandas as pd
import glob

class AIScholarshipAnalyzer:
    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.headers = {'User-Agent': f'mailto:{email}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Define keyword groups - split into smaller batches
        self.safety_keyword_groups = [
            ['safety', 'alignment', 'ethics', 'ethical'],
            ['fairness', 'bias', 'transparency', 'interpretability', 'explainable'],
            ['risk', 'governance', 'regulation', 'responsible', 'trustworthy'],
            ['cooperative', 'moral', 'transformative', 'control problem', 'existential'],
            ['learned optimization', 'emergent abilities', 'mesa-optimization']
        ]
        
        self.ai_keyword_groups = [
            ['artificial intelligence', 'machine learning', 'language models'],
            ['llms', 'artificial general intelligence', 'superintelligence', 'intelligent machines']
        ]
        
        # Filter these out AFTER fetching
        self.without_topics = [
            'Machine Learning in Bioinformatics', 
            'Artificial Intelligence in Healthcare and Education', 
            'Face Recognition and Perception',
            'Genetics, Bioinformatics, and Biomedical Research',
            'Genetics and Neurodevelopmental Disorders'
        ]
    
    def _build_filter_string_topic_only(self, topic_id: str, year: int = None, month: int = None) -> str:
        """Build filter string for the ethics topic only"""
        filters = []
        
        # Date filters
        if year and month:
            month_str = f"{month:02d}"
            filters.append(f'publication_date:{year}-{month_str}')
        elif year:
            filters.append(f'publication_year:{year}')
        
        # Type filter
        filters.append('type:article')
        
        # Topic filter
        filters.append(f'topics.id:{topic_id}')
        
        return ','.join(filters)
    
    def _build_filter_string_keywords(self, safety_group: List[str], ai_group: List[str], 
                                     year: int = None, month: int = None) -> str:
        """Build filter string with keywords only (no topic requirement)"""
        filters = []
        
        # Date filters
        if year and month:
            month_str = f"{month:02d}"
            filters.append(f'publication_date:{year}-{month_str}')
        elif year:
            filters.append(f'publication_year:{year}')
        
        # Type filter
        filters.append('type:article')
        
        # Keyword filters only
        safety_query = '|'.join(safety_group)
        ai_query = '|'.join(ai_group)
        
        filters.append(f'title_and_abstract.search:({safety_query})')
        filters.append(f'title_and_abstract.search:({ai_query})')
        
        return ','.join(filters)
    
    def _should_exclude_paper(self, paper: Dict) -> bool:
        """Check if paper should be excluded based on topics"""
        paper_topics = [topic.get('display_name', '') for topic in paper.get('topics', [])]
        
        for excluded_topic in self.without_topics:
            if excluded_topic in paper_topics:
                return True
        return False
    
    def _fetch_with_multiple_strategies(self, topic_id: str, year: int = None, month: int = None) -> List[Dict]:
        """Fetch papers using BOTH topic filter AND keyword combinations"""
        all_papers = {}  # Use dict with DOI as key to avoid duplicates
        excluded_count = 0
        
        # STRATEGY 1: Fetch all papers with the ethics topic
        print(f"  Fetching papers with topic {topic_id}...")
        topic_papers = self._fetch_papers_for_filter(
            self._build_filter_string_topic_only(topic_id, year, month)
        )
        
        for paper in topic_papers:
            paper_id = paper.get('doi') or paper.get('id')
            if paper_id and paper_id not in all_papers:
                if not self._should_exclude_paper(paper):
                    all_papers[paper_id] = paper
                else:
                    excluded_count += 1
        
        print(f"    Found {len(all_papers)} papers from topic filter")
        
        # STRATEGY 2: Fetch papers with keyword combinations (may not have the topic)
        total_combinations = len(self.safety_keyword_groups) * len(self.ai_keyword_groups)
        current = 0
        
        for safety_group in self.safety_keyword_groups:
            for ai_group in self.ai_keyword_groups:
                current += 1
                print(f"  Keyword combo {current}/{total_combinations}: {safety_group[:2]}... + {ai_group[:2]}...")
                
                papers = self._fetch_papers_for_filter(
                    self._build_filter_string_keywords(safety_group, ai_group, year, month)
                )
                
                # Deduplicate by DOI or ID and filter out excluded topics
                for paper in papers:
                    paper_id = paper.get('doi') or paper.get('id')
                    if paper_id and paper_id not in all_papers:
                        if not self._should_exclude_paper(paper):
                            all_papers[paper_id] = paper
                        else:
                            excluded_count += 1
                
                time.sleep(0.2)  # Be polite between queries
        
        unique_papers = list(all_papers.values())
        print(f"  Total unique papers: {len(unique_papers)} (excluded {excluded_count} based on topics)")
        return unique_papers
    
    def _fetch_papers_for_filter(self, filter_string: str) -> List[Dict]:
        """Fetch all papers for a given filter string"""
        papers = []
        page = 1
        per_page = 200
        url = f"{self.base_url}/works"
        
        while True:
            params = {
                'filter': filter_string,
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
                
                papers.extend(batch)
                
                if len(batch) < per_page:
                    break
                    
                page += 1
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"    Error on page {page}: {e}")
                break
        
        return papers
        
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
            
            # Fetch papers using both strategies
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
    
    def _fetch_year(self, topic_id: str, year: int) -> Dict[str, int]:
        """Fetch all papers for a single year using both topic and keyword strategies"""
        # Fetch papers with both strategies
        papers = self._fetch_with_multiple_strategies(topic_id, year=year)
        
        # Group by month
        papers_by_month = {}
        for paper in papers:
            pub_date = paper.get('publication_date')
            if pub_date and len(pub_date) >= 7:
                year_month = pub_date[:7]
            else:
                year_month = f"{year}-01"
            
            if year_month not in papers_by_month:
                papers_by_month[year_month] = []
            papers_by_month[year_month].append(paper)
        
        return self._save_papers_by_month(papers_by_month)
    
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
                    ]),
                    'num_references': len(paper.get('referenced_works', [])),
                    'referenced_works': '; '.join(paper.get('referenced_works', [])) if paper.get('referenced_works') else ''
                }
                flattened_papers.append(flat_paper)

            df = pd.DataFrame(flattened_papers)
            df.to_csv(filepath, index=False, encoding='utf-8')
            saved_counts[filepath] = len(df)
            print(f"✅ Saved {len(df)} papers to {filepath}")

        return saved_counts

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your email
    analyzer = AIScholarshipAnalyzer("ninelloldenburg@gmail.com") 
    analyzer.get_and_save_articles()