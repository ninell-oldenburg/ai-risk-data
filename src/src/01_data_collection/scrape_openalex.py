import os
import time
from datetime import datetime
from typing import List, Dict
import requests
import pandas as pd
import glob
import matplotlib.pyplot as plt
from collections import Counter
import json

# At the top, update TOPIC_IDS:

TOPIC_IDS = {
    # Core topics
    "Ethics & Social Impacts of AI": "T10883",
    "Adversarial Robustness in ML": "T11689",
    "Explainable AI": "T12026",
    "Hate Speech and Cyberbullying Detection": "T12262",
}

# Boolean search terms
AI_TERMS = [
    'artificial intelligence', 'machine learning', 'deep learning', 'reward function',
    'neural network', 'reinforcement learning', 'language model', 'language models',
    'ai', 'ml', 'llm', 'nlp', 'agi', 'artificial general intelligence',
]

SAFETY_TERMS = [
    'safety', 'alignment', 'fairness', 'bias', 'cooperative', 'red teaming',
    'interpretability', 'explainability', 'robustness', 'human feedback', 'risks',
    'adversarial', 'ethics', 'governance', 'risk', 'human preference', 'malicious use',
    'trustworthy', 'responsible', 'cooperation', 'circuits', 'policy', 'governance',
    'dangers', 'amplification',  'capabilities',
]

class AIScholarshipAnalyzer:
    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.headers = {'User-Agent': f'mailto:{email}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_and_save_articles(self, 
                                start_date: datetime = datetime(2000, 1, 1), 
                                end_date: datetime = datetime(2025, 6, 30)) -> Dict[str, int]:
        """
        Hybrid approach: Collect papers via topics (split by individual topic IDs) + targeted keywords.
        Deduplicates automatically.
        """
        print(f"\n{'='*70}")
        print(f"HYBRID COLLECTION: TOPICS + BOOLEAN KEYWORDS SEARCH")
        print(f"{'='*70}")
        print(f"Topics: {list(TOPIC_IDS.keys())}")
        print(f"AI Keywords: {AI_TERMS}")
        print(f"ETHICS / SAFETY Keywords: {SAFETY_TERMS}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"{'='*70}\n")

        # STEP 1: Collect via topics (individually)
        print("\n" + "="*70)
        print("STEP 1: COLLECTING VIA TOPICS (split by topic ID)")
        print("="*70)

        all_topic_papers = []
        for topic_name, topic_id in TOPIC_IDS.items():
            print(f"\n▶ Collecting for topic: {topic_name} ({topic_id})")
            topic_papers = self._collect_all_papers_by_topics(topic_id, start_date, end_date)
            print(f"✓ {topic_name}: collected {len(topic_papers):,} papers")
            all_topic_papers.extend(topic_papers)

        print(f"\n✓ Total (raw) collected via topics: {len(all_topic_papers):,}")

        # STEP 2: Collect via targeted keywords
        print("\n" + "="*70)
        print("STEP 2: COLLECTING VIA TARGETED KEYWORDS")
        print("="*70)

        keyword_papers = self._collect_papers_by_keywords(AI_TERMS, SAFETY_TERMS, start_date, end_date)
        print(f"\n✓ Collected {len(keyword_papers):,} papers via keywords")

        # STEP 3: Merge and deduplicate
        print("\n" + "="*70)
        print("STEP 3: MERGING AND DEDUPLICATING")
        print("="*70)

        all_papers = self._merge_and_deduplicate(all_topic_papers, keyword_papers)

        print(f"\n✓ Total unique papers: {len(all_papers):,}")
        print(f"  - From topics only: {len(all_topic_papers):,}")
        print(f"  - From keywords only: {len(keyword_papers):,}")
        print(f"  - Overlap: {len(all_topic_papers) + len(keyword_papers) - len(all_papers):,}")

        # STEP 4: Save to files
        print("\n" + "="*70)
        print("STEP 4: SAVING TO FILES")
        print("="*70)

        saved_counts = self._save_papers_by_month_from_list(all_papers)

        print(f"\n✅ COMPLETE: Saved {len(all_papers):,} papers across {len(saved_counts)} files")

        # Save metadata
        self._save_collection_metadata_hybrid(
            len(all_papers),
            len(all_topic_papers),
            len(keyword_papers),
            saved_counts,
            start_date,
            end_date,
        )

        return saved_counts

    def _collect_all_papers_by_topics(self, topic_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect all papers for a single topic ID (split by year)."""
        all_papers = []

        for year in range(start_date.year, end_date.year + 1):
            print(f"\n  Year {year} for topic {topic_id}...")
            count = self._get_paper_count_multi_topic(topic_id, year)
            print(f"    Found {count:,} papers")
            if count == 0:
                continue

            # Fetch monthly batches
            papers_by_month = self._fetch_all_papers_for_year(topic_id, year)
            for month_papers in papers_by_month.values():
                all_papers.extend(month_papers)

            print(f"    Collected {sum(len(v) for v in papers_by_month.values()):,} papers")

        return all_papers

    def _collect_papers_by_keywords(self, 
                                    keywords1: List[str],
                                    keywords2: List[str], 
                                    start_date: datetime, 
                                    end_date: datetime) -> List[Dict]:
        """
        Collect papers matching (ANY keyword in keywords1) AND (ANY keyword in keywords2).
        It searches OpenAlex using keywords1 and then locally filters the results using keywords2.
        """
        print("\n  Searching OpenAlex with combined keyword filters (A AND B)...")
        print(f"  A (Searched via OpenAlex): {keywords1}")
        print(f"  B (Filtered locally): {keywords2}")
        
        all_papers_raw = []
        seen_ids = set()
        
        # --- Step 1: Search OpenAlex using each keyword in the primary set (keywords1) ---
        for keyword in keywords1:
            print(f"\n  Searching OpenAlex for papers matching: '{keyword}'...")
            
            url = f"{self.base_url}/works"
            page = 1
            
            while True:
                params = {
                    'filter': f'publication_year:{start_date.year}-{end_date.year},type:article',
                    'search': keyword,  # Uses the search endpoint for best relevance
                    'per-page': 200,
                    'page': page,
                    'select': 'id,doi,title,publication_year,publication_date,type,cited_by_count,concepts,authorships,topics,referenced_works,abstract_inverted_index,keywords'
                }
                
                try:
                    response = self.session.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.get('results', [])
                    
                    if not data:
                        break
                    
                    # Add unique papers from this batch
                    for paper in data:
                        paper_id = paper.get('id')
                        if paper_id not in seen_ids:
                            # Perform the original local check for the *searched* keyword (optional but good practice)
                            if self._paper_matches_keywords(paper, [keyword]):
                                all_papers_raw.append(paper)
                                seen_ids.add(paper_id)
                    
                    # Stop conditions
                    if page >= 5 or len(data) < 200:
                        break
                    
                    page += 1
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"    Error during OpenAlex search for '{keyword}': {e}")
                    break
            
            print(f"    Found {len([p for p in all_papers_raw if p.get('id') in seen_ids])} unique papers after searching '{keyword}'")
            time.sleep(0.5)
        
        # --- Step 2: Apply the local "AND" filter using keywords2 ---
        final_papers = []
        print(f"\n  Applying local filter: MUST also match ANY keyword in {keywords2}...")
        
        for i, paper in enumerate(all_papers_raw):
            # This checks if the paper matches ANY keyword in the second list
            if self._paper_matches_keywords(paper, keywords2):
                final_papers.append(paper)
        
        print(f"\n  Total papers matching (A AND B): {len(final_papers):,}")
        return final_papers


    def _paper_matches_keywords(self, paper: Dict, keywords: List[str]) -> bool:
        """Check if paper actually contains any of the keywords."""
        # Get searchable text
        title = paper.get('title', '').lower()
        
        # Reconstruct abstract
        abstract_inv = paper.get('abstract_inverted_index', {})
        abstract = ''
        if abstract_inv:
            words = [(pos, word.lower()) for word, positions in abstract_inv.items() 
                    for pos in positions]
            words.sort()
            abstract = ' '.join([w[1] for w in words])
        
        # Get concepts
        concepts = ' '.join([c['display_name'].lower() for c in paper.get('concepts', [])])
        
        text = f"{title} {abstract} {concepts}"
        
        # Check if ANY keyword matches
        return any(kw.lower() in text for kw in keywords)


    def _fetch_all_papers_for_year(self, topic_filter: str, year: int) -> List[Dict]:
        """Fetch all papers for a year, handling pagination."""
        papers_by_month = {}
        page = 1
        per_page = 200
        
        url = f"{self.base_url}/works"
        
        while True:
            params = {
                'filter': f'publication_year:{year},type:article,topics.id:{topic_filter}',
                'per-page': per_page,
                'page': page,
                'select': 'id,doi,title,publication_year,publication_date,type,cited_by_count,concepts,authorships,topics,referenced_works,abstract_inverted_index,keywords'
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
                    year_month = pub_date[:7] if pub_date and len(pub_date) >= 7 else f"{year}-01"
                    
                    if year_month not in papers_by_month:
                        papers_by_month[year_month] = []
                    papers_by_month[year_month].append(paper)
                
                print(f"  Page {page}: {len(batch)} papers")
                
                if len(batch) < per_page:
                    break
                    
                page += 1
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break
        
        return papers_by_month

    def _merge_and_deduplicate(self, topic_papers: List[Dict], 
                            keyword_papers: List[Dict]) -> List[Dict]:
        """Merge two lists and remove duplicates by OpenAlex ID."""
        seen_ids = set()
        merged = []
        
        # Add all topic papers first
        for paper in topic_papers:
            paper_id = paper.get('id')
            if paper_id not in seen_ids:
                paper['source'] = 'topic'  # Tag for tracking
                merged.append(paper)
                seen_ids.add(paper_id)
        
        # Add keyword papers that aren't duplicates
        for paper in keyword_papers:
            paper_id = paper.get('id')
            if paper_id not in seen_ids:
                paper['source'] = 'keyword'  # Tag for tracking
                merged.append(paper)
                seen_ids.add(paper_id)
        
        return merged


    def _save_papers_by_month_from_list(self, papers: List[Dict]) -> Dict[str, int]:
        """Save a list of papers, grouped by month."""
        # Group papers by month
        papers_by_month = {}
        
        for paper in papers:
            pub_date = paper.get('publication_date')
            year_month = pub_date[:7] if pub_date and len(pub_date) >= 7 else '2015-01'
            
            if year_month not in papers_by_month:
                papers_by_month[year_month] = []
            papers_by_month[year_month].append(paper)
        
        # Save each month
        saved_counts = {}
        
        for year_month, month_papers in papers_by_month.items():
            year = year_month.split('-')[0]
            year_dir = f"src/raw_data/openalex/csv/{year}"
            os.makedirs(year_dir, exist_ok=True)
            filepath = os.path.join(year_dir, f"{year_month}.csv")
            
            # Flatten papers
            flattened_papers = []
            for paper in month_papers:
                flat_paper = {
                    'id': paper.get('id'),
                    'doi': paper.get('doi'),
                    'title': paper.get('title'),
                    'publication_year': paper.get('publication_year'),
                    'publication_date': paper.get('publication_date'),
                    'type': paper.get('type'),
                    'cited_by_count': paper.get('cited_by_count'),
                    'source': paper.get('source', 'topic'),  # Track if from topic or keyword
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
                    'num_keywords': len(paper.get('keywords', [])) if paper.get('keywords') else 0,
                    'keywords': '; '.join([
                        kw.get('display_name', '') 
                        for kw in paper.get('keywords', [])
                    ]) if paper.get('keywords') else '',
                    'num_references': len(paper.get('referenced_works', [])),
                    'referenced_works': '; '.join(paper.get('referenced_works', [])) if paper.get('referenced_works') else ''
                }
                flattened_papers.append(flat_paper)
            
            df = pd.DataFrame(flattened_papers)
            df.to_csv(filepath, index=False, encoding='utf-8')
            saved_counts[filepath] = len(df)
            print(f"  ✓ Saved {len(df):,} papers to {filepath}")
        
        return saved_counts


    def _save_collection_metadata(self, total_papers: int, topic_papers: int,
                                        keyword_papers: int, saved_counts: Dict[str, int],
                                        start_date: datetime, end_date: datetime):
        """Save metadata about the hybrid collection."""
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'methodology': 'Hybrid: Topics OR Targeted Keywords',
            'topics_used': TOPIC_IDS,
            'ai_keywords': AI_TERMS,
            'safety_keywords': SAFETY_TERMS,
            'validation_coverage': '~96%',
            'statistics': {
                'total_papers': total_papers,
                'papers_from_topics': topic_papers,
                'papers_from_keywords': keyword_papers,
                'overlap': topic_papers + keyword_papers - total_papers,
                'keyword_contribution': keyword_papers - (topic_papers + keyword_papers - total_papers)
            },
            'total_files': len(saved_counts),
        }
        
        metadata_dir = "src/metadata"
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, "openalex_collection_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📋 Metadata saved to {metadata_path}")

    def _get_paper_count_multi_topic(self, topic_filter: str, year: int) -> int:
        """Get count of papers for multiple topics (OR logic)"""
        url = f"{self.base_url}/works"
        params = {
            'filter': f'publication_year:{year},type:article,topics.id:{topic_filter}',
            'per-page': 1
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('meta', {}).get('count', 0)
        except Exception as e:
            print(f"Error getting count for {year}: {e}")
            return 0
        
def main():
    """Run full validation comparison."""
    print("""
    ═══════════════════════════════════════════════════════════════
    SCRAPE OPENALEX PAPERS
    ═══════════════════════════════════════════════════════════════
    
    Please provide your email for OpenAlex API access:
    """)
    
    email = input("Email: ").strip()
    if not email:
        print("Email required for OpenAlex API. Exiting.")
        return
    
    analyzer = AIScholarshipAnalyzer(email)
    analyzer.get_and_save_articles()
    
if __name__ == "__main__":
    main()
    