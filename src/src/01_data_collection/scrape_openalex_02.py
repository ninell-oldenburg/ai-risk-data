import os
import time
from datetime import datetime
from typing import List, Dict
import requests
import pandas as pd

class AIScholarshipAnalyzer:
    def __init__(self, email: str):
        self.base_url = "https://api.openalex.org"
        self.headers = {'User-Agent': f'mailto:{email}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # ---------------------
        # 1. Define keyword groups
        # ---------------------
        self.safety_keyword_groups = [
            ['safety', 'alignment', 'ethics', 'ethical'],
            ['fairness', 'bias', 'transparency', 'interpretability', 'explainable'],
            ['risk', 'governance', 'regulation', 'responsible', 'trustworthy'],
            ['control problem', 'existential', 'long-term', 'transformative'],
            ['ai safety', 'ai governance', 'value alignment', 'robustness']
        ]

        self.ai_keyword_groups = [
            ['artificial intelligence', 'machine learning', 'language model', 'large language model', 'neural network'],
            ['artificial general intelligence', 'deep learning', 'generative ai', 'superintelligence']
        ]

        # ---------------------
        # 2. Topic-based precision filter
        # ---------------------
        self.keep_topics = [
            "Ethics and Societal Impact of AI",
            "AI Safety",
            "Responsible AI",
            "AI Policy",
            "Fairness, Accountability, and Transparency",
            "AI Governance"
        ]

    # ------------------------------------------------------------------
    # Build OpenAlex filter string
    # ------------------------------------------------------------------
    def _build_filter_string(self, safety_group: List[str], ai_group: List[str],
                             year: int = None, month: int = None) -> str:
        filters = ['type:article']
        if year and month:
            filters.append(f'publication_date:{year}-{month:02d}')
        elif year:
            filters.append(f'publication_year:{year}')

        safety_query = '|'.join(safety_group)
        ai_query = '|'.join(ai_group)
        filters.append(f'title_and_abstract.search:({safety_query})')
        filters.append(f'title_and_abstract.search:({ai_query})')
        return ','.join(filters)

    # ------------------------------------------------------------------
    # Fetch papers for one filter
    # ------------------------------------------------------------------
    def _fetch_papers_for_filter(self, filter_string: str) -> List[Dict]:
        url = f"{self.base_url}/works"
        per_page = 200
        papers, page = [], 1
        while True:
            params = {
                'filter': filter_string,
                'per-page': per_page,
                'page': page,
                'select': 'id,doi,title,publication_year,publication_date,type,cited_by_count,concepts,authorships,topics,referenced_works,abstract_inverted_index'
            }
            try:
                r = self.session.get(url, params=params, timeout=40)
                r.raise_for_status()
                data = r.json()
                batch = data.get('results', [])
                if not batch:
                    break
                papers.extend(batch)
                if len(batch) < per_page:
                    break
                page += 1
                time.sleep(0.2)
            except requests.exceptions.RequestException as e:
                print(f"Error on page {page}: {e}")
                break
        return papers

    # ------------------------------------------------------------------
    # 3. High-recall fetch + dedup
    # ------------------------------------------------------------------
    def _fetch_with_keyword_combinations(self, year: int = None, month: int = None) -> List[Dict]:
        all_papers = {}
        total = len(self.safety_keyword_groups) * len(self.ai_keyword_groups)
        counter = 0
        for s_group in self.safety_keyword_groups:
            for a_group in self.ai_keyword_groups:
                counter += 1
                print(f"→ Combination {counter}/{total}: {s_group} + {a_group}")
                fstr = self._build_filter_string(s_group, a_group, year, month)
                papers = self._fetch_papers_for_filter(fstr)
                for p in papers:
                    pid = p.get('doi') or p.get('id')
                    if pid and pid not in all_papers:
                        all_papers[pid] = p
        print(f"✓ Retrieved {len(all_papers)} unique papers for {year}")
        return list(all_papers.values())

    # ------------------------------------------------------------------
    # 4. Stage-2 precision filter
    # ------------------------------------------------------------------
    def _filter_for_relevant_topics(self, papers: List[Dict]) -> List[Dict]:
        """Keep only papers with matching AI-ethics-related topics."""
        filtered = []
        for p in papers:
            paper_topics = [t.get('display_name', '') for t in p.get('topics', [])]
            if any(kt in paper_topics for kt in self.keep_topics):
                filtered.append(p)
        print(f"✓ Filtered to {len(filtered)} relevant papers")
        return filtered

    # ------------------------------------------------------------------
    # 5. Save utilities
    # ------------------------------------------------------------------
    def _save(self, papers: List[Dict], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame([
            {
                'id': p.get('id'),
                'doi': p.get('doi'),
                'title': p.get('title'),
                'publication_year': p.get('publication_year'),
                'publication_date': p.get('publication_date'),
                'type': p.get('type'),
                'cited_by_count': p.get('cited_by_count'),
                'topics': '; '.join([t.get('display_name', '') for t in p.get('topics', [])]),
                'concepts': '; '.join([c.get('display_name', '') for c in p.get('concepts', [])]),
                'num_authors': len(p.get('authorships', []))
            }
            for p in papers
        ])
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"💾 Saved {len(df)} papers → {path}")

    # ------------------------------------------------------------------
    # 6. Main orchestrator
    # ------------------------------------------------------------------
    def get_and_save_articles(self,
                              start_date: datetime = datetime(2000, 1, 1),
                              end_date: datetime = datetime(2025, 6, 30)):
        total = 0
        for year in range(start_date.year, end_date.year + 1):
            print(f"\n=== Processing {year} ===")
            papers = self._fetch_with_keyword_combinations(year=year)
            raw_path = f"src/raw_data/openalex/raw/{year}.csv"
            self._save(papers, raw_path)

            # precision filter
            filtered = self._filter_for_relevant_topics(papers)
            filtered_path = f"src/raw_data/openalex/filtered/{year}.csv"
            self._save(filtered, filtered_path)
            total += len(filtered)
        print(f"\n✓ COMPLETE: {total} filtered papers saved")

# --------------------------
# Example run
# --------------------------
if __name__ == "__main__":
    analyzer = AIScholarshipAnalyzer("ninelloldenburg@gmail.com")
    analyzer.get_and_save_articles()
