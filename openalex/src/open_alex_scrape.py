import requests
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import time
import json
from typing import List, Dict, Set, Tuple
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class AIScholarshipAnalyzer:
    def __init__(self, email: str):
        """
        Initialize the analyzer with OpenAlex API access
        
        Args:
            email: Your email for OpenAlex API (polite pool access)
        """
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.headers = {'User-Agent': f'mailto:{email}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def search_papers(self, topic_id: str = "T10883", years: str = "2016-2025", limit: int = 10000) -> List[Dict]:
        papers = []
        page = 1
        per_page = 100
        
        while True:
            url = f"{self.base_url}/works"
            params = {
                'filter': f'publication_year:{years},type:article,topics.id:{topic_id}',
                'per-page': per_page,
                'page': page,
                'select': 'id,title,publication_year,cited_by_count,concepts,authorships,referenced_works'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            batch = data.get('results', [])
            
            if not batch:
                print("No more results available")
                break
                
            papers.extend(batch)
            print(f"Retrieved {len(papers)} papers...")
            
            if limit and len(papers) >= limit:
                papers = papers[:limit]
                break
                
            if len(batch) < per_page:
                print("Reached last page")
                break
                
            page += 1
            time.sleep(1)  # Increase delay to be extra polite
            
        return papers
    
    def build_citation_network(self, papers: List[Dict]) -> nx.Graph:
        """
        Build citation network from papers
        
        Args:
            papers: List of paper dictionaries from OpenAlex
        """
        G = nx.Graph()
        paper_ids = set(paper['id'] for paper in papers)
        
        # Add nodes
        for paper in papers:
            G.add_node(paper['id'], 
                      title=paper.get('title', ''),
                      year=paper.get('publication_year'),
                      citations=paper.get('cited_by_count', 0))
        
        # Add edges based on citations
        for paper in papers:
            paper_id = paper['id']
            referenced_works = paper.get('referenced_works', [])
            
            for ref_id in referenced_works:
                if ref_id in paper_ids:
                    G.add_edge(paper_id, ref_id)
        
        return G
    
    def detect_communities(self, G: nx.Graph) -> Dict[str, int]:
        """
        Detect communities in the citation network using Louvain algorithm
        """
        if community_louvain is not None:
            # Use python-louvain if available
            partition = community_louvain.best_partition(G)
        else:
            # Fall back to NetworkX's built-in community detection
            print("Using NetworkX community detection (install python-louvain for better results)")
            communities = nx_community.louvain_communities(G)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
        
        # Print community statistics
        community_sizes = Counter(partition.values())
        print(f"Found {len(community_sizes)} communities:")
        for comm_id, size in community_sizes.most_common():
            print(f"  Community {comm_id}: {size} papers")
            
        return partition
    
    def analyze_communities(self, papers: List[Dict], partition: Dict[str, int]) -> pd.DataFrame:
        """
        Analyze the detected communities to understand their characteristics
        """
        paper_dict = {paper['id']: paper for paper in papers}
        
        community_data = []
        for paper_id, comm_id in partition.items():
            paper = paper_dict.get(paper_id, {})
            
            # Extract concepts/keywords
            concepts = paper.get('concepts', [])
            concept_names = [c.get('display_name', '') for c in concepts[:5]]  # Top 5 concepts
            
            community_data.append({
                'paper_id': paper_id,
                'community': comm_id,
                'title': paper.get('title', ''),
                'year': paper.get('publication_year'),
                'citations': paper.get('cited_by_count', 0),
                'concepts': ', '.join(concept_names)
            })
        
        df = pd.DataFrame(community_data)
        return df
    
    def classify_communities(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Classify communities as x-risk vs critical AI based on common concepts/keywords
        """
        xrisk_keywords = ['existential risk', 'superintelligence', 'ai safety', 'alignment', 
                         'artificial general intelligence', 'control problem', 'ai risk', 
                         'agi', 'ai security']
        critical_keywords = ['algorithmic bias', 'fairness', 'ethics', 'accountability',
                           'discrimination', 'social justice', 'governance', 'transparency', 
                           'dystopia', 'ai hype']
        
        community_labels = {}
        
        for comm_id in df['community'].unique():
            comm_papers = df[df['community'] == comm_id]
            
            # Combine all concepts for this community
            all_concepts = ' '.join(comm_papers['concepts'].fillna('')).lower()
            all_titles = ' '.join(comm_papers['title'].fillna('')).lower()
            combined_text = all_concepts + ' ' + all_titles
            
            # Count keyword matches
            xrisk_score = sum(1 for kw in xrisk_keywords if kw in combined_text)
            critical_score = sum(1 for kw in critical_keywords if kw in combined_text)
            
            # Classify based on scores
            if xrisk_score > critical_score:
                label = 'X-Risk'
            elif critical_score > xrisk_score:
                label = 'Critical AI'
            else:
                label = 'Mixed/Other'
                
            community_labels[comm_id] = label
            
            print(f"Community {comm_id}: {label} (X-risk: {xrisk_score}, Critical: {critical_score})")
            print(f"  Size: {len(comm_papers)} papers")
            print(f"  Sample titles: {list(comm_papers['title'].head(3))}")
            print()
            
        return community_labels
    
    def extract_author_gender_info(self, papers: List[Dict]) -> pd.DataFrame:
        """
        Extract author information for gender analysis
        Note: This extracts available author data - you'll need additional tools for gender inference
        """
        author_data = []
        
        for paper in papers:
            authorships = paper.get('authorships', [])
            
            for authorship in authorships:
                author = authorship.get('author', {})
                
                author_data.append({
                    'paper_id': paper['id'],
                    'author_id': author.get('id'),
                    'author_name': author.get('display_name'),
                    'position': authorship.get('author_position'),
                    'is_corresponding': authorship.get('is_corresponding', False),
                    'paper_title': paper.get('title', ''),
                    'year': paper.get('publication_year')
                })
        
        return pd.DataFrame(author_data)
    
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Dict[int, str], pd.DataFrame]:
        """
        Run the complete analysis pipeline
        """
        print("=== AI Scholarship Citation Network Analysis ===\n")
        
        # Step 1: Collect papers
        print("1. Collecting papers...")
        papers = self.search_papers()
        print(f"\nTotal papers collected: {len(papers)}")
        
        # Step 2: Build citation network
        print("\n3. Building citation network...")
        G = self.build_citation_network(papers)
        print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Step 3: Detect communities
        print("\n4. Detecting communities...")
        partition = self.detect_communities(G)
        
        # Step 4: Analyze communities
        print("\n5. Analyzing communities...")
        community_df = self.analyze_communities(papers, partition)
        
        # Step 5: Classify communities
        print("\n6. Classifying communities...")
        community_labels = self.classify_communities(community_df)
        
        # Step 6: Extract author info
        print("\n7. Extracting author information...")
        author_df = self.extract_author_gender_info(papers)
        
        return community_df, community_labels, author_df

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your email
    analyzer = AIScholarshipAnalyzer("ninelloldenburg@gmail.com") 
    
    # Run analysis
    community_df, community_labels, author_df = analyzer.run_full_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y_%m_%d")
    
    community_df.to_csv(f'openalex/data/ai_communities_{timestamp}.csv', index=False)
    author_df.to_csv(f'openalex/data/ai_authors_{timestamp}.csv', index=False)
    
    # Convert numpy int64 keys to regular int for JSON serialization
    community_labels_serializable = {int(k): v for k, v in community_labels.items()}
    
    with open(f'community_labels_{timestamp}.json', 'w') as f:
        json.dump(community_labels_serializable, f, indent=2)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved with timestamp: {timestamp}")
    print(f"Community data: ai_communities_{timestamp}.csv")
    print(f"Author data: ai_authors_{timestamp}.csv")
    print(f"Community labels: community_labels_{timestamp}.json")
    
    # Quick summary
    print(f"\nQuick Summary:")
    print(f"Total papers analyzed: {len(community_df)}")
    print(f"Communities found: {len(set(community_df['community']))}")
    print(f"Total authors: {len(author_df)}")
    
    for label in set(community_labels.values()):
        count = sum(1 for l in community_labels.values() if l == label)
        papers_count = len(community_df[community_df['community'].isin(
            [k for k, v in community_labels.items() if v == label])])
        print(f"{label} communities: {count} (containing {papers_count} papers)")