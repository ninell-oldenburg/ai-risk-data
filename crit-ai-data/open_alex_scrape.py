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
        self.email = "ninelloldenburg@gmail.com"
        self.headers = {'User-Agent': f'AI-Scholarship-Analyzer (mailto:{email})'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def search_papers(self, query: str, years: str = "2016-2025", limit: int = 1000) -> List[Dict]:
        """
        Search for papers using OpenAlex API
        
        Args:
            query: Search query
            years: Year range (e.g., "2016-2025")
            limit: Maximum number of papers to retrieve
        """
        papers = []
        page = 1
        per_page = 200  # Max allowed by OpenAlex
        
        while len(papers) < limit:
            url = f"{self.base_url}/works"
            params = {
                'search': query,
                'filter': f'publication_year:{years},type:article',
                'per-page': per_page,
                'page': page,
                'select': 'id,title,publication_year,cited_by_count,concepts,authorships,referenced_works'
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break
                
            data = response.json()
            batch = data.get('results', [])
            
            if not batch:
                break
                
            papers.extend(batch)
            print(f"Retrieved {len(papers)} papers...")
            
            if len(batch) < per_page:  # Last page
                break
                
            page += 1
            time.sleep(0.1)  # Be polite to the API
            
        return papers[:limit]
    
    def get_xrisk_papers(self) -> List[Dict]:
        """Get existential risk papers using targeted keywords"""
        xrisk_queries = [
            "existential risk artificial intelligence",
            "ai safety",
            'ai alignment',
            "control problem", 
            "artificial general intelligence",
            "existential risk",
            "agi",
            "superintelligence",
            "ai governance",
            "ai risk",
            "security generative ai",
            "ai security",
            "ai deception",
            "catastrophic ai risks",
            "x-risk",
        ]
        
        all_papers = []
        for query in xrisk_queries:
            print(f"Searching for: {query}")
            papers = self.search_papers(query, limit=200)
            all_papers.extend(papers)
            time.sleep(1)
        
        # Deduplicate by ID
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper['id'] not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper['id'])
                
        print(f"Found {len(unique_papers)} unique x-risk papers")
        return unique_papers
    
    def get_critical_ai_papers(self) -> List[Dict]:
        """Get critical AI papers using targeted keywords"""
        critical_queries = [
            "algorithmic bias",
            "ai fairness",
            "ai ethics",
            "responsible artificial intelligence",
            "critical algorithm studies",
            "ai regulation",
            "artificial intelligence social justice",
            "algorithmic accountability transparency",
            "AI hype",
            "technological solutionism",
            "tescreal",
            "dystopia ai",
            "ethical concerns artificial intelligence",
            "sociocultural artificial intelligence",
            "sociopolitical artificial intelligence",
            "critical theory artificial intelligence"
        ]
        
        all_papers = []
        for query in critical_queries:
            print(f"Searching for: {query}")
            papers = self.search_papers(query, limit=300)
            all_papers.extend(papers)
            time.sleep(1)
        
        # Deduplicate
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper['id'] not in seen_ids:
                unique_papers.append(paper)
                seen_ids.add(paper['id'])
                
        print(f"Found {len(unique_papers)} unique critical AI papers")
        return unique_papers
    
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
        print("1. Collecting X-risk papers...")
        xrisk_papers = self.get_xrisk_papers()
        
        print("\n2. Collecting Critical AI papers...")
        critical_papers = self.get_critical_ai_papers()
        
        all_papers = xrisk_papers + critical_papers
        print(f"\nTotal papers collected: {len(all_papers)}")
        
        # Step 2: Build citation network
        print("\n3. Building citation network...")
        G = self.build_citation_network(all_papers)
        print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Step 3: Detect communities
        print("\n4. Detecting communities...")
        partition = self.detect_communities(G)
        
        # Step 4: Analyze communities
        print("\n5. Analyzing communities...")
        community_df = self.analyze_communities(all_papers, partition)
        
        # Step 5: Classify communities
        print("\n6. Classifying communities...")
        community_labels = self.classify_communities(community_df)
        
        # Step 6: Extract author info
        print("\n7. Extracting author information...")
        author_df = self.extract_author_gender_info(all_papers)
        
        return community_df, community_labels, author_df

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your email
    analyzer = AIScholarshipAnalyzer("your.email@university.edu")  # Replace with your email
    
    # Run analysis
    community_df, community_labels, author_df = analyzer.run_full_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    community_df.to_csv(f'ai_communities_{timestamp}.csv', index=False)
    author_df.to_csv(f'ai_authors_{timestamp}.csv', index=False)
    
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