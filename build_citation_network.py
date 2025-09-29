from typing import List, Optional, Dict
from collections import defaultdict, Counter
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns


class CitationGraph:
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