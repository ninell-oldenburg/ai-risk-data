"""
Process LessWrong/AlignmentForum and OpenAlex data into node and edge tables
for network analysis and publication.

Expected input structure:
- src/graphql/data/{lw,af}/csv_cleanedwithtopic/{year}/{year}-{month}.csv
- src/openalex/data/csv/{year}/{year}-{month}.csv (or similar)

Output structure in data/:
- nodes_posts.csv
- nodes_authors.csv
- nodes_openalex_works.csv
- nodes_openalex_authors.csv (optional)
- edges_post_citations.csv
- edges_post_openalex.csv
- edges_openalex_authorship.csv (optional)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from urllib.parse import urlparse
from collections import defaultdict
import json

class ForumGraphBuilder:
    def __init__(self):
        self.forum_data_dir = Path('src/processed_data/')
        self.openalex_data_dir = Path('src/processed_data/openalex/02_with_gender')
        self.output_dir = Path('data')
        self.output_dir.mkdir(exist_ok=True)
        
        # URL patterns for identifying forum posts
        self.lw_patterns = [
            r'lesswrong\.com/posts/([^/]+)',
            r'lesserwrong\.com/posts/([^/]+)',
            r'lesswrong\.com/lw/([^/]+)',
        ]
        self.af_patterns = [
            r'alignmentforum\.org/posts/([^/]+)',
        ]
        
    def load_forum_data(self):
        """Load all forum CSVs from both LW and AF"""
        all_posts = []
        
        for forum in ['lesswrong', 'alignment_forum']:
            forum_path = self.forum_data_dir / forum / '03_with_topics'
            
            if not forum_path.exists():
                print(f"Warning: {forum_path} does not exist, skipping {forum}")
                continue
                
            # Iterate through year directories
            for year_dir in sorted(forum_path.glob('*')):
                if not year_dir.is_dir():
                    continue
                    
                # Load all month CSVs in this year
                for csv_file in sorted(year_dir.glob('*.csv')):
                    print(f"Loading {csv_file}")
                    df = pd.read_csv(csv_file)
                    df['source'] = forum  # Add source column
                    all_posts.append(df)
        
        if not all_posts:
            raise ValueError("No forum data found!")
            
        combined = pd.concat(all_posts, ignore_index=True)
        print(f"Loaded {len(combined)} total posts from forums")
        return combined
    
    def load_openalex_data(self):
        """Load all OpenAlex CSVs"""
        all_works = []
        
        if not self.openalex_data_dir.exists():
            print(f"Warning: {self.openalex_data_dir} does not exist")
            return pd.DataFrame()
        
        # Iterate through year directories or flat structure
        csv_files = list(self.openalex_data_dir.rglob('*.csv'))
        
        for csv_file in sorted(csv_files):
            print(f"Loading {csv_file}")
            df = pd.read_csv(csv_file)
            all_works.append(df)
        
        if not all_works:
            print("Warning: No OpenAlex data found")
            return pd.DataFrame()
            
        combined = pd.concat(all_works, ignore_index=True)
        print(f"Loaded {len(combined)} OpenAlex works")
        return combined
    
    def standardize_forum_columns(self, df):
        """Rename columns to match our schema"""
        column_mapping = {
            '_id': 'post_id',
            'pageUrl': 'page_url',
            'postedAt': 'posted_at',
            'baseScore': 'base_score',
            'voteCount': 'vote_count',
            'commentCount': 'comment_count',
            'question': 'is_question',
            'htmlBody': 'html_body',
            'user.username': 'author_username',
            'user.slug': 'author_slug',
            'user.displayName': 'author_display_name',
            'cleaned_htmlBody': 'text',
            'user_gender': 'author_gender_inferred',
            'topic_cluster_id': 'topic_id',
            'topic_label': 'topic_label',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add missing columns with defaults
        if 'doi' not in df.columns:
            df['doi'] = None
        if 'openalex_id' not in df.columns:
            df['openalex_id'] = None
        if 'is_crosspost' not in df.columns:
            df['is_crosspost'] = False
            
        return df
    
    def create_nodes_posts(self, df):
        """Create the posts node table"""
        columns = [
            'post_id', 'source', 'title', 'posted_at', 'text', 
            'topic_id', 'topic_label', 'base_score', 'vote_count', 
            'comment_count', 'is_question', 'is_linkpost', 'doi', 
            'openalex_id', 'author_username', 'page_url', 'slug'
        ]
        
        # Select only columns that exist
        available_cols = [col for col in columns if col in df.columns]
        posts_df = df[available_cols].copy()
        
        # Detect crossposts (same title appearing in both lw and af)
        if 'title' in posts_df.columns:
            title_sources = posts_df.groupby('title')['source'].apply(lambda x: set(x))
            crosspost_titles = title_sources[title_sources.apply(len) > 1].index
            posts_df['is_crosspost'] = posts_df['title'].isin(crosspost_titles)
        
        return posts_df
    
    def create_nodes_authors(self, df):
        """Create the authors node table by aggregating posts"""
        author_stats = df.groupby('author_username').agg({
            'post_id': lambda x: list(x),  # List of post IDs
            'author_display_name': 'first',
            'author_gender_inferred': 'first',
            'source': lambda x: list(x),  # All sources they've posted to
        }).reset_index()
        
        author_stats.columns = ['author_username', 'post_ids', 'author_display_name', 
                               'author_gender_inferred', 'sources']
        
        # Determine primary source
        def get_primary_source(sources):
            if isinstance(sources, list):
                unique = set(sources)
                if len(unique) > 1:
                    return 'both'
                return list(unique)[0] if unique else None
            return sources
        
        author_stats['primary_source'] = author_stats['sources'].apply(get_primary_source)
        author_stats['post_count'] = author_stats['post_ids'].apply(len)
        
        # Convert post_ids to JSON string for CSV storage
        author_stats['post_ids'] = author_stats['post_ids'].apply(json.dumps)
        
        # Select final columns
        final_cols = ['author_username', 'author_display_name', 'author_gender_inferred',
                     'post_count', 'primary_source', 'post_ids']
        
        return author_stats[final_cols]
    
    def extract_forum_post_id(self, url):
        """Extract post ID from a forum URL"""
        if not isinstance(url, str):
            return None
            
        for pattern in self.lw_patterns + self.af_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def extract_doi(self, url):
        """Extract DOI from a URL"""
        if not isinstance(url, str):
            return None
            
        # DOI patterns
        doi_patterns = [
            r'doi\.org/(.+)',
            r'dx\.doi\.org/(.+)',
            r'doi:\s*(.+)',
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def create_edges_post_citations(self, df):
        """Extract citations between forum posts"""
        edges = []
        
        # Create a lookup of slug/URL to post_id
        post_lookup = {}
        for _, row in df.iterrows():
            if pd.notna(row.get('page_url')):
                post_lookup[row['page_url']] = row['post_id']
            if pd.notna(row.get('slug')):
                post_lookup[row['slug']] = row['post_id']
        
        for _, row in df.iterrows():
            citing_id = row['post_id']
            links = row.get('extracted_links')
            
            if pd.isna(links) or links == '':
                continue
            
            # Parse links (assuming it's a JSON list or similar)
            try:
                if isinstance(links, str):
                    link_list = json.loads(links)
                else:
                    link_list = links
            except:
                # If not JSON, try splitting by common delimiters
                link_list = str(links).split(',') if ',' in str(links) else [links]
            
            for link in link_list:
                if not isinstance(link, str):
                    continue
                    
                # Check if this links to a forum post
                cited_slug = self.extract_forum_post_id(link)
                
                # Try to find the cited post
                cited_id = None
                if cited_slug and cited_slug in post_lookup:
                    cited_id = post_lookup[cited_slug]
                elif link in post_lookup:
                    cited_id = post_lookup[link]
                
                if cited_id and cited_id != citing_id:  # Avoid self-citations
                    edges.append({
                        'citing_post_id': citing_id,
                        'cited_post_id': cited_id,
                    })
        
        edges_df = pd.DataFrame(edges)
        
        # Remove duplicates
        if not edges_df.empty:
            edges_df = edges_df.drop_duplicates()
        
        print(f"Found {len(edges_df)} post-to-post citations")
        return edges_df
    
    def create_edges_post_openalex(self, forum_df, openalex_df):
        """Extract citations from posts to OpenAlex works"""
        edges = []
        
        # Create DOI lookup for OpenAlex works
        doi_lookup = {}
        if not openalex_df.empty and 'doi' in openalex_df.columns:
            for _, row in openalex_df.iterrows():
                if pd.notna(row.get('doi')):
                    doi = str(row['doi']).lower().strip()
                    doi_lookup[doi] = row.get('id', row.get('openalex_id'))
        
        # Create openalex_id lookup
        id_lookup = {}
        if not openalex_df.empty and 'id' in openalex_df.columns:
            for _, row in openalex_df.iterrows():
                if pd.notna(row.get('id')):
                    id_lookup[str(row['id'])] = row.get('id')
        
        for _, row in forum_df.iterrows():
            citing_id = row['post_id']
            links = row.get('extracted_links')
            
            if pd.isna(links) or links == '':
                continue
            
            # Parse links
            try:
                if isinstance(links, str):
                    link_list = json.loads(links)
                else:
                    link_list = links
            except:
                link_list = str(links).split(',') if ',' in str(links) else [links]
            
            for link in link_list:
                if not isinstance(link, str):
                    continue
                
                # Check for DOI
                doi = self.extract_doi(link)
                if doi:
                    doi_clean = doi.lower().strip()
                    openalex_id = doi_lookup.get(doi_clean)
                    
                    if openalex_id:
                        edges.append({
                            'citing_post_id': citing_id,
                            'openalex_id': openalex_id,
                            'openalex_doi': doi
                        })
        
        edges_df = pd.DataFrame(edges)
        
        # Remove duplicates
        if not edges_df.empty:
            edges_df = edges_df.drop_duplicates()
        
        print(f"Found {len(edges_df)} post-to-OpenAlex citations")
        return edges_df
    
    def create_nodes_openalex_works(self, openalex_df):
        """Create OpenAlex works node table"""
        if openalex_df.empty:
            return pd.DataFrame(columns=['openalex_id', 'openalex_doi', 'title', 'publication_year', 'type', 'cited_by_count'])
        
        # Select relevant columns for works
        works_columns = ['id', 'doi', 'title', 'publication_year', 'type', 'cited_by_count']
        available_cols = [col for col in works_columns if col in openalex_df.columns]
        
        works_df = openalex_df[available_cols].copy()
        
        # Rename to match our schema
        works_df = works_df.rename(columns={
            'id': 'openalex_id',
            'doi': 'openalex_doi'
        })
        
        return works_df.drop_duplicates(subset=['openalex_id'])
    
    def parse_authorships(self, authorships_str):
        """Parse authorships field (likely JSON list)"""
        if pd.isna(authorships_str) or authorships_str == '':
            return []
        
        try:
            if isinstance(authorships_str, str):
                authorships = json.loads(authorships_str)
            else:
                authorships = authorships_str
            
            if not isinstance(authorships, list):
                return []
            
            return authorships
        except:
            return []
    
    def parse_authors_gender(self, authors_gender_str):
        """Parse authors_gender field (likely JSON list)"""
        if pd.isna(authors_gender_str) or authors_gender_str == '':
            return []
        
        try:
            if isinstance(authors_gender_str, str):
                genders = json.loads(authors_gender_str)
            else:
                genders = authors_gender_str
            
            if not isinstance(genders, list):
                return []
            
            return genders
        except:
            return []
    
    def create_nodes_openalex_authors(self, openalex_df):
        """Create OpenAlex authors node table"""
        if openalex_df.empty:
            return pd.DataFrame(columns=['author_id', 'author_name', 'inferred_gender', 'work_count', 'work_ids'])
        
        authors_dict = defaultdict(lambda: {
            'author_name': None,
            'inferred_gender': None,
            'work_ids': [],
            'author_positions': []
        })
        
        for _, row in openalex_df.iterrows():
            work_id = row.get('id')
            authorships = self.parse_authorships(row.get('authorships'))
            genders = self.parse_authors_gender(row.get('authors_gender'))
            
            # Match authorships with genders (assuming they're in the same order)
            for idx, authorship in enumerate(authorships):
                # Extract author info from authorship
                # Authorship structure varies, but typically has 'author' dict
                if isinstance(authorship, dict):
                    author_info = authorship.get('author', {})
                    author_id = author_info.get('id', authorship.get('author_id'))
                    author_name = author_info.get('display_name', authorship.get('author_name'))
                    author_position = authorship.get('author_position', 'unknown')
                    
                    if not author_id:
                        continue
                    
                    # Get gender for this position if available
                    gender = genders[idx] if idx < len(genders) else None
                    
                    # Update author record
                    if authors_dict[author_id]['author_name'] is None:
                        authors_dict[author_id]['author_name'] = author_name
                    
                    # Keep track of gender (use most common if multiple works)
                    if gender:
                        authors_dict[author_id]['inferred_gender'] = gender
                    
                    authors_dict[author_id]['work_ids'].append(work_id)
                    authors_dict[author_id]['author_positions'].append(author_position)
        
        # Convert to DataFrame
        authors_data = []
        for author_id, info in authors_dict.items():
            authors_data.append({
                'author_id': author_id,
                'author_name': info['author_name'],
                'inferred_gender': info['inferred_gender'],
                'work_count': len(info['work_ids']),
                'work_ids': json.dumps(info['work_ids'])  # Store as JSON string
            })
        
        authors_df = pd.DataFrame(authors_data)
        print(f"Extracted {len(authors_df)} unique OpenAlex authors")
        return authors_df
    
    def create_edges_openalex_authorship(self, openalex_df):
        """Create edges between OpenAlex authors and works"""
        if openalex_df.empty:
            return pd.DataFrame(columns=['author_id', 'openalex_id', 'author_position'])
        
        edges = []
        
        for _, row in openalex_df.iterrows():
            work_id = row.get('id')
            authorships = self.parse_authorships(row.get('authorships'))
            
            for authorship in authorships:
                if isinstance(authorship, dict):
                    author_info = authorship.get('author', {})
                    author_id = author_info.get('id', authorship.get('author_id'))
                    author_position = authorship.get('author_position', 'unknown')
                    
                    if author_id and work_id:
                        edges.append({
                            'author_id': author_id,
                            'openalex_id': work_id,
                            'author_position': author_position
                        })
        
        edges_df = pd.DataFrame(edges)
        
        if not edges_df.empty:
            edges_df = edges_df.drop_duplicates()
        
        print(f"Found {len(edges_df)} author-work relationships")
        return edges_df
    
    def run_pipeline(self):
        """Execute the full data processing pipeline"""
        print("=" * 60)
        print("Starting Forum Graph Builder Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n[1/7] Loading forum data...")
        # forum_df = self.load_forum_data()
        
        print("\n[2/7] Loading OpenAlex data...")
        openalex_df = self.load_openalex_data()
        
        # Step 2: Standardize columns
        print("\n[3/7] Standardizing column names...")
        #forum_df = self.standardize_forum_columns(forum_df)
        
        # Step 3: Create node tables
        print("\n[4/7] Creating nodes_posts.csv...")
        #nodes_posts = self.create_nodes_posts(forum_df)
        #nodes_posts.to_csv(self.output_dir / 'nodes_posts.csv', index=False)
        #print(f"  → Saved {len(nodes_posts)} posts")
        
        print("\n[5/7] Creating nodes_authors.csv...")
        #nodes_authors = self.create_nodes_authors(forum_df)
        #nodes_authors.to_csv(self.output_dir / 'nodes_authors.csv', index=False)
        #print(f"  → Saved {len(nodes_authors)} authors")
        
        print("\n[6/7] Creating nodes_openalex_works.csv...")
        nodes_openalex = self.create_nodes_openalex_works(openalex_df)
        nodes_openalex.to_csv(self.output_dir / 'nodes_openalex_works.csv', index=False)
        print(f"  → Saved {len(nodes_openalex)} OpenAlex works")
        
        print("\n[6b/7] Creating nodes_openalex_authors.csv...")
        nodes_openalex_authors = self.create_nodes_openalex_authors(openalex_df)
        nodes_openalex_authors.to_csv(self.output_dir / 'nodes_openalex_authors.csv', index=False)
        print(f"  → Saved {len(nodes_openalex_authors)} OpenAlex authors")
        
        # Step 4: Create edge tables
        print("\n[7/7] Creating edge tables...")
        
        print("  - Extracting post-to-post citations...")
        #edges_citations = self.create_edges_post_citations(forum_df)
        #edges_citations.to_csv(self.output_dir / 'edges_post_citations.csv', index=False)
        
        print("  - Extracting post-to-OpenAlex citations...")
        #edges_openalex = self.create_edges_post_openalex(forum_df, openalex_df)
        #edges_openalex.to_csv(self.output_dir / 'edges_post_openalex.csv', index=False)
        
        print("  - Extracting OpenAlex authorship edges...")
        edges_authorship = self.create_edges_openalex_authorship(openalex_df)
        edges_authorship.to_csv(self.output_dir / 'edges_openalex_authorship.csv', index=False)
        
        # Summary
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"\nNode Tables:")
        #print(f"  - nodes_posts.csv: {len(nodes_posts)} rows")
        #print(f"  - nodes_authors.csv: {len(nodes_authors)} rows")
        print(f"  - nodes_openalex_works.csv: {len(nodes_openalex)} rows")
        print(f"  - nodes_openalex_authors.csv: {len(nodes_openalex_authors)} rows")
        print(f"\nEdge Tables:")
        #print(f"  - edges_post_citations.csv: {len(edges_citations)} rows")
        #print(f"  - edges_post_openalex.csv: {len(edges_openalex)} rows")
        print(f"  - edges_openalex_authorship.csv: {len(edges_authorship)} rows")
        print("\nAll files saved to data/")

if __name__ == "__main__":
    builder = ForumGraphBuilder()
    builder.run_pipeline()