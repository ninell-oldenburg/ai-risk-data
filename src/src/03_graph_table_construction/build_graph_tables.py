"""
Process LessWrong/AlignmentForum and OpenAlex data into node and edge tables
for network analysis and publication.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from urllib.parse import urlparse
from collections import defaultdict
import json
import ast

class ForumGraphBuilder:
    def __init__(self):
        self.forum_data_dir = Path('src/processed_data/')
        self.openalex_data_dir = Path('src/processed_data/openalex/02_with_gender')
        self.output_dir = Path('data')
        self.output_dir.mkdir(exist_ok=True)
        
        # URL patterns for identifying forum posts
        self.lw_patterns = [
            r'lesswrong\.com/posts/([^/\?#]+)',
            r'lesserwrong\.com/posts/([^/\?#]+)',
            r'lesswrong\.com/lw/([^/\?#]+)',
        ]
        self.af_patterns = [
            r'alignmentforum\.org/posts/([^/\?#]+)',
        ]
        
    def load_forum_data(self):
        """Load all forum CSVs from both LW and AF"""
        all_posts = []
        
        for forum in ['lesswrong', 'alignment_forum']:
            forum_path = self.forum_data_dir / forum / '03_with_topics'
            
            if not forum_path.exists():
                print(f"Warning: {forum_path} does not exist, skipping {forum}")
                continue
            
            # Find all CSV files recursively
            csv_files = list(forum_path.rglob('*.csv'))
            
            for csv_file in sorted(csv_files):
                print(f"Loading {csv_file}")
                df = pd.read_csv(csv_file)
                df['source'] = forum
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
            
        return df
    
    def create_nodes_posts(self, df):
        """Create the posts node table"""
        columns = [
            'post_id', 'source', 'title', 'posted_at', 'text', 
            'topic_id', 'topic_label', 'base_score', 'vote_count', 
            'comment_count', 'is_question', 'doi', 
            'openalex_id', 'author_username', 'page_url', 'slug'
        ]
        
        # Select only columns that exist
        available_cols = [col for col in columns if col in df.columns]
        posts_df = df[available_cols].copy()
        
        # Detect crossposts (same title appearing in both lw and af)
        if 'title' in posts_df.columns and 'source' in posts_df.columns:
            title_sources = posts_df.groupby('title')['source'].apply(lambda x: set(x))
            crosspost_titles = title_sources[title_sources.apply(len) > 1].index
            posts_df['is_crosspost'] = posts_df['title'].isin(crosspost_titles)
        else:
            posts_df['is_crosspost'] = False

        return posts_df
    
    def create_nodes_authors(self, df):
        """Create the authors node table by aggregating posts"""
        if 'author_username' not in df.columns:
            print("Warning: No author_username column found")
            return pd.DataFrame()
        
        df_with_authors = df[df['author_username'].notna()].copy()
        
        if len(df_with_authors) == 0:
            print("Warning: No posts with author information")
            return pd.DataFrame()
                
        author_stats = df_with_authors.groupby('author_username').agg({
            'post_id': lambda x: list(x),
            'author_display_name': 'first',
            'author_gender_inferred': 'first' if 'author_gender_inferred' in df_with_authors.columns else lambda x: None,
            'source': lambda x: list(x),
        }).reset_index()
        
        author_stats.columns = ['author_username', 'post_ids', 'author_display_name', 
                               'author_gender_inferred', 'sources']
        
        # Determine primary source
        def get_primary_source(sources):
            if isinstance(sources, list):
                unique = list(set(sources))
                if len(unique) > 1:
                    return 'alignment_forum'
                return unique[0] if unique else None
            return sources
        
        author_stats['primary_source'] = author_stats['sources'].apply(get_primary_source)
        author_stats['post_count'] = author_stats['post_ids'].apply(len)
        
        # Convert post_ids to JSON string
        author_stats['post_ids'] = author_stats['post_ids'].apply(json.dumps)
        
        # Select final columns
        final_cols = ['author_username', 'author_display_name', 'author_gender_inferred',
                     'post_count', 'primary_source', 'post_ids']
        
        return author_stats[final_cols]
    
    def parse_list_field(self, field):
        """Parse a field that might be a string representation of a list"""
        if pd.isna(field) or field == '':
            return []
        
        if isinstance(field, list):
            return field
        
        if isinstance(field, str):
            field = field.strip()
            
            # Try JSON parsing first
            if field.startswith('['):
                try:
                    return json.loads(field)
                except:
                    pass
            
            # Try ast.literal_eval for Python list strings
            try:
                parsed = ast.literal_eval(field)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            
            # Handle semicolon-separated strings (common in your data)
            if ';' in field:
                return [item.strip() for item in field.split(';') if item.strip()]
            
            # Single item
            return [field]
        
        return []
    
    # In extract_forum_post_id, add this check:
    def extract_forum_post_id(self, url):
        """Extract post ID from a forum URL"""
        if not isinstance(url, str):
            return None
        
        original_url = url  # Save for debugging
        
        # Sequences format: /s/{sequence_id}/p/{post_id}
        sequences = re.search(r'/s/[^/]+/p/([^/\?#]+)', url)
        if sequences:
            result = sequences.group(1)
            if 'grue' in url:
                print(f"  [extract] Matched sequences: {result}")
            return result
        
        # New format: extract post_id
        new_format = re.search(r'/posts/([^/\?#]+)', url)
        if new_format:
            result = new_format.group(1)
            if 'grue' in url:
                print(f"  [extract] Matched new format: {result}")
            return result
        
        # Discussion format: /r/discussion/lw/XX/slug
        discussion = re.search(r'/r/discussion/lw/[^/]+/([^/\?#]+)', url)
        if discussion:
            slug = discussion.group(1).rstrip('/')
            slug = slug.replace('_', '-')
            if 'grue' in url:
                print(f"  [extract] Matched discussion: {slug}")
            return slug
        
        # Old LW format: /lw/XX/slug
        old_format = re.search(r'/lw/[^/]+/([^/\?#]+)', url)
        if old_format:
            slug = old_format.group(1).rstrip('/')
            slug = slug.replace('_', '-')
            if 'grue' in url:
                print(f"  [extract] Matched old format: {slug}")
            return slug
        
        if 'grue' in url:
            print(f"  [extract] NO MATCH for: {url}")
        return None
    
    def is_forum_post_link(self, url):
        """Check if URL is a link to a specific forum post"""
        if not isinstance(url, str):
            return False
        
        # Accept both lesswrong and lesserwrong
        if not re.search(r'(lesserwrong|lesswrong|alignmentforum)\.(com|org)', url, re.IGNORECASE):
            return False
        
        # Exclude non-post patterns first
        exclude_patterns = [
            r'wiki\.lesswrong',
            r'/user/',
            r'/tag/',
            r'/rationality/',
            r'/comments/?$',
            r'/recentposts',
            r'/top\?',
            r'/message/',
            r'google\.(com|ie)/url',
            r'%20',
            r'/r/[^/]+-drafts/',  # Exclude personal draft sections
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        
        # Include post formats
        if re.search(r'/s/[^/]+/p/', url):
            return True
        if re.search(r'/posts/', url):
            return True
        if re.search(r'/r/discussion/lw/', url):
            return True
        if re.search(r'/lw/[^/]+/[^/]+', url):  # Old format with slug
            return True
        
        return False
    
    def extract_doi(self, url):
        """Extract DOI from a URL or string"""
        if not isinstance(url, str):
            return None
        
        # Pattern that matches DOIs starting with 10.
        # More permissive about what comes after the slash
        doi_patterns = [
            r'doi\.org/(10\.\d+/[^\s\?#<>"]+)',
            r'dx\.doi\.org/(10\.\d+/[^\s\?#<>"]+)',
            r'doi:\s*(10\.\d+/[^\s<>"]+)',
            r'\b(10\.\d{4,9}/[^\s;<>"]+)',  # Bare DOI in text
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                doi = match.group(1).strip()
                # Clean up trailing punctuation that's not part of DOI
                doi = re.sub(r'[.,;)\]]+$', '', doi)
                return doi
        return None

    def normalize_doi(self, doi):
        """
        Comprehensive DOI normalization for matching.
        Handles all edge cases discovered during testing.
        """
        if not doi or pd.isna(doi):
            return None
        
        doi_str = str(doi).lower().strip()
        
        # Remove common prefixes
        doi_str = doi_str.replace('https://doi.org/', '')
        doi_str = doi_str.replace('http://doi.org/', '')
        doi_str = doi_str.replace('https://dx.doi.org/', '')
        doi_str = doi_str.replace('http://dx.doi.org/', '')
        doi_str = doi_str.replace('doi:', '')
        
        # Remove URL fragments (e.g., #page-1, #.u14eh_ldvjm)
        doi_str = re.sub(r'#.*$', '', doi_str)
        
        # Remove query parameters (e.g., ?uid=...)
        if '?' in doi_str:
            doi_str = doi_str.split('?')[0]
        
        # Remove &amp; and other HTML entities
        doi_str = doi_str.replace('&amp;', '').replace('&amp', '')
        
        # Remove common path suffixes
        doi_str = re.sub(r'/abstract$', '', doi_str)
        doi_str = re.sub(r'/full$', '', doi_str)
        doi_str = re.sub(r'/pdf$', '', doi_str)
        doi_str = re.sub(r'/epdf$', '', doi_str)
        doi_str = re.sub(r'/issuetoc$', '', doi_str)
        
        # Remove version indicators (v1.full, v2.full, etc.)
        doi_str = re.sub(r'v\d+\.full$', '', doi_str)
        
        # Remove weird caret suffixes (.^node, .^b, .^f, etc.)
        doi_str = re.sub(r'\.\^[a-z]+$', '', doi_str, flags=re.IGNORECASE)
        
        # Remove bracket artifacts and anything after them
        doi_str = re.sub(r'\[[^\]]*$', '', doi_str)
        
        # Remove trailing parentheses that look incomplete
        if doi_str.endswith('('):
            doi_str = doi_str[:-1]
        
        # Remove trailing punctuation
        doi_str = re.sub(r'[.,;:!?\)]+$', '', doi_str)
        
        return doi_str.strip() if doi_str else None
    
    def create_edges_openalex_citations(self, openalex_df):
        """Create edges between OpenAlex works (paper citations)"""
        if openalex_df.empty or 'referenced_works' not in openalex_df.columns:
            print("Warning: No referenced_works column in OpenAlex data")
            return pd.DataFrame(columns=['citing_work_id', 'cited_work_id'])
        
        edges = []
        
        # Create lookup of all work IDs in our dataset
        available_work_ids = set(openalex_df['id'].dropna())
        print(f"Total works in dataset: {len(available_work_ids)}")
        
        total_references = 0
        matched_references = 0
        
        for _, row in openalex_df.iterrows():
            citing_id = row.get('id')
            if pd.isna(citing_id):
                continue
            
            refs_raw = row.get('referenced_works')
            if pd.isna(refs_raw) or refs_raw == '':
                continue
            
            # Parse the referenced_works field
            refs = self.parse_list_field(refs_raw)
            total_references += len(refs)
            
            for cited_id in refs:
                if not cited_id or pd.isna(cited_id):
                    continue
                
                cited_id = str(cited_id).strip()
                
                # Only create edge if cited work is in our dataset
                if cited_id in available_work_ids:
                    matched_references += 1
                    edges.append({
                        'citing_work_id': citing_id,
                        'cited_work_id': cited_id
                    })
        
        print(f"Total references found: {total_references}")
        print(f"References to works in our dataset: {matched_references}")
        print(f"Match rate: {matched_references/total_references*100:.1f}%")
        
        edges_df = pd.DataFrame(edges)
        if not edges_df.empty:
            edges_df = edges_df.drop_duplicates()
        
        print(f"Found {len(edges_df)} unique OpenAlex-to-OpenAlex citations")
        return edges_df
    
    def create_edges_post_citations(self, df):
        """Extract citations between forum posts"""
        edges = []
        
        if 'extracted_links' not in df.columns:
            print("Warning: No extracted_links column found")
            return pd.DataFrame(columns=['citing_post_id', 'cited_post_id'])
        
        # Create TWO lookups: by post_id AND by slug
        postid_lookup = {}
        slug_lookup = {}
        
        for _, row in df.iterrows():
            post_id = row.get('post_id')
            slug = row.get('slug')
            
            if pd.notna(post_id):
                postid_lookup[str(post_id)] = post_id
            
            if pd.notna(slug) and pd.notna(post_id):
                slug_lookup[str(slug)] = post_id
        
        print(f"Built lookups: {len(postid_lookup)} post_ids, {len(slug_lookup)} slugs")
        
        # Track for debugging
        total_links = 0
        forum_links = 0
        matched_by_postid = 0
        matched_by_slug = 0
        unmatched_samples = []
        
        for _, row in df.iterrows():
            citing_id = row['post_id']
            links_str = row.get('extracted_links')
            
            if pd.isna(links_str) or links_str == '':
                continue
            
            if isinstance(links_str, str):
                links = [link.strip() for link in links_str.split(';') if link.strip()]
            else:
                links = self.parse_list_field(links_str)
            
            total_links += len(links)
            
            for link in links:
                if not isinstance(link, str):
                    continue
                
                if not self.is_forum_post_link(link):
                    continue
                
                forum_links += 1
                cited_id = None
                
                # Extract FIRST
                post_id_or_slug = self.extract_forum_post_id(link)
                
                # Then do matching
                if post_id_or_slug:
                    if post_id_or_slug in postid_lookup:
                        cited_id = postid_lookup[post_id_or_slug]
                        matched_by_postid += 1
                    elif post_id_or_slug in slug_lookup:
                        cited_id = slug_lookup[post_id_or_slug]
                        matched_by_slug += 1
                else:
                    # Save some samples of unmatched links for debugging
                    if len(unmatched_samples) < 10:
                        unmatched_samples.append(link)
                
                if cited_id and cited_id != citing_id:
                    edges.append({
                        'citing_post_id': citing_id,
                        'cited_post_id': cited_id,
                    })
        
        print(f"Total links processed: {total_links}")
        print(f"Forum links found: {forum_links}")
        print(f"Matched by post_id: {matched_by_postid}")
        print(f"Matched by slug: {matched_by_slug}")
        print(f"\nSample unmatched links:")
        for link in unmatched_samples:
            print(f"  {link}")
        
        edges_df = pd.DataFrame(edges)
        if not edges_df.empty:
            edges_df = edges_df.drop_duplicates()
        
        print(f"\nFound {len(edges_df)} unique post-to-post citations")
        return edges_df
    
    def create_edges_post_openalex(self, forum_df, openalex_df):
        """Extract citations from posts to OpenAlex works"""
        edges = []
        
        if openalex_df.empty:
            print("Warning: No OpenAlex data to match against")
            return pd.DataFrame(columns=['citing_post_id', 'openalex_id', 'openalex_doi'])
        
        # Check for DOI columns
        has_extracted_dois = 'extracted_dois' in forum_df.columns
        has_extracted_links = 'extracted_links' in forum_df.columns
        
        if not has_extracted_dois and not has_extracted_links:
            print("Warning: No extracted_dois or extracted_links column found")
            return pd.DataFrame(columns=['citing_post_id', 'openalex_id', 'openalex_doi'])
        
        # Create DOI lookup from OpenAlex data
        doi_lookup = {}
        for _, row in openalex_df.iterrows():
            doi = self.normalize_doi(row.get('doi'))
            if doi:
                openalex_id = row.get('id')
                if openalex_id:
                    doi_lookup[doi] = openalex_id
        
        print(f"Built DOI lookup with {len(doi_lookup)} entries")
        sample_dois = list(doi_lookup.keys())[:5]
        print(f"Sample OpenAlex DOIs: {sample_dois}")
        
        # Track stats
        total_dois_found = 0
        matched_dois = 0
        dois_extracted_from_column = 0
        dois_extracted_from_links = 0
        sample_unmatched_dois = []
        
        # Extract citations
        for _, row in forum_df.iterrows():
            citing_id = row['post_id']
            dois = []
            
            # Try extracted_dois first
            if has_extracted_dois:
                dois_str = row.get('extracted_dois')
                if pd.notna(dois_str) and dois_str != '' and dois_str != '[]':
                    if isinstance(dois_str, str):
                        if dois_str.startswith('['):
                            dois_from_col = self.parse_list_field(dois_str)
                        else:
                            dois_from_col = [d.strip() for d in dois_str.split(';') if d.strip()]
                    else:
                        dois_from_col = self.parse_list_field(dois_str)
                    
                    dois.extend(dois_from_col)
                    dois_extracted_from_column += len(dois_from_col)
            
            # If no DOIs found, try extracting from links
            if not dois and has_extracted_links:
                links_str = row.get('extracted_links')
                if pd.notna(links_str) and links_str != '':
                    if isinstance(links_str, str):
                        links = [link.strip() for link in links_str.split(';') if link.strip()]
                    else:
                        links = self.parse_list_field(links_str)
                    
                    for link in links:
                        if isinstance(link, str):
                            doi = self.extract_doi(link)
                            if doi:
                                dois.append(doi)
                                dois_extracted_from_links += 1
            
            total_dois_found += len(dois)
            
            # Match DOIs against OpenAlex
            for doi in dois:
                if not doi:
                    continue
                
                doi_clean = self.normalize_doi(doi)
                if doi_clean and doi_clean in doi_lookup:
                    matched_dois += 1
                    openalex_id = doi_lookup[doi_clean]
                    edges.append({
                        'citing_post_id': citing_id,
                        'openalex_id': openalex_id,
                        'openalex_doi': doi_clean
                    })
                else:
                    # Track unmatched
                    if len(sample_unmatched_dois) < 10 and doi_clean:
                        sample_unmatched_dois.append((doi, doi_clean))
        
        print(f"Total DOIs found in posts: {total_dois_found}")
        print(f"DOIs from extracted_dois column: {dois_extracted_from_column}")
        print(f"DOIs from extracted_links: {dois_extracted_from_links}")
        print(f"Successfully matched to OpenAlex: {matched_dois}")
        
        print(f"\nSample unmatched DOIs:")
        for orig, normalized in sample_unmatched_dois:
            print(f"  Original: {orig}")
            print(f"  Normalized: {normalized}")
            print(f"  In lookup: {normalized in doi_lookup}")
        
        edges_df = pd.DataFrame(edges)
        if not edges_df.empty:
            edges_df = edges_df.drop_duplicates()
        
        print(f"\nFound {len(edges_df)} unique post-to-OpenAlex citations")
        return edges_df
    
    def create_nodes_openalex_works(self, openalex_df):
        """Create OpenAlex works node table"""
        if openalex_df.empty:
            return pd.DataFrame(columns=['openalex_id', 'openalex_doi', 'title', 
                                        'publication_year', 'type', 'cited_by_count'])
        
        works_columns = ['id', 'doi', 'title', 'publication_year', 'type', 'cited_by_count']
        available_cols = [col for col in works_columns if col in openalex_df.columns]
        
        works_df = openalex_df[available_cols].copy()
        works_df = works_df.rename(columns={'id': 'openalex_id', 'doi': 'openalex_doi'})
        
        return works_df.drop_duplicates(subset=['openalex_id'])
    
    def create_nodes_openalex_authors(self, openalex_df):
        """Create OpenAlex authors node table from author_names and author_genders"""
        if openalex_df.empty:
            return pd.DataFrame(columns=['author_id', 'author_name', 'inferred_gender', 
                                        'work_count', 'work_ids'])
        
        if 'author_names' not in openalex_df.columns:
            print("Warning: No 'author_names' column in OpenAlex data")
            return pd.DataFrame(columns=['author_id', 'author_name', 'inferred_gender', 
                                        'work_count', 'work_ids'])
        
        authors_dict = defaultdict(lambda: {
            'inferred_gender': None,
            'work_ids': [],
            'genders': []
        })
        
        for _, row in openalex_df.iterrows():
            work_id = row.get('id')
            if pd.isna(work_id):
                continue
            
            # Parse author names - handle both string and list formats
            author_names_raw = row.get('author_names')
            if pd.isna(author_names_raw) or author_names_raw == '':
                continue
            
            # Try parsing as list first, then fall back to single string
            author_names = self.parse_list_field(author_names_raw)
            if not author_names:  # If parsing failed, treat as single author
                author_names = [str(author_names_raw).strip()]
            
            # Same for genders
            author_genders_raw = row.get('author_genders')
            author_genders = []
            if pd.notna(author_genders_raw) and author_genders_raw != '':
                author_genders = self.parse_list_field(author_genders_raw)
                if not author_genders:  # If parsing failed, treat as single gender
                    author_genders = [str(author_genders_raw).strip()]
            
            # Process each author
            for idx, author_name in enumerate(author_names):
                if not author_name or pd.isna(author_name) or author_name == '':
                    continue
                
                author_id = str(author_name).strip()
                authors_dict[author_id]['work_ids'].append(work_id)
                
                # Store gender if available
                if idx < len(author_genders):
                    gender = author_genders[idx]
                    if gender and not pd.isna(gender) and gender != '':
                        authors_dict[author_id]['genders'].append(gender)
        
        # Convert to DataFrame
        authors_data = []
        for author_id, info in authors_dict.items():
            # Use most common gender
            gender = None
            if info['genders']:
                gender = max(set(info['genders']), key=info['genders'].count)
            
            authors_data.append({
                'author_id': author_id,
                'author_name': author_id,
                'inferred_gender': gender,
                'work_count': len(info['work_ids']),
                'work_ids': json.dumps(info['work_ids'])
            })
        
        authors_df = pd.DataFrame(authors_data)
        print(f"Extracted {len(authors_df)} unique OpenAlex authors")
        return authors_df
    
    # Add this to your create_edges_openalex_authorship function to debug:
    def create_edges_openalex_authorship(self, openalex_df):
        """Create edges between OpenAlex authors and works"""
        if openalex_df.empty or 'author_names' not in openalex_df.columns:
            return pd.DataFrame(columns=['author_id', 'openalex_id', 'author_position'])
        
        edges = []
        works_with_no_authors = 0
        works_processed = 0
        
        for _, row in openalex_df.iterrows():
            work_id = row.get('id')
            if pd.isna(work_id):
                continue
            
            works_processed += 1
            author_names_raw = row.get('author_names')
            
            if pd.isna(author_names_raw) or author_names_raw == '':
                works_with_no_authors += 1
                continue
            
            # Try parsing as list, fall back to single string
            author_names = self.parse_list_field(author_names_raw)
            if not author_names:
                author_names = [str(author_names_raw).strip()]
            
            for idx, author_name in enumerate(author_names):
                if not author_name or pd.isna(author_name) or author_name == '':
                    continue
                
                author_id = str(author_name).strip()
                
                # Determine position
                if len(author_names) == 1:
                    position = 'sole'
                elif idx == 0:
                    position = 'first'
                elif idx == len(author_names) - 1:
                    position = 'last'
                else:
                    position = 'middle'
                
                edges.append({
                    'author_id': author_id,
                    'openalex_id': work_id,
                    'author_position': position,
                    'position_index': idx
                })
        
        print(f"Works processed: {works_processed}")
        print(f"Works with no authors: {works_with_no_authors}")
        
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
        
        # Load data
        print("\n[1/7] Loading forum data...")
        forum_df = self.load_forum_data()
        print(f"Forum data shape: {forum_df.shape}")
        print(f"Sample columns: {list(forum_df.columns)[:15]}")
        
        print("\n[2/7] Loading OpenAlex data...")
        openalex_df = self.load_openalex_data()
        if not openalex_df.empty:
            print(f"OpenAlex data shape: {openalex_df.shape}")
            print(f"OpenAlex columns: {list(openalex_df.columns)}")
        
        # Standardize
        print("\n[3/7] Standardizing column names...")
        forum_df = self.standardize_forum_columns(forum_df)
        
        # Create nodes
        print("\n[4/7] Creating nodes_posts.csv...")
        nodes_posts = self.create_nodes_posts(forum_df)
        nodes_posts.to_csv(self.output_dir / 'nodes_forum_posts.csv', index=False)
        print(f"  → Saved {len(nodes_posts)} posts")
        
        print("\n[5/7] Creating nodes_authors.csv...")
        nodes_authors = self.create_nodes_authors(forum_df)
        nodes_authors.to_csv(self.output_dir / 'nodes_forum_authors.csv', index=False)
        print(f"  → Saved {len(nodes_authors)} authors")
        
        print("\n[6/7] Creating nodes_openalex_works.csv...")
        nodes_openalex = self.create_nodes_openalex_works(openalex_df)
        nodes_openalex.to_csv(self.output_dir / 'nodes_openalex_works.csv', index=False)
        print(f"  → Saved {len(nodes_openalex)} OpenAlex works")
        
        print("\n[6b/7] Creating nodes_openalex_authors.csv...")
        nodes_openalex_authors = self.create_nodes_openalex_authors(openalex_df)
        nodes_openalex_authors.to_csv(self.output_dir / 'nodes_openalex_authors.csv', index=False)
        print(f"  → Saved {len(nodes_openalex_authors)} OpenAlex authors")
        
        # Create edges
        print("\n[7/7] Creating edge tables...")
        
        print("  - Extracting post-to-post citations...")
        edges_citations = self.create_edges_post_citations(forum_df)
        edges_citations.to_csv(self.output_dir / 'edges_post_to_post.csv', index=False)
        
        print("  - Extracting post-to-OpenAlex citations...")
        edges_openalex = self.create_edges_post_openalex(forum_df, openalex_df)
        edges_openalex.to_csv(self.output_dir / 'edges_post_to_openalex.csv', index=False)

        print("  - Extracting OpenAlex authorship edges...")
        edges_authorship = self.create_edges_openalex_authorship(openalex_df)
        edges_authorship.to_csv(self.output_dir / 'edges_openalex_authorship.csv', index=False)

        print("  - Extracting OpenAlex-to-OpenAlex citations...")
        edges_openalex_citations = self.create_edges_openalex_citations(openalex_df)
        edges_openalex_citations.to_csv(self.output_dir / 'edges_openalex_to_openalex.csv', index=False)
        
        # Summary
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"\nNode Tables:")
        print(f"  - nodes_posts.csv: {len(nodes_posts)} rows")
        print(f"  - nodes_authors.csv: {len(nodes_authors)} rows")
        print(f"  - nodes_openalex_works.csv: {len(nodes_openalex)} rows")
        print(f"  - nodes_openalex_authors.csv: {len(nodes_openalex_authors)} rows")
        print(f"\nEdge Tables:")
        print(f"  - edges_post_citations.csv: {len(edges_citations)} rows")
        print(f"  - edges_post_openalex.csv: {len(edges_openalex)} rows")
        print(f"  - edges_openalex_authorship.csv: {len(edges_authorship)} rows")
        print(f"  - edges_openalex_to_openalex.csv: {len(edges_openalex_citations)} rows")

        print(f"Unique post_ids: {forum_df['post_id'].nunique()}")
        print(f"Total rows: {len(forum_df)}")
        print(f"Duplicates: {len(forum_df) - forum_df['post_id'].nunique()}")

if __name__ == "__main__":
    builder = ForumGraphBuilder()
    builder.run_pipeline()