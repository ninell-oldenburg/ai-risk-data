import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import re
from collections import Counter
from pathlib import Path
import warnings
import gc
import pickle
warnings.filterwarnings('ignore')

class MemoryEfficientBERTopic:
    """Memory-optimized BERTopic for large datasets"""
    
    def __init__(self, platform):
        try:
            if platform in ['lw', 'af']:
                self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")
        self.base_path = f"src/processed_data/{self.platform}/02_with_links_and_gender"
        self.blog_posts = []
        self.bertopic_results = {}
        
    def load_csv_files(self, start_year=2015, end_year=2024, max_posts=None):
        """Load CSV files with optional limit"""
        print(f"Loading CSV files from {start_year} to {end_year}...")
        
        all_posts = []
        file_count = 0
        
        for year in range(start_year, end_year + 1):
            year_path = Path(self.base_path) / str(year)
            
            if not year_path.exists():
                continue
                
            for month in range(1, 13):
                month_str = f"{month:02d}"
                csv_path = year_path / f"{year}-{month_str}.csv"
                
                if csv_path.exists():
                    try:
                        # Load only necessary columns
                        df = pd.read_csv(csv_path, usecols=['_id', 'title', 'cleaned_htmlBody'])
                        df.columns = df.columns.str.strip()
            
                        for _, row in df.iterrows():
                            text_content = ""
                            if pd.notna(row['title']):
                                text_content += str(row['title']) + " "
                            if pd.notna(row['cleaned_htmlBody']):
                                text_content += str(row['cleaned_htmlBody'])
                        
                            if text_content.strip():
                                all_posts.append({
                                    '_id': str(row['_id']),
                                    'text': text_content.strip(),
                                    'title': str(row['title']).strip() if pd.notna(row.get('title')) else "",
                                    'year': year,
                                    'month': month,
                                })
                        
                        file_count += 1
                        print(f"Loaded {len(df)} posts from {csv_path}")
                        
                        # Check if we've hit the limit
                        if max_posts and len(all_posts) >= max_posts:
                            print(f"Reached max_posts limit of {max_posts}")
                            break
                            
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
                
                if max_posts and len(all_posts) >= max_posts:
                    break
            
            if max_posts and len(all_posts) >= max_posts:
                break

        self.blog_posts = all_posts[:max_posts] if max_posts else all_posts
        print(f"\nTotal: Loaded {len(self.blog_posts)} blog posts from {file_count} files")
        return len(self.blog_posts) > 0
    
    def preprocess_text(self, text, max_length=5000):
        """Light preprocessing with length limit"""
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Truncate very long texts
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
        return text
    
    def compute_embeddings_in_batches(self, texts, model_name='all-MiniLM-L6-v2', batch_size=32):
        """Compute embeddings in batches to save memory"""
        print(f"Computing embeddings in batches of {batch_size}...")
        print(f"This may take a while for {len(texts)} documents...")
        
        model = SentenceTransformer(model_name)
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(texts)} documents")
                gc.collect()  # Force garbage collection
        
        embeddings = np.vstack(embeddings)
        print(f"Embeddings computed: {embeddings.shape}")
        
        # Clear model from memory
        del model
        gc.collect()
        
        return embeddings
    
    def save_embeddings(self, embeddings, filepath):
        """Save embeddings to disk"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, embeddings)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath):
        """Load embeddings from disk"""
        embeddings = np.load(filepath)
        print(f"Embeddings loaded from {filepath}: {embeddings.shape}")
        return embeddings
    
    def train_bertopic_memory_efficient(self, 
                                       embedding_model='all-MiniLM-L6-v2',
                                       min_topic_size=15,
                                       nr_topics=None,  # Changed default from 'auto'
                                       batch_size=32,
                                       precompute_embeddings=True,
                                       embeddings_path=None):
        """
        Memory-efficient BERTopic training
        
        Args:
            embedding_model: Smaller models like 'all-MiniLM-L6-v2' use less memory
            min_topic_size: Larger values = fewer topics = less memory
            nr_topics: Number of topics (None = no reduction, int = reduce to N topics)
            batch_size: Smaller batches use less memory
            precompute_embeddings: If True, compute embeddings separately
            embeddings_path: Path to save/load embeddings
        """
        print(f"Training BERTopic (memory-efficient mode)...")
        
        # Preprocess texts
        texts = [self.preprocess_text(post['text']) for post in self.blog_posts]
        
        # Remove very short texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if len(text.split()) >= 10:
                valid_texts.append(text)
                valid_indices.append(i)
        
        print(f"Using {len(valid_texts)} posts with sufficient content")
        self.blog_posts = [self.blog_posts[i] for i in valid_indices]
        
        # Handle embeddings
        embeddings = None
        if precompute_embeddings:
            embeddings_exist = embeddings_path and Path(embeddings_path).exists()
            
            if embeddings_exist:
                print("Checking pre-computed embeddings...")
                temp_embeddings = self.load_embeddings(embeddings_path)
                
                # Check if embeddings match current dataset size
                if temp_embeddings.shape[0] == len(valid_texts):
                    print("✓ Embeddings match dataset size")
                    embeddings = temp_embeddings
                else:
                    print(f"✗ Embeddings mismatch: {temp_embeddings.shape[0]} vs {len(valid_texts)} documents")
                    print("Recomputing embeddings...")
                    embeddings = self.compute_embeddings_in_batches(
                        valid_texts, 
                        embedding_model, 
                        batch_size
                    )
                    if embeddings_path:
                        self.save_embeddings(embeddings, embeddings_path)
            else:
                print("Computing embeddings (will be cached for next run)...")
                embeddings = self.compute_embeddings_in_batches(
                    valid_texts, 
                    embedding_model, 
                    batch_size
                )
                if embeddings_path:
                    self.save_embeddings(embeddings, embeddings_path)
        
        # Configure models with memory-efficient settings
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            low_memory=True  # Memory-efficient mode
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            core_dist_n_jobs=1  # Use single core to save memory
        )
        
        vectorizer_model = CountVectorizer(
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 2),
            min_df=2,  # Lower threshold to avoid issues
            max_df=0.95,  # Higher threshold
            max_features=2000  # More features for better topics
        )
        
        # Create BERTopic with pre-computed embeddings
        if embeddings is not None:
            sentence_model = None  # Don't load model if using pre-computed embeddings
        else:
            sentence_model = SentenceTransformer(embedding_model)
        
        self.topic_model = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            nr_topics=nr_topics,
            calculate_probabilities=False,  # Disable to save memory
            verbose=True
        )
        
        # Fit model
        print("Fitting BERTopic model...")
        if embeddings is not None:
            topics = self.topic_model.fit_transform(valid_texts, embeddings)[0]
        else:
            topics = self.topic_model.fit_transform(valid_texts)[0]
        
        print(f"Initial topics discovered: {len(set(topics))}")
        
        # Manual topic reduction if requested (safer than auto)
        if nr_topics is not None and isinstance(nr_topics, int):
            print(f"Manually reducing topics to {nr_topics}...")
            try:
                self.topic_model.reduce_topics(valid_texts, nr_topics=nr_topics)
                topics = self.topic_model.topics_
                print(f"Topics reduced to {nr_topics}")
            except Exception as e:
                print(f"Topic reduction failed: {e}")
                print("Continuing with original topics...")
        
        # Clear embeddings from memory
        del embeddings
        gc.collect()
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        n_topics = len(topic_info) - 1
        
        print(f"\nBERTopic discovered {n_topics} topics")
        print(f"Outliers: {np.sum(np.array(topics) == -1)} documents")
        
        # Add to blog posts
        for i, post in enumerate(self.blog_posts):
            post['bertopic_topic'] = topics[i]
        
        # Calculate topic statistics
        topic_stats = {}
        for topic_id in range(-1, n_topics):
            topic_posts = [post for post in self.blog_posts if post['bertopic_topic'] == topic_id]
            
            if len(topic_posts) == 0:
                continue
            
            year_dist = Counter([post['year'] for post in topic_posts])
            
            try:
                topic_words = self.topic_model.get_topic(topic_id)
                top_words = [(word, score) for word, score in topic_words[:20]] if topic_words else []
            except:
                top_words = []
            
            topic_stats[topic_id] = {
                'size': len(topic_posts),
                'percentage': len(topic_posts) / len(self.blog_posts) * 100,
                'year_distribution': dict(year_dist),
                'top_words': top_words
            }
        
        self.bertopic_results = {
            'model': self.topic_model,
            'n_topics': n_topics,
            'topics': topics,
            'topic_info': topic_info,
            'topic_stats': topic_stats
        }
        
        return self.bertopic_results
    
    def visualize_bertopic_simple(self):
        """Simple matplotlib-only visualizations (no plotly = less memory)"""
        if not self.bertopic_results:
            print("No results available.")
            return
        
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        active_topics = {tid: stats for tid, stats in self.bertopic_results['topic_stats'].items() 
                        if tid >= 0}
        
        # Topic sizes
        ax1 = axes[0, 0]
        topic_ids = sorted(active_topics.keys())
        sizes = [active_topics[tid]['size'] for tid in topic_ids]
        ax1.bar(range(len(sizes)), sizes)
        ax1.set_xlabel('Topic ID')
        ax1.set_ylabel('Number of Posts')
        ax1.set_title(f'Topic Sizes ({len(active_topics)} topics)')
        
        # Year distribution
        ax2 = axes[0, 1]
        years = sorted(set(post['year'] for post in self.blog_posts))
        topic_year_matrix = np.zeros((len(topic_ids), len(years)))
        
        for i, tid in enumerate(topic_ids):
            year_dist = active_topics[tid]['year_distribution']
            for j, year in enumerate(years):
                topic_year_matrix[i, j] = year_dist.get(year, 0)
        
        im = ax2.imshow(topic_year_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax2)
        ax2.set_xticks(range(len(years)))
        ax2.set_xticklabels(years, rotation=45)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Topic')
        ax2.set_title('Distribution by Year')
        
        # Topic size distribution
        ax3 = axes[1, 0]
        ax3.hist(sizes, bins=min(20, len(sizes)), edgecolor='black')
        ax3.set_xlabel('Topic Size')
        ax3.set_ylabel('Number of Topics')
        ax3.set_title('Topic Size Distribution')
        
        # Summary
        ax4 = axes[1, 1]
        summary_text = f"""BERTopic Summary
        
Active Topics: {len(active_topics)}
Outliers: {self.bertopic_results['topic_stats'].get(-1, {}).get('size', 0)}
Total Posts: {len(self.blog_posts)}

Top 5 Topics:
"""
        sorted_topics = sorted(active_topics.items(), 
                              key=lambda x: x[1]['size'], 
                              reverse=True)[:5]
        
        for tid, stats in sorted_topics:
            if stats['top_words']:
                words = ', '.join([w[0] for w in stats['top_words'][:4]])
                summary_text += f"\n{tid}: {words}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')
        
        plt.tight_layout()
        
        import os
        output_path = f"src/metadata/img/{self.platform}/bertopic_summary.pdf"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.show()
    
    def print_summary(self):
        """Print summary"""
        if not self.bertopic_results:
            return
        
        active_topics = {tid: stats for tid, stats in self.bertopic_results['topic_stats'].items() 
                        if tid >= 0}
        
        print(f"\n{'='*60}")
        print(f"BERTOPIC SUMMARY")
        print(f"{'='*60}")
        print(f"Topics: {len(active_topics)}")
        print(f"Outliers: {self.bertopic_results['topic_stats'].get(-1, {}).get('size', 0)}")
        print(f"Total: {len(self.blog_posts)}")
        
        sorted_topics = sorted(active_topics.items(), 
                              key=lambda x: x[1]['size'], 
                              reverse=True)[:10]
        
        for topic_id, stats in sorted_topics:
            print(f"\nTopic {topic_id} ({stats['size']} posts, {stats['percentage']:.1f}%):")
            if stats['top_words']:
                words = [f"{w[0]}({w[1]:.3f})" for w in stats['top_words'][:5]]
                print(f"  {', '.join(words)}")
    
    def save_results(self, output_file='bertopic_results.csv'):
        """Save results"""
        if not self.bertopic_results:
            return
        
        import os
        
        results_data = []
        for post in self.blog_posts:
            results_data.append({
                '_id': post['_id'],
                'title': post['title'],
                'year': post['year'],
                'month': post['month'],
                'topic': post['bertopic_topic'],
            })
        
        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Save model
        model_path = output_file.replace('.csv', '_model')
        self.topic_model.save(model_path, serialization="pickle")
        print(f"Model saved to {model_path}")


def main(platform, max_posts=None, min_topic_size=20, batch_size=16, nr_topics=None):
    """
    Memory-efficient main function
    
    Args:
        platform: 'lw' or 'af'
        max_posts: Limit number of posts (None for all)
        min_topic_size: Minimum topic size (higher = less topics = less memory)
        batch_size: Batch size for embeddings (lower = less memory)
        nr_topics: Reduce to this many topics after discovery (None = no reduction)
    """
    print("\n" + "="*60)
    print("MEMORY-EFFICIENT BERTOPIC")
    print("="*60)
    
    analyzer = MemoryEfficientBERTopic(platform)
    
    if not analyzer.load_csv_files(start_year=2015, end_year=2024, max_posts=max_posts):
        return
    
    embeddings_path = f"src/metadata/embeddings/{analyzer.platform}_embeddings.npy"
    
    analyzer.train_bertopic_memory_efficient(
        embedding_model='all-MiniLM-L6-v2',  # Small, efficient model
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,  # Pass topic reduction parameter
        batch_size=batch_size,
        precompute_embeddings=True,
        embeddings_path=embeddings_path
    )
    
    analyzer.print_summary()
    analyzer.visualize_bertopic_simple()
    
    output_path = f'src/metadata/clustering_results/{analyzer.platform}/bertopic_results.csv'
    analyzer.save_results(output_path)
    
    print("\nComplete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("USAGE: python optimized_bertopic.py <FORUM> [MAX_POSTS] [MIN_TOPIC_SIZE] [BATCH_SIZE] [NR_TOPICS]")
        print("\nParams:")
        print("  FORUM: 'lw' or 'af' (required)")
        print("  MAX_POSTS: Limit posts (optional, e.g., 5000 for testing)")
        print("  MIN_TOPIC_SIZE: Minimum topic size (default: 20, higher = less memory)")
        print("  BATCH_SIZE: Embedding batch size (default: 16, lower = less memory)")
        print("  NR_TOPICS: Reduce to N topics after discovery (optional, e.g., 30)")
        print("\nExamples:")
        print("  python optimized_bertopic.py lw 5000 25 16      # Test with 5k posts")
        print("  python optimized_bertopic.py af None 30 8 25    # All posts, reduce to 25 topics")
        print("  python optimized_bertopic.py lw None 20 16 None # All posts, no reduction")
        sys.exit(1)
    
    platform = sys.argv[1]
    max_posts = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'None' else None
    min_topic_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    nr_topics = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != 'None' else None
    
    main(platform, max_posts, min_topic_size, batch_size, nr_topics)