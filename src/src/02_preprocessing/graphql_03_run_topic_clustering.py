import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import pickle
from collections import Counter
import sys
import time
import csv

# BERTopic and dependencies
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')

class EmbeddingTopicModeling:
    """
    BERTopic-based topic modeling with embeddings
    
    Advantages over LDA:
    - Captures semantic meaning (not just word co-occurrence)
    - Works better with technical/specialized vocabulary
    - Automatic topic number selection
    - Better coherence for modern text
    """
    
    def __init__(self, platform):
        try:
            if platform in ['lw', 'af']:
                self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")
        
        self.base_path = f"src/processed_data/{self.platform}/02_with_links_and_gender"
        self.blog_posts = []
        self.topic_model = None
        self.embeddings = None
        
    def load_csv_files(self, max_posts=None):
        """Load all CSV files"""
        print(f"Loading CSV files from {self.platform}...")
        
        all_posts = []
        file_count = 0
        
        years = sorted([
            int(name) for name in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, name)) and name.isdigit()
        ])
        
        for year in years:
            year_path = Path(self.base_path) / str(year)
            
            if not year_path.exists():
                continue
                
            for month in range(1, 13):
                month_str = f"{month:02d}"
                csv_path = year_path / f"{year}-{month_str}.csv"
                
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        df.columns = df.columns.str.strip()
            
                        for _, row in df.iterrows():
                            text_content = ""
                            if pd.notna(row['title']):
                                text_content += str(row['title']) + " "
                            if pd.notna(row['cleaned_htmlBody']):
                                text_content += str(row['cleaned_htmlBody'])
                        
                            if text_content.strip() and len(text_content.strip()) > 100:
                                all_posts.append({
                                    '_id': str(row['_id']),
                                    'text': text_content.strip(),
                                    'title': str(row['title']).strip() if 'title' in row and pd.notna(row['title']) else "",
                                    'year': year,
                                    'month': month,
                                    'file': str(csv_path)
                                })
                        
                        file_count += 1
                        print(f"  Loaded {len(df)} posts from {year}-{month_str}")
                        
                        if max_posts and len(all_posts) >= max_posts:
                            break
                            
                    except Exception as e:
                        print(f"  Error reading {csv_path}: {e}")
                
                if max_posts and len(all_posts) >= max_posts:
                    break
            
            if max_posts and len(all_posts) >= max_posts:
                break

        self.blog_posts = all_posts[:max_posts] if max_posts else all_posts
        print(f"\nTotal: Loaded {len(self.blog_posts)} blog posts from {file_count} files")
        return len(self.blog_posts) > 0
    
    def train_topic_model(self, 
                        min_topic_size=None,
                        n_neighbors=25,
                        n_components=5,
                        min_cluster_size=None,
                        embedding_model='all-MiniLM-L6-v2',
                        nr_topics='auto',
                        reduce_outliers=True,
                        verbose=True):
        """
        Train BERTopic model
        
        Parameters:
        -----------
        min_topic_size : int (default 1% of the input posts)
            Minimum documents per topic - HIGH VALUES = fewer topics
        n_neighbors : int (default: 15)
            UMAP n_neighbors - HIGHER = broader, fewer topics (try 25-50)
        n_components : int (default: 5)
            UMAP dimensions
        min_cluster_size : int (default 1% of the input posts)
            HDBSCAN minimum cluster size - MUST match min_topic_size
        embedding_model : str
            Sentence transformer model
        nr_topics : int or 'auto'
            Target number of topics to reduce to (recommended: 15-30)
        reduce_outliers : bool
            Whether to reduce outliers
        """
         
        print(f"\n{'='*60}")
        print(f"TRAINING BERTOPIC MODEL")
        print(f"{'='*60}")

        docs = [post['text'] for post in self.blog_posts]
        self.docs = docs  # store for later evaluation

        # === Auto-adjust cluster size ===
        if min_topic_size is None:
            min_topic_size = int(len(docs) * 0.02)  # 2% of corpus, but at least 20
            print(f"Auto-set min_topic_size to {min_topic_size} (1% of {len(docs)})")

        if min_cluster_size is None:
            min_cluster_size = min_topic_size
            print(f"Auto-set min_cluster_size to {min_cluster_size}")
        
        # 1. Embedding Model
        print(f"\n1. Loading embedding model: {embedding_model}")
        sentence_model = SentenceTransformer(embedding_model)
        
        # 2. UMAP for dimensionality reduction
        print(f"2. Configuring UMAP (n_neighbors={n_neighbors}, n_components={n_components})")
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # 3. HDBSCAN for clustering
        print(f"3. Configuring HDBSCAN (min_cluster_size={min_cluster_size})")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(10, min_cluster_size // 10),  # Scale with cluster size
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            cluster_selection_epsilon=0.0  # More aggressive merging
        )
        
        # 4. Vectorizer - custom stop words for AI/rationality content
        custom_stop_words = [
            'think', 'like', 'want', 'know', 'people', 'thing', 'things',
            'good', 'better', 'make', 'way', 'just', 'really', 'dont',
            'thats', 'im', 'youre', 'ive', 'lot', 'probably', 'basically'
        ]
        
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1, 
            max_df=1.0
        )
        
        # 5. Representation models for better topic descriptions
        # KeyBERT-inspired: extract keywords semantically similar to topic
        keybert_model = KeyBERTInspired()
        
        # MMR: diverse keywords (avoid redundancy)
        mmr_model = MaximalMarginalRelevance(diversity=0.3)
        
        # 6. Create BERTopic model
        print(f"4. Creating BERTopic model")
        self.topic_model = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=[keybert_model, mmr_model],
            top_n_words=10,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=verbose
        )
        
        # 7. Fit model and generate embeddings
        print(f"\n5. Generating embeddings for {len(docs)} documents...")
        print("   (This may take a few minutes...)")
        
        topics, probs = self.topic_model.fit_transform(docs)
        
        # Get embeddings (useful for visualization)
        self.embeddings = sentence_model.encode(docs, show_progress_bar=True)
        
        # 8. Store results
        topic_info = self.topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
        
        print(f"\n{'='*60}")
        print(f"MODEL TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Topics discovered: {n_topics}")
        print(f"Outliers (topic -1): {sum(t == -1 for t in topics)} documents")
        print(f"Documents with topics: {sum(t != -1 for t in topics)} documents")
        
        # Optionally reduce outliers
        if reduce_outliers and sum(t == -1 for t in topics) > 0:
            print(f"\nReducing outliers...")
            new_topics = self.topic_model.reduce_outliers(docs, topics)
            print(f"  Outliers after reduction: {sum(t == -1 for t in new_topics)}")
            topics = new_topics
            
            # Update probabilities
            probs = self.topic_model.probabilities_
        
        # Calculate reasonable number of topics based on corpus size
        docs_per_topic = len(docs) // 30  # Aim for ~30 topics
        recommended_topics = max(15, min(40, len(docs) // min_cluster_size))
        
        # Auto-reduce if we have too many topics
        if nr_topics == 'auto':
            if n_topics > 50:
                target = min(30, recommended_topics)
                print(f"\n⚠️  {n_topics} topics is quite high!")
                print(f"   With {len(docs)} docs and min_topic_size={min_topic_size},")
                print(f"   Automatically reducing to {target} topics...")
                self.topic_model.reduce_topics(docs, nr_topics=target)
                topics = self.topic_model.topics_
                probs = self.topic_model.probabilities_
                
                n_topics = len(self.topic_model.get_topic_info()) - 1
                print(f"   ✓ Topics after reduction: {n_topics}")
            else:
                print(f"   {n_topics} topics seems reasonable for this corpus")
        elif isinstance(nr_topics, int):
            print(f"\nReducing to {nr_topics} topics...")
            self.topic_model.reduce_topics(docs, nr_topics=nr_topics)
            topics = self.topic_model.topics_
            probs = self.topic_model.probabilities_
            
            n_topics = len(self.topic_model.get_topic_info()) - 1
            print(f"   ✓ Topics after reduction: {n_topics}")
        
        # Add topic assignments to blog posts
        for i, post in enumerate(self.blog_posts):
            post['topic'] = int(topics[i])
            if probs is not None:
                topic_id = topics[i]
                if topic_id != -1:
                    post['topic_probability'] = float(probs[i][topic_id])
                else:
                    post['topic_probability'] = 0.0
            else:
                post['topic_probability'] = 0.0
        
        return topics, probs
    
    def reduce_topics(self, nr_topics):
        """
        Reduce number of topics by merging similar ones
        Useful if automatic detection creates too many topics
        """
        if self.topic_model is None:
            print("Train model first!")
            return
        
        print(f"\nReducing to {nr_topics} topics...")
        docs = [post['text'] for post in self.blog_posts]
        
        self.topic_model.reduce_topics(docs, nr_topics=nr_topics)
        
        # Update topic assignments
        new_topics = self.topic_model.topics_
        for i, post in enumerate(self.blog_posts):
            post['topic'] = int(new_topics[i])
        
        print(f"Topics after reduction: {len(self.topic_model.get_topic_info()) - 1}")
    
    def get_topic_statistics(self):
        """Calculate detailed topic statistics"""
        if self.topic_model is None:
            return None
        
        topic_info = self.topic_model.get_topic_info()
        topics = [post['topic'] for post in self.blog_posts]
        
        stats = []
        for topic_id in sorted(set(topics)):
            if topic_id == -1:  # Skip outliers
                continue
            
            topic_posts = [p for p in self.blog_posts if p['topic'] == topic_id]
            
            # Year distribution
            year_dist = Counter([p['year'] for p in topic_posts])
            
            # Average probability
            avg_prob = np.mean([p['topic_probability'] for p in topic_posts])
            
            # Get top words
            topic_words = self.topic_model.get_topic(topic_id)
            
            stats.append({
                'topic_id': topic_id,
                'size': len(topic_posts),
                'percentage': len(topic_posts) / len(self.blog_posts) * 100,
                'avg_probability': avg_prob,
                'year_distribution': dict(year_dist),
                'top_words': topic_words,
                'representative_docs': self._get_representative_docs(topic_id, n=3)
            })
        
        return sorted(stats, key=lambda x: x['size'], reverse=True)
    
    def _get_representative_docs(self, topic_id, n=3):
        """Get most representative documents for a topic"""
        topic_posts = [(i, p) for i, p in enumerate(self.blog_posts) if p['topic'] == topic_id]
        
        # Sort by probability
        topic_posts.sort(key=lambda x: x[1]['topic_probability'], reverse=True)
        
        return [{'title': p['title'], 'prob': p['topic_probability']} 
                for i, p in topic_posts[:n]]
    
    def visualize_topics(self):
        """Create comprehensive visualizations"""
        if self.topic_model is None:
            print("Train model first!")
            return
        
        print("\nCreating visualizations...")
        
        stats = self.get_topic_statistics()
        topics = [p['topic'] for p in self.blog_posts]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Topic sizes
        ax1 = axes[0, 0]
        topic_ids = [s['topic_id'] for s in stats]
        sizes = [s['size'] for s in stats]
        
        bars = ax1.bar(range(len(sizes)), sizes, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Topic ID', fontsize=12)
        ax1.set_ylabel('Number of Documents', fontsize=12)
        ax1.set_title(f'Topic Sizes ({len(stats)} topics)', fontsize=14, fontweight='bold')
        
        if len(topic_ids) <= 30:
            ax1.set_xticks(range(len(topic_ids)))
            ax1.set_xticklabels(topic_ids, rotation=45)
        
        # 2. Topic distribution pie chart (top 10)
        ax2 = axes[0, 1]
        top_10 = stats[:10]
        other_size = sum(s['size'] for s in stats[10:])
        
        pie_sizes = [s['size'] for s in top_10]
        pie_labels = [f"T{s['topic_id']}" for s in top_10]
        
        if other_size > 0:
            pie_sizes.append(other_size)
            pie_labels.append('Other')
        
        ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Topic Distribution (Top 10)', fontsize=14, fontweight='bold')
        
        # 3. Topics over time
        ax3 = axes[0, 2]
        years = sorted(set(p['year'] for p in self.blog_posts))
        
        # Create matrix: topics x years
        n_topics_to_show = min(10, len(stats))
        topic_year_matrix = np.zeros((n_topics_to_show, len(years)))
        
        for i, s in enumerate(stats[:n_topics_to_show]):
            for j, year in enumerate(years):
                topic_year_matrix[i, j] = s['year_distribution'].get(year, 0)
        
        im = ax3.imshow(topic_year_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax3, label='# Documents')
        
        ax3.set_xticks(range(len(years)))
        ax3.set_xticklabels(years, rotation=45)
        ax3.set_yticks(range(n_topics_to_show))
        ax3.set_yticklabels([f"T{s['topic_id']}" for s in stats[:n_topics_to_show]])
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Topic', fontsize=12)
        ax3.set_title('Topic Trends Over Time', fontsize=14, fontweight='bold')
        
        # 4. Topic quality (size vs probability)
        ax4 = axes[1, 0]
        x_sizes = [s['size'] for s in stats]
        y_probs = [s['avg_probability'] for s in stats]
        
        scatter = ax4.scatter(x_sizes, y_probs, s=100, alpha=0.6, c=range(len(stats)), cmap='viridis')
        ax4.set_xlabel('Topic Size', fontsize=12)
        ax4.set_ylabel('Average Probability', fontsize=12)
        ax4.set_title('Topic Quality: Size vs Confidence', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Top topics summary
        ax5 = axes[1, 1]
        summary_text = f"BERTopic Results\n\n"
        summary_text += f"Total Topics: {len(stats)}\n"
        summary_text += f"Total Documents: {len(self.blog_posts)}\n"
        summary_text += f"Outliers: {sum(t == -1 for t in topics)}\n\n"
        summary_text += f"Top 5 Topics:\n\n"
        
        for i, s in enumerate(stats[:5], 1):
            top_words = ', '.join([w[0] for w in s['top_words'][:5]])
            summary_text += f"{i}. Topic {s['topic_id']} ({s['size']} docs, {s['percentage']:.1f}%)\n"
            summary_text += f"   {top_words}\n\n"
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax5.axis('off')
        
        # 6. Size distribution histogram
        ax6 = axes[1, 2]
        ax6.hist(sizes, bins=min(30, len(sizes)), edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Topic Size', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.set_title('Topic Size Distribution', fontsize=14, fontweight='bold')
        ax6.axvline(np.median(sizes), color='red', linestyle='--', 
                   label=f'Median: {np.median(sizes):.0f}')
        ax6.legend()
        
        plt.tight_layout()
        
        # Save
        output_path = f"src/metadata/img/{self.platform}/bertopic_results.pdf"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.show()
    
    def visualize_topic_hierarchy(self):
        """Visualize hierarchical structure of topics"""
        if self.topic_model is None:
            return
        
        print("Creating topic hierarchy visualization...")
        try:
            fig = self.topic_model.visualize_hierarchy()
            
            output_path = f"src/metadata/img/{self.platform}/topic_hierarchy.html"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            print(f"Hierarchy saved to {output_path}")
        except Exception as e:
            print(f"Could not create hierarchy: {e}")
    
    def visualize_topic_space(self):
        """Visualize topics in 2D space (interactive)"""
        if self.topic_model is None:
            return
        
        print("Creating topic space visualization...")
        try:
            fig = self.topic_model.visualize_topics()
            
            output_path = f"src/metadata/img/{self.platform}/topic_space.html"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            print(f"Topic space saved to {output_path}")
        except Exception as e:
            print(f"Could not create topic space: {e}")

    def save_detailed_topics(self, output_path=None, n_topics=10):
        """
        Save the same output as print_detailed_topics() to a text file.
        """
        if output_path is None:
            output_path = f"src/metadata/clustering_results/{self.platform}/detailed_topics.txt"

        # Capture printed output
        import io, sys
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        try:
            self.print_detailed_topics(n_topics=n_topics)
        finally:
            sys.stdout = old_stdout

        content = buffer.getvalue()
        
        # Make sure folder exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        print(f"✅ Saved detailed topics to: {output_path}")
    
    def print_detailed_topics(self, n_topics=10):
        """Print detailed information about top topics"""
        stats = self.get_topic_statistics()
        
        print(f"\n{'='*80}")
        print(f"DETAILED TOPIC ANALYSIS (Top {min(n_topics, len(stats))} topics)")
        print(f"{'='*80}")
        
        for i, s in enumerate(stats[:n_topics], 1):
            print(f"\n{'─'*80}")
            print(f"Topic {s['topic_id']}: {s['size']} documents ({s['percentage']:.2f}%)")
            print(f"Average probability: {s['avg_probability']:.3f}")
            
            print(f"\nTop words:")
            for word, score in s['top_words'][:10]:
                print(f"  • {word:20s} ({score:.4f})")
            
            print(f"\nRepresentative documents:")
            for j, doc in enumerate(s['representative_docs'], 1):
                print(f"  {j}. {doc['title'][:70]}... (prob: {doc['prob']:.3f})")
            
            print(f"\nTemporal distribution:")
            year_items = sorted(s['year_distribution'].items())
            for year, count in year_items:
                bar = '█' * int(count / max(s['year_distribution'].values()) * 30)
                print(f"  {year}: {bar} {count}")
    
    def save_results(self, output_file=None):
        """Save results to CSV and pickle model"""
        if self.topic_model is None:
            print("No model to save!")
            return
        
        if output_file is None:
            output_file = f'src/metadata/clustering_results/{self.platform}/bertopic_results.csv'
        
        # Prepare data
        results_data = []
        for post in self.blog_posts:
            results_data.append({
                '_id': post['_id'],
                'title': post['title'],
                'year': post['year'],
                'month': post['month'],
                'topic': post['topic'],
                'topic_probability': post['topic_probability'],
                'file': post['file']
            })
        
        # Save CSV
        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Save model
        model_path = output_file.replace('.csv', '_model.pkl')
        self.topic_model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save embeddings
        embeddings_path = output_file.replace('.csv', '_embeddings.npy')
        np.save(embeddings_path, self.embeddings)
        print(f"Embeddings saved to {embeddings_path}")

    def compute_diversity_from_model(self, top_n_words=10):
        topics = self.topic_model.get_topics()
        top_words_per_topic = [
            [w for w, _ in ws][:top_n_words] for tid, ws in topics.items() if tid != -1
        ]
        all_words = [w for t in top_words_per_topic for w in t]
        return (len(set(all_words)) / len(all_words)) if all_words else 0.0

    def compute_coherence_from_docs(self, docs, top_n_words=10):
        tokenized_docs = [d.split() for d in docs]
        dictionary = Dictionary(tokenized_docs)
        topics = self.topic_model.get_topics()
        top_words_per_topic = [
            [w for w, _ in ws][:top_n_words] for tid, ws in topics.items() if tid != -1
        ]
        if not top_words_per_topic:
            return 0.0
        cm = CoherenceModel(topics=top_words_per_topic, texts=tokenized_docs, dictionary=dictionary, coherence="c_v")
        return cm.get_coherence()

    def sweep_parameters(self,
                        n_neighbors_list=[10,15,25,50],
                        min_topic_size_list=[50,100,200],
                        n_components=5,
                        embedding_model=None,
                        top_n_words=10,
                        max_posts_for_sweep=5000,
                        output_dir="sweep_results"):
        """
        Sweep UMAP n_neighbors and BERTopic min_topic_size (and min_cluster_size),
        retrain model for each combo, and record: n_neighbors, min_topic_size,
        num_topics, coherence (c_v), diversity.
        """
        # prepare output
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(output_dir) / f"sweep_{int(time.time())}.csv"

        # Use a representative subset for sweeps to save time
        docs_all = [p["text"] for p in self.blog_posts]
        docs = docs_all[:max_posts_for_sweep] if max_posts_for_sweep and len(docs_all) > max_posts_for_sweep else docs_all

        # If embedding_model provided, use it; otherwise reuse trained embedding if present
        for nn in n_neighbors_list:
            for mts in min_topic_size_list:
                print(f"\n--- SWEEP: n_neighbors={nn}, min_topic_size={mts} ---")
                # Build UMAP & HDBSCAN & BERTopic with these params
                from umap import UMAP
                from hdbscan import HDBSCAN
                from sentence_transformers import SentenceTransformer
                from bertopic import BERTopic
                from sklearn.feature_extraction.text import CountVectorizer
                from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

                # optionally reuse embedding model if set in self (expensive otherwise)
                model_name = embedding_model or getattr(self, "embedding_model_name", "all-MiniLM-L6-v2")
                sentence_model = SentenceTransformer(model_name)

                umap_model = UMAP(n_neighbors=nn, n_components=n_components, min_dist=0.0, metric="cosine", random_state=42)
                hdbscan_model = HDBSCAN(min_cluster_size=mts, min_samples=max(1, mts//10), metric='euclidean', prediction_data=True)

                vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words='english', min_df=2, max_df=0.95)
                keybert_model = KeyBERTInspired()
                mmr_model = MaximalMarginalRelevance(diversity=0.3)

                topic_model = BERTopic(embedding_model=sentence_model,
                                    umap_model=umap_model,
                                    hdbscan_model=hdbscan_model,
                                    vectorizer_model=vectorizer_model,
                                    representation_model=[keybert_model, mmr_model],
                                    top_n_words=top_n_words,
                                    min_topic_size=mts,
                                    nr_topics='auto',
                                    calculate_probabilities=False,
                                    verbose=False)

                # Fit (this is the expensive step)
                topics, probs = topic_model.fit_transform(docs)

                # Attach to self temporarily so compute functions work
                old_model = getattr(self, "topic_model", None)
                self.topic_model = topic_model
                self.docs = docs

                # Compute metrics
                num_topics = len(topic_model.get_topic_info()) - 1
                diversity = self.compute_diversity_from_model(top_n_words=top_n_words)
                coherence = self.compute_coherence_from_docs(docs, top_n_words=top_n_words)

                print(f"-> topics: {num_topics}, coherence: {coherence:.3f}, diversity: {diversity:.3f}")

                # Write CSV row (append)
                header = ['n_neighbors', 'min_topic_size', 'n_topics', 'coherence_c_v', 'diversity', 'time_s']
                if not csv_path.exists():
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([nn, mts, num_topics, round(coherence, 4), round(diversity, 4), int(time.time())])

                # restore old model if present
                if old_model is not None:
                    self.topic_model = old_model

        # After sweep, plot coherence vs diversity
        import pandas as pd
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8,6))
        sc = plt.scatter(df['diversity'], df['coherence_c_v'], s=df['n_topics']*2, cmap='viridis')
        for idx, row in df.iterrows():
            plt.text(row['diversity']+0.001, row['coherence_c_v']+0.001, f"nn={row['n_neighbors']},mts={row['min_topic_size']}", fontsize=8)
        plt.xlabel('Diversity')
        plt.ylabel('Coherence (C_v)')
        plt.title('Sweep: Diversity vs Coherence (marker ~ #topics)')
        plt.grid(True)
        plot_path = Path(output_dir) / "sweep_coherence_vs_diversity.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Sweep finished. CSV: {csv_path}, Plot: {plot_path}")
        return csv_path, plot_path

    def evaluate_topics(self):
        """
        Compute and print topic coherence (C_v) and diversity for the current model.
        Works for any BERTopic version.
        """
        model = self.topic_model
        docs = getattr(self, "docs", [p["text"] for p in self.blog_posts])

        print("\n" + "=" * 80)
        print("EVALUATING TOPIC MODEL")
        print("=" * 80)

        # topic diversity
        topics = model.get_topics()
        top_words_per_topic = [
            [word for word, _ in words] for tid, words in topics.items() if tid != -1
        ]
        all_words = [w for topic in top_words_per_topic for w in topic]
        unique_words = set(all_words)
        topic_diversity = len(unique_words) / len(all_words) if all_words else 0.0

        # coherence score (c_v)
        tokenized_docs = [d.split() for d in docs]
        dictionary = Dictionary(tokenized_docs)

        cm = CoherenceModel(
            topics=top_words_per_topic,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v"
        )
        topic_coherence = cm.get_coherence()

        # --- 3️⃣ Print results ---
        print(f"Topic Coherence (C_v): {topic_coherence:.3f}")
        print(f"Topic Diversity: {topic_diversity:.3f}")
        print("-" * 80)

        # Store for reuse in summaries
        self.topic_coherence = topic_coherence
        self.topic_diversity = topic_diversity

        return topic_coherence, topic_diversity

    def plot_topic_distribution(self, output_path=None):
        """
        Create and save a bar plot of the number of posts per topic.
        """
        if output_path is None:
            output_path = f"{self.platform}_topic_distribution.png"

        topic_info = self.topic_model.get_topic_info()
        topic_info = topic_info[topic_info.Topic != -1]
        topic_info = topic_info.sort_values("Count", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(topic_info)), topic_info["Count"])
        plt.xticks(range(len(topic_info)), topic_info["Topic"], rotation=90)
        plt.xlabel("Topic ID")
        plt.ylabel("Number of Posts")
        plt.title(f"Topic Distribution ({self.platform.upper()})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"📊 Saved topic distribution plot: {output_path}")

def main(platform, max_posts=None):
    """
    Main function to run BERTopic analysis
    
    Args:
        platform: 'lw' or 'af'
        max_posts: Limit number of posts (None = all)
        min_topic_size: Minimum documents per topic (400+ for large corpora)
        nr_topics: Target number of topics (15-30 is reasonable)
        embedding_model: Which sentence transformer to use
    """
    print("\n" + "="*80)
    print("BERTOPIC - EMBEDDING-BASED TOPIC MODELING")
    print("="*80)
    
    analyzer = EmbeddingTopicModeling(platform)
    
    # Load data
    if not analyzer.load_csv_files(max_posts=max_posts):
        print("Failed to load data!")
        return
    
    analyzer.sweep_parameters(
        n_neighbors_list=[10,15,25,50],
        min_topic_size_list=[50,100,200],
        embedding_model="all-mpnet-base-v2",   # optional; slow but better
        max_posts_for_sweep=None,
        output_dir="sweep_results_run1"
    )
    
    # Train model
    analyzer.train_topic_model(
        min_topic_size=100,
        min_cluster_size=100,
        n_neighbors=15,            # tighter local density, more separation
        n_components=5,
        embedding_model='all-MiniLM-L6-v2',
        nr_topics='auto',
        reduce_outliers=True
    )
    
    # Optional: reduce topics if too many were discovered
    # analyzer.reduce_topics(nr_topics=20)
    
    # Analysis
    analyzer.print_detailed_topics(n_topics=10)
    analyzer.save_detailed_topics(n_topics=10)
    analyzer.visualize_topics()
    analyzer.evaluate_topics()
    analyzer.plot_topic_distribution()
    
    # Interactive visualizations (saved as HTML)
    analyzer.visualize_topic_hierarchy()
    analyzer.visualize_topic_space()
    
    # Save
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("USAGE: python bertopic_analyzer.py <FORUM> [MAX_POSTS] [MIN_TOPIC_SIZE] [NR_TOPICS]")
        print("\nParameters:")
        print("  FORUM: 'lw' or 'af' (required)")
        print("  MAX_POSTS: Limit posts for testing (default: None = all)")
        print("  MIN_TOPIC_SIZE: Minimum documents per topic (default: 400)")
        print("  NR_TOPICS: Target number of topics (default: 25)")
        print("\nExamples:")
        print("  python bertopic_analyzer.py lw None 400 25  # 400 docs/topic, reduce to 25")
        print("  python bertopic_analyzer.py af None 500 20  # 500 docs/topic, reduce to 20")
        print("  python bertopic_analyzer.py lw 5000 100 15  # Test with 5k posts")
        print("\n⚠️  If you still get too many topics:")
        print("  • SET nr_topics lower (15-20)")
        print("  • INCREASE min_topic_size (500-1000)")
        print("  • Edit code: increase n_neighbors to 30-50")
        print("\nNote: First run will download the embedding model (~80MB)")
        sys.exit(1)
    
    platform = sys.argv[1]
    max_posts = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'None' else None
    min_topic_size = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    nr_topics = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    
    main(platform, max_posts)