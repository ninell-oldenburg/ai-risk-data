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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches

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

# Nature figure specifications
# - Single column: 89mm (3.5 inches)
# - Double column: 183mm (7.2 inches)  
# - Max height: 247mm (9.7 inches)
# - Resolution: 300-600 DPI
# - Fonts: Arial or Helvetica, min 5-7pt final size
# - Line weights: 0.5-1pt

# Set Nature style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.titlesize': 8,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
})

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
        top_words_per_topic = self.get_top_words_safe(topics, top_n_words=top_n_words)
        if not top_words_per_topic:
            return 0.0
        
        all_words = [w for t in top_words_per_topic for w in t]
        return (len(set(all_words)) / len(all_words)) if all_words else 0.0
    
    def get_top_words_safe(self, topics, top_n_words=10):
        top_words_per_topic = []
        for tid, ws in topics.items():
            if tid == -1 or not ws:
                continue
            words = []
            for item in ws:
                # Skip if item is not tuple/list
                if not isinstance(item, (tuple, list)):
                    continue
                # Take first element if string
                if len(item) > 0 and isinstance(item[0], str) and item[0].strip():
                    words.append(item[0].strip())
            if words:
                top_words_per_topic.append(words[:top_n_words])
        return top_words_per_topic

    def compute_coherence_from_docs(self, docs, top_n_words=10):
        analyzer = self.topic_model.vectorizer_model.build_analyzer()
        tokenized_docs = [analyzer(d) for d in docs]
        dictionary = Dictionary(tokenized_docs)
        top_words_per_topic = self.get_top_words_safe(self.topic_model.get_topics(), top_n_words=top_n_words)
        
        cm = CoherenceModel(
            topics=top_words_per_topic,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v"
        )
        return cm.get_coherence()

    def sweep_parameters(self,
                        n_neighbors_list=[10,15,25,50],
                        min_topic_size_list=[50,100,200],
                        n_components=5,
                        embedding_model=None,
                        top_n_words=10,
                        max_posts_for_sweep=None,  # Changed: use ALL by default
                        apply_auto_reduction=True,  # NEW: match main logic
                        output_dir="src/hidden/sweep_results"):
        """
        Sweep parameters to find optimal settings.
        Set apply_auto_reduction=True to match train_topic_model() behavior.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(output_dir) / f"sweep_{int(time.time())}.csv"
        
        model_name = embedding_model or "all-MiniLM-L6-v2"
        sentence_model = SentenceTransformer(model_name)

        # Use same data as main training (or a large representative subset)
        docs_all = [p["text"] for p in self.blog_posts]
        if max_posts_for_sweep and len(docs_all) > max_posts_for_sweep:
            print(f"⚠️  Using {max_posts_for_sweep} posts for sweep (out of {len(docs_all)})")
            print("   Results may differ from full corpus!")
            docs = docs_all[:max_posts_for_sweep]
        else:
            docs = docs_all

        for nn in n_neighbors_list:
            for mts in min_topic_size_list:
                print(f"\n--- SWEEP: n_neighbors={nn}, min_topic_size={mts} ---")
                
                # Build same components as main training
                umap_model = UMAP(
                    n_neighbors=nn, 
                    n_components=n_components, 
                    min_dist=0.0, 
                    metric="cosine", 
                    random_state=42
                )
                
                hdbscan_model = HDBSCAN(
                    min_cluster_size=mts, 
                    min_samples=max(10, mts//10),  # Match main training
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True,
                    cluster_selection_epsilon=0.0
                )

                vectorizer_model = CountVectorizer(
                    ngram_range=(1,2), 
                    stop_words='english', 
                    min_df=1,  # Match main training
                    max_df=1.0
                )
                
                keybert_model = KeyBERTInspired()
                mmr_model = MaximalMarginalRelevance(diversity=0.3)

                topic_model = BERTopic(
                    embedding_model=sentence_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    representation_model=[keybert_model, mmr_model],
                    top_n_words=top_n_words,
                    min_topic_size=mts,
                    nr_topics='auto',
                    calculate_probabilities=False,
                    verbose=False
                )

                # Fit model
                topics, probs = topic_model.fit_transform(docs)
                
                # Apply outlier reduction (MATCH MAIN TRAINING)
                new_topics = topic_model.reduce_outliers(docs, topics)
                topics = new_topics
                
                # Get initial topic count
                n_topics_initial = len(topic_model.get_topic_info()) - 1
                
                # Apply auto-reduction logic if enabled (MATCH MAIN TRAINING)
                if apply_auto_reduction and n_topics_initial > 50:
                    recommended_topics = max(15, min(40, len(docs) // mts))
                    target = min(30, recommended_topics)
                    print(f"  Auto-reducing {n_topics_initial} -> {target} topics")
                    topic_model.reduce_topics(docs, nr_topics=target)
                    topics = topic_model.topics_
                
                # Attach to self temporarily
                old_model = getattr(self, "topic_model", None)
                self.topic_model = topic_model
                self.docs = docs

                # Compute final metrics
                num_topics = len(topic_model.get_topic_info()) - 1
                num_outliers = sum(t == -1 for t in topics)
                diversity = self.compute_diversity_from_model(top_n_words=top_n_words)
                coherence = self.compute_coherence_from_docs(docs, top_n_words=top_n_words)

                print(f"-> topics: {num_topics}, outliers: {num_outliers}, "
                    f"coherence: {coherence:.3f}, diversity: {diversity:.3f}")

                # Write results
                header = ['n_neighbors', 'min_topic_size', 'n_topics', 'n_outliers', 
                        'coherence_c_v', 'diversity']
                if not csv_path.exists():
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([nn, mts, num_topics, num_outliers,
                                round(coherence, 4), round(diversity, 4)])

                # Restore
                if old_model is not None:
                    self.topic_model = old_model

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

        top_words_per_topic = self.get_top_words_safe(self.topic_model.get_topics(), top_n_words=10)
        
        all_words = [w for topic in top_words_per_topic for w in topic]
        unique_words = set(all_words)
        topic_diversity = len(unique_words) / len(all_words) if all_words else 0.0

        # coherence score (c_v)
        analyzer = self.topic_model.vectorizer_model.build_analyzer()
        tokenized_docs = [analyzer(d) for d in docs]
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
    EMBEDDING_MODEL = 'all-mpnet-base-v2'
    
    # Load data
    if not analyzer.load_csv_files(max_posts=max_posts):
        print("Failed to load data!")
        return
    
    """analyzer.sweep_parameters(
        n_neighbors_list=[10,15,25,50],
        min_topic_size_list=[50,100,200,400],
        embedding_model=EMBEDDING_MODEL,
        max_posts_for_sweep=None,
        output_dir="sweep_results_run1"
    )"""
    
    # Train model
    analyzer.train_topic_model(
        min_topic_size=400,
        min_cluster_size=400,
        n_neighbors=15,
        n_components=5,
        embedding_model=EMBEDDING_MODEL,
        nr_topics='auto',
        reduce_outliers=True
    )
    
    # Optional: reduce topics if too many were discovered
    # analyzer.reduce_topics(nr_topics=20)
    
    analyzer.print_detailed_topics(n_topics=10)
    analyzer.save_detailed_topics(n_topics=10)
    analyzer.evaluate_topics()
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