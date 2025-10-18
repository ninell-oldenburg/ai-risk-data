import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import HdpModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from collections import Counter
from pathlib import Path
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class OptimizedHierarchicalLDA:
    """
    Hierarchical Dirichlet Process (HDP) with hyperparameter tuning
    
    HDP automatically discovers the number of topics - no need to specify!
    We test different hyperparameters to find the best model.
    """
    
    def __init__(self, platform):
        try:
            if platform in ['lw', 'af']:
                self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        except ValueError:
            print("FORUM variable has to be 'lw' or 'af'")
        self.base_path = f"src/processed_data/{self.platform}/02_with_links_and_gender"
        self.blog_posts = []
        self.hdp_results = {}
        
    def load_csv_files(self, start_year=2015, end_year=2024, max_posts=None):
        """Load all CSV files"""
        print(f"Loading CSV files from {start_year} to {end_year}...")
        
        all_posts = []
        file_count = 0
        
        for year in range(start_year, end_year + 1):
            year_path = Path(self.base_path) / str(year)
            
            if not year_path.exists():
                print(f"Warning: Directory {year_path} does not exist")
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
                        
                            if text_content.strip():
                                all_posts.append({
                                    '_id': str(row['_id']),
                                    'text': text_content.strip(),
                                    'title': str(row['title']).strip() if 'title' in row and pd.notna(row['title']) else "",
                                    'year': year,
                                    'month': month,
                                    'file': str(csv_path)
                                })
                        
                        file_count += 1
                        print(f"Loaded {len(df)} posts from {csv_path}")
                        
                        if max_posts and len(all_posts) >= max_posts:
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
    
    def get_wordnet_pos(self, treebank_tag):
        """Convert treebank POS tags to WordNet POS tags"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess_text(self, text):
        """Preprocess text with lemmatization"""
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split()
        pos_tags = nltk.pos_tag(tokens)
        
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [
            lemmatizer.lemmatize(token, self.get_wordnet_pos(pos)) 
            for token, pos in pos_tags
            if token not in ENGLISH_STOP_WORDS and len(token) > 2
        ]
        
        return lemmatized_tokens
    
    def prepare_corpus(self, min_df=3, max_df=0.9, keep_n=3000):
        """Prepare corpus for HDP"""
        print("Preparing corpus...")
        
        # Preprocess all texts
        texts = [self.preprocess_text(post['text']) for post in self.blog_posts]
        
        # Remove empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if len(text) >= 10:  # At least 10 words
                valid_texts.append(text)
                valid_indices.append(i)
        
        print(f"Using {len(valid_texts)} posts with sufficient content")
        
        # Create dictionary
        self.dictionary = Dictionary(valid_texts)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=keep_n)
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in valid_texts]
        self.processed_texts = valid_texts
        
        # Update blog_posts
        self.blog_posts = [self.blog_posts[i] for i in valid_indices]
        
        print(f"Dictionary size: {len(self.dictionary)}")
        print(f"Corpus size: {len(self.corpus)}")
        
        return self.corpus
    
    def train_single_hdp(self, max_topics=50, gamma=1.0, alpha=1.0):
        """
        Train a single HDP model with specific hyperparameters
        
        Key hyperparameters:
        - max_topics (K): Upper bound on topics (default: 50)
        - gamma: Concentration parameter for topic distribution (higher = more topics)
        - alpha: Concentration parameter for document-topic distribution
        """
        print(f"Training HDP (K={max_topics}, gamma={gamma}, alpha={alpha})...")
        
        if not hasattr(self, 'corpus'):
            self.prepare_corpus()
        
        # Train HDP model
        hdp_model = HdpModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            max_chunks=None,
            max_time=None,
            chunksize=256,
            kappa=1.0,
            tau=64.0,
            K=max_topics,  # Maximum number of topics
            T=150,  # Top level truncation
            alpha=alpha,
            gamma=gamma,
            eta=0.01,
            scale=1.0,
            var_converge=0.0001,
            random_state=42
        )
        
        # Get topics - use a more lenient threshold
        # show_topics returns ALL topics, even very weak ones
        all_topics = hdp_model.show_topics(num_topics=-1, num_words=20, formatted=False)
        
        # Count topics that actually have documents assigned
        doc_topics = [hdp_model[doc] for doc in self.corpus[:100]]  # Sample for speed
        topic_counts = {}
        for doc_topic_dist in doc_topics:
            for topic_id, prob in doc_topic_dist:
                if prob > 0.01:  # At least 1% probability
                    topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        
        n_topics = len(topic_counts)
        print(f"  → Discovered {n_topics} active topics (with >1% prob in docs)")
        
        # Calculate coherence
        cm = CoherenceModel(
            model=hdp_model,
            texts=self.processed_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
        print(f"  → Coherence: {coherence:.4f}")
        
        return hdp_model, n_topics, coherence
    
    def find_optimal_hdp(self, 
                        max_topics_range=[50, 75, 100, 150],
                        gamma_range=[1.0, 2.0, 5.0],
                        alpha_range=[1.0, 2.0]):
        """
        Test different hyperparameter combinations to find best model
        
        HDP discovers topics automatically, but hyperparameters affect:
        - How many topics are discovered
        - Topic quality (coherence)
        - Topic granularity
        
        Important: HDP needs higher values than you might expect!
        - gamma should be >= 1.0 (often 1-10)
        - K (max_topics) should be generous (50-150)
        """
        print(f"\n{'='*60}")
        print(f"FINDING OPTIMAL HDP MODEL")
        print(f"{'='*60}")
        print(f"Testing {len(max_topics_range) * len(gamma_range) * len(alpha_range)} configurations...")
        
        results = []
        best_coherence = -1
        best_model = None
        best_params = None
        
        for max_topics in max_topics_range:
            for gamma in gamma_range:
                for alpha in alpha_range:
                    print(f"\n{'─'*60}")
                    
                    try:
                        hdp_model, n_topics, coherence = self.train_single_hdp(
                            max_topics=max_topics,
                            gamma=gamma,
                            alpha=alpha
                        )
                        
                        results.append({
                            'max_topics': max_topics,
                            'gamma': gamma,
                            'alpha': alpha,
                            'n_topics_discovered': n_topics,
                            'coherence': coherence
                        })
                        
                        # Track best model (must have at least 5 topics)
                        if coherence > best_coherence and n_topics >= 5:
                            best_coherence = coherence
                            best_model = hdp_model
                            best_params = {
                                'max_topics': max_topics,
                                'gamma': gamma,
                                'alpha': alpha,
                                'n_topics': n_topics
                            }
                        
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
                        continue
        
        if len(results) == 0:
            print("No valid results!")
            return None
        
        if best_model is None:
            print("\n⚠️  WARNING: No models with >= 5 topics found!")
            print("Using model with highest coherence regardless...")
            df_results = pd.DataFrame(results)
            best_idx = df_results['coherence'].idxmax()
            best_row = df_results.iloc[best_idx]
            best_params = {
                'max_topics': int(best_row['max_topics']),
                'gamma': best_row['gamma'],
                'alpha': best_row['alpha'],
                'n_topics': int(best_row['n_topics_discovered']),
                'coherence': best_row['coherence']
            }
            best_coherence = best_row['coherence']
            # Retrain with these params
            best_model, _, _ = self.train_single_hdp(
                max_topics=best_params['max_topics'],
                gamma=best_params['gamma'],
                alpha=best_params['alpha']
            )
        
        df_results = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print(f"OPTIMAL HDP MODEL FOUND")
        print(f"{'='*60}")
        print(f"Max topics (K): {best_params['max_topics']}")
        print(f"Gamma: {best_params['gamma']}")
        print(f"Alpha: {best_params['alpha']}")
        print(f"Topics discovered: {best_params['n_topics']}")
        print(f"Coherence: {best_coherence:.4f}")
        
        # Visualize results
        self._visualize_hyperparameter_search(df_results, best_params)
        
        # Save results
        output_path = f"src/metadata/optimization/{self.platform}_hdp_optimization.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        return best_model, best_params, df_results
    
    def _visualize_hyperparameter_search(self, df_results, best_params):
        """Visualize hyperparameter search results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Topics discovered vs max_topics
        ax1 = axes[0, 0]
        for gamma in df_results['gamma'].unique():
            subset = df_results[df_results['gamma'] == gamma]
            ax1.plot(subset['max_topics'], subset['n_topics_discovered'], 
                    'o-', label=f'γ={gamma}', linewidth=2, markersize=8)
        ax1.axhline(y=best_params['n_topics'], color='r', linestyle='--', 
                   label=f"Best: {best_params['n_topics']} topics")
        ax1.set_xlabel('Max Topics (K)', fontsize=12)
        ax1.set_ylabel('Topics Discovered', fontsize=12)
        ax1.set_title('Topics Discovered vs K', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Coherence vs topics discovered
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df_results['n_topics_discovered'], 
                             df_results['coherence'],
                             c=df_results['gamma'], 
                             s=100, alpha=0.6, cmap='viridis')
        ax2.scatter(best_params['n_topics'], best_params['coherence'], 
                   color='red', s=200, marker='*', 
                   label='Best Model', edgecolors='black', linewidths=2)
        plt.colorbar(scatter, ax=ax2, label='γ (gamma)')
        ax2.set_xlabel('Topics Discovered', fontsize=12)
        ax2.set_ylabel('Coherence Score', fontsize=12)
        ax2.set_title('Coherence vs Number of Topics', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Coherence by gamma
        ax3 = axes[1, 0]
        gamma_coherence = df_results.groupby('gamma')['coherence'].mean()
        ax3.bar(range(len(gamma_coherence)), gamma_coherence.values)
        ax3.set_xticks(range(len(gamma_coherence)))
        ax3.set_xticklabels([f'{g:.1f}' for g in gamma_coherence.index])
        ax3.set_xlabel('Gamma (γ)', fontsize=12)
        ax3.set_ylabel('Average Coherence', fontsize=12)
        ax3.set_title('Effect of Gamma on Coherence', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary table
        ax4 = axes[1, 1]
        summary_text = f"""HDP Hyperparameter Optimization

Best Model:
  Max Topics (K): {best_params['max_topics']}
  Gamma (γ): {best_params['gamma']}
  Alpha (α): {best_params['alpha']}
  
Results:
  Topics Discovered: {best_params['n_topics']}
  Coherence: {best_params['coherence']:.4f}

Top 3 Configurations by Coherence:
"""
        top3 = df_results.nlargest(3, 'coherence')
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            summary_text += f"\n{i}. K={int(row['max_topics'])}, γ={row['gamma']}, α={row['alpha']}"
            summary_text += f"\n   → {int(row['n_topics_discovered'])} topics, coherence={row['coherence']:.4f}"
        
        summary_text += f"\n\nInterpretation:"
        summary_text += f"\n• Higher γ → more topics"
        summary_text += f"\n• Higher K → allows more topics"
        summary_text += f"\n• Optimal balance at {best_params['n_topics']} topics"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax4.axis('off')
        
        plt.tight_layout()
        
        output_path = f"src/metadata/img/{self.platform}/hdp_optimization.pdf"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.show()
    
    def train_final_hdp(self, max_topics=50, gamma=1.0, alpha=1.0):
        """Train final HDP model with chosen hyperparameters"""
        print(f"\n{'='*60}")
        print(f"TRAINING FINAL HDP MODEL")
        print(f"{'='*60}")
        
        if not hasattr(self, 'corpus'):
            self.prepare_corpus()
        
        # Train model
        self.hdp_model = HdpModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            max_chunks=None,
            max_time=None,
            chunksize=256,
            kappa=1.0,
            tau=64.0,
            K=max_topics,
            T=150,
            alpha=alpha,
            gamma=gamma,
            eta=0.01,
            scale=1.0,
            var_converge=0.0001,
            random_state=42
        )
        
        # Get topics
        all_topics = self.hdp_model.show_topics(num_topics=-1, num_words=20, formatted=False)
        
        # Count actual active topics by checking document assignments
        print("Counting active topics...")
        topic_counts = {}
        for doc in self.corpus:
            doc_topic_dist = self.hdp_model[doc]
            for topic_id, prob in doc_topic_dist:
                if prob > 0.01:  # At least 1% probability
                    topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        
        active_topic_ids = [tid for tid, count in topic_counts.items() if count >= 5]  # At least 5 docs
        
        print(f"Discovered {len(active_topic_ids)} active topics (used in >= 5 documents)")
        
        # Get document-topic distributions
        doc_topic_dist = []
        for doc in self.corpus:
            topic_dist = self.hdp_model[doc]
            # Create dense array for ALL topics (not just active ones)
            dense_dist = np.zeros(len(all_topics))
            for topic_id, prob in topic_dist:
                if topic_id < len(all_topics):
                    dense_dist[topic_id] = prob
            doc_topic_dist.append(dense_dist)
        
        doc_topic_dist = np.array(doc_topic_dist)
        topic_assignments = np.argmax(doc_topic_dist, axis=1)
        
        # Add to blog posts
        for i, post in enumerate(self.blog_posts):
            post['hdp_topic'] = topic_assignments[i]
            post['topic_probabilities'] = doc_topic_dist[i]
        
        # Calculate topic statistics
        topic_stats = {}
        for topic_id in range(len(topics)):
            topic_posts = [post for post in self.blog_posts if post['hdp_topic'] == topic_id]
            
            if len(topic_posts) == 0:
                continue
            
            year_dist = Counter([post['year'] for post in topic_posts])
            avg_prob = np.mean([post['topic_probabilities'][topic_id] for post in topic_posts])
            
            topic_words = self.hdp_model.show_topic(topic_id, topn=20)
            
            topic_stats[topic_id] = {
                'size': len(topic_posts),
                'percentage': len(topic_posts) / len(self.blog_posts) * 100,
                'average_probability': avg_prob,
                'year_distribution': dict(year_dist),
                'top_words': topic_words
            }
        
        # Calculate coherence
        cm = CoherenceModel(
            model=self.hdp_model,
            texts=self.processed_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
        
        self.hdp_results = {
            'model': self.hdp_model,
            'n_topics': len(active_topics),
            'doc_topic_dist': doc_topic_dist,
            'topic_assignments': topic_assignments,
            'topic_stats': topic_stats,
            'coherence': coherence,
            'hyperparameters': {
                'max_topics': max_topics,
                'gamma': gamma,
                'alpha': alpha
            }
        }
        
        print(f"Coherence score: {coherence:.4f}")
        
        return self.hdp_results
    
    def visualize_hdp(self):
        """Visualize HDP results"""
        if not self.hdp_results:
            print("No results available. Train model first.")
            return
        
        print("Creating visualizations...")
        
        active_topics = {tid: stats for tid, stats in self.hdp_results['topic_stats'].items() 
                        if stats['size'] > 0}
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Topic sizes
        ax1 = axes[0, 0]
        topic_ids = list(active_topics.keys())
        sizes = [active_topics[tid]['size'] for tid in topic_ids]
        ax1.bar(range(len(sizes)), sizes)
        ax1.set_xlabel('Topic ID')
        ax1.set_ylabel('Number of Posts')
        ax1.set_title(f'Topic Sizes (HDP - {len(active_topics)} topics)')
        if len(topic_ids) <= 30:
            ax1.set_xticks(range(len(topic_ids)))
            ax1.set_xticklabels(topic_ids, rotation=45)
        
        # Topic probabilities
        ax2 = axes[0, 1]
        avg_probs = [active_topics[tid]['average_probability'] for tid in topic_ids]
        ax2.bar(range(len(avg_probs)), avg_probs)
        ax2.set_xlabel('Topic ID')
        ax2.set_ylabel('Average Probability')
        ax2.set_title('Average Topic Probabilities')
        if len(topic_ids) <= 30:
            ax2.set_xticks(range(len(topic_ids)))
            ax2.set_xticklabels(topic_ids, rotation=45)
        
        # Year distribution
        ax3 = axes[0, 2]
        years = sorted(set(post['year'] for post in self.blog_posts))
        topic_year_matrix = np.zeros((len(topic_ids), len(years)))
        
        for i, tid in enumerate(topic_ids):
            year_dist = active_topics[tid]['year_distribution']
            for j, year in enumerate(years):
                topic_year_matrix[i, j] = year_dist.get(year, 0)
        
        im = ax3.imshow(topic_year_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax3)
        ax3.set_xticks(range(len(years)))
        ax3.set_xticklabels(years, rotation=45)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Topic')
        ax3.set_title('Distribution by Year')
        
        # Model summary
        ax4 = axes[1, 0]
        params = self.hdp_results['hyperparameters']
        summary_text = f"""HDP Model Summary
        
Active Topics: {len(active_topics)}
Total Posts: {len(self.blog_posts)}
Coherence: {self.hdp_results['coherence']:.4f}

Hyperparameters:
  K (max topics): {params['max_topics']}
  γ (gamma): {params['gamma']}
  α (alpha): {params['alpha']}

Top 5 Largest Topics:
"""
        sorted_topics = sorted(active_topics.items(), 
                              key=lambda x: x[1]['size'], 
                              reverse=True)[:5]
        
        for tid, stats in sorted_topics:
            top_words = ', '.join([w[0] for w in stats['top_words'][:4]])
            summary_text += f"\n{tid} ({stats['size']}): {top_words}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.axis('off')
        
        # Topic size distribution
        ax5 = axes[1, 1]
        ax5.hist(sizes, bins=min(20, len(sizes)), edgecolor='black')
        ax5.set_xlabel('Topic Size')
        ax5.set_ylabel('Number of Topics')
        ax5.set_title('Topic Size Distribution')
        
        # Probability vs Size
        ax6 = axes[1, 2]
        ax6.scatter(avg_probs, sizes, alpha=0.6, s=100)
        ax6.set_xlabel('Average Probability')
        ax6.set_ylabel('Topic Size')
        ax6.set_title('Probability vs Size')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = f"src/metadata/img/{self.platform}/hdp_results.pdf"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.show()
    
    def print_summary(self):
        """Print summary"""
        if not self.hdp_results:
            return
        
        active_topics = {tid: stats for tid, stats in self.hdp_results['topic_stats'].items() 
                        if stats['size'] > 0}
        
        print(f"\n{'='*60}")
        print(f"HDP SUMMARY")
        print(f"{'='*60}")
        print(f"Topics: {len(active_topics)}")
        print(f"Total: {len(self.blog_posts)}")
        print(f"Coherence: {self.hdp_results['coherence']:.4f}")
        
        sorted_topics = sorted(active_topics.items(), 
                              key=lambda x: x[1]['size'], 
                              reverse=True)
        
        for topic_id, stats in sorted_topics:
            print(f"\nTopic {topic_id} ({stats['size']} posts, {stats['percentage']:.1f}%):")
            words = [f"{w[0]}({w[1]:.3f})" for w in stats['top_words'][:8]]
            print(f"  {', '.join(words)}")
    
    def save_results(self, output_file='hdp_results.csv'):
        """Save results"""
        if not self.hdp_results:
            return
        
        results_data = []
        for post in self.blog_posts:
            topic_probs = {f'topic_{j}_prob': post['topic_probabilities'][j] 
                          for j in range(len(post['topic_probabilities']))}
            
            row_data = {
                '_id': post['_id'],
                'title': post['title'],
                'year': post['year'],
                'month': post['month'],
                'dominant_topic': post['hdp_topic'],
                'file': post['file']
            }
            row_data.update(topic_probs)
            results_data.append(row_data)
        
        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Save model
        model_path = output_file.replace('.csv', '_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.hdp_model, f)
        print(f"Model saved to {model_path}")


def main(platform, max_posts=None, optimize=True, max_topics=50, gamma=1.0, alpha=1.0):
    """
    Main function
    
    Args:
        platform: 'lw' or 'af'
        max_posts: Limit posts
        optimize: If True, search for best hyperparameters
        max_topics, gamma, alpha: Used if optimize=False
    """
    print("\n" + "="*60)
    print("HIERARCHICAL LDA (HDP)")
    print("="*60)
    
    analyzer = OptimizedHierarchicalLDA(platform)
    
    if not analyzer.load_csv_files(max_posts=max_posts):
        return
    
    analyzer.prepare_corpus()
    
    if optimize:
        print("\nSearching for optimal hyperparameters...")
        best_model, best_params, results = analyzer.find_optimal_hdp(
            max_topics_range=[30, 50, 70],
            gamma_range=[0.5, 1.0, 1.5],
            alpha_range=[0.5, 1.0]
        )
        
        # Train final model with best params
        analyzer.train_final_hdp(
            max_topics=best_params['max_topics'],
            gamma=best_params['gamma'],
            alpha=best_params['alpha']
        )
    else:
        analyzer.train_final_hdp(max_topics=max_topics, gamma=gamma, alpha=alpha)
    
    analyzer.print_summary()
    analyzer.visualize_hdp()
    
    output_path = f'src/metadata/clustering_results/{analyzer.platform}/hdp_results.csv'
    analyzer.save_results(output_path)
    
    print("\nComplete!")

one two three

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("USAGE: python hdp_optimized.py <FORUM> [MAX_POSTS] [OPTIMIZE]")
        print("\nParams:")
        print("  FORUM: 'lw' or 'af' (required)")
        print("  MAX_POSTS: Limit posts (default: None)")
        print("  OPTIMIZE: True/False - search for best params (default: True)")
        print("\nExamples:")
        print("  python hdp_optimized.py lw 5000 True   # Test with optimization")
        print("  python hdp_optimized.py af None True   # Full data with optimization")
        print("  python hdp_optimized.py lw None False  # Use default params")
        sys.exit(1)
    
    platform = sys.argv[1]
    max_posts = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'None' else None
    optimize = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True
    
    main(platform, max_posts, optimize)