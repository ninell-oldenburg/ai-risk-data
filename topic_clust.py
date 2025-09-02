import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from collections import defaultdict, Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BlogTopicClustering:
    def __init__(self, base_path="lw_csv"):
        self.base_path = base_path
        self.blog_posts = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.cluster_results = {}
        
    def load_csv_files(self, start_year=2016, end_year=2025):
        """Load all CSV files from the specified year range"""
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
                        
                        # Clean column names (remove whitespace)
                        df.columns = df.columns.str.strip()
            
                        for _, row in df.iterrows():
                            text_content = ""
                            if pd.notna(row['title']):
                                text_content += str(row['title']) + " "
                            if pd.notna(row['cleaned_htmlBody']):
                                text_content += str(row['cleaned_htmlBody'])
                            
                            if text_content.strip():
                                all_posts.append({
                                    'text': text_content.strip(),
                                    'title': str(row['title']).strip() if 'title' in row and pd.notna(row['title']) else "",
                                    'year': year,
                                    'month': month,
                                    'file': str(csv_path)
                                })
                        
                        file_count += 1
                        print(f"Loaded {len(df)} posts from {csv_path}")
                            
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")
        
        self.blog_posts = all_posts
        print(f"\nTotal: Loaded {len(all_posts)} blog posts from {file_count} files")
        
        if len(all_posts) == 0:
            print("No blog posts loaded! Please check your file structure and column names.")
            return False
        
        return True
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_tfidf_matrix(self, max_features=5000, min_df=2, max_df=0.95):
        """Create TF-IDF matrix from blog posts"""
        print("Creating TF-IDF matrix...")
        
        # Preprocess all texts
        texts = [self.preprocess_text(post['text']) for post in self.blog_posts]
        
        # Remove empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if len(text.split()) >= 5:  # At least 5 words
                valid_texts.append(text)
                valid_indices.append(i)
        
        print(f"Using {len(valid_texts)} posts with sufficient text content")

        extra_stopwords = set([
            'people', 'like', 'think', 'just', 'don', 'time', 'way', 'good', 'want', 'new', 'thing', 'things',
            'make', 'know', 'https', 'com', 'plus', 'pm',
        ])
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words=list(ENGLISH_STOP_WORDS.union(extra_stopwords)),
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
        
        # Update blog_posts to only include valid ones
        self.blog_posts = [self.blog_posts[i] for i in valid_indices]
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return self.tfidf_matrix
    
    def elbow_test(self, k_range=range(2, 21), n_samples=None):
        """Perform elbow test to find optimal number of clusters"""
        print("Performing elbow test...")
        
        # Use subset for efficiency if dataset is large
        if n_samples and len(self.blog_posts) > n_samples:
            indices = np.random.choice(len(self.blog_posts), n_samples, replace=False)
            X = self.tfidf_matrix[indices]
        else:
            X = self.tfidf_matrix
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score (skip for k=1)
            if k > 1:
                sil_score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Find elbow using the elbow method
        # Calculate the rate of change of inertias
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diff_ratios = np.diff(diffs)
            optimal_k_idx = np.argmax(diff_ratios) + 2  # +2 because we start from k=2 and take second diff
            optimal_k = k_range[optimal_k_idx] if optimal_k_idx < len(k_range) else k_range[len(k_range)//2]
        else:
            optimal_k = k_range[len(k_range)//2]  # Default to middle value
        
        # Also find k with best silhouette score
        best_sil_k = k_range[np.argmax(silhouette_scores)] if silhouette_scores else optimal_k
        
        # Plot elbow curve
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(k_range, inertias, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Elbow at k={optimal_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.title('Elbow Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(k_range, silhouette_scores, 'go-')
        plt.axvline(x=best_sil_k, color='r', linestyle='--', label=f'Best silhouette at k={best_sil_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Combined plot
        plt.subplot(1, 3, 3)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(k_range, inertias, 'bo-', label='Inertia')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        line2 = ax2.plot(k_range, silhouette_scores, 'go-', label='Silhouette')
        ax2.set_ylabel('Silhouette Score', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        plt.title('Combined Metrics')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Suggested optimal k (elbow method): {optimal_k}")
        print(f"Best k (silhouette score): {best_sil_k}")
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': optimal_k,
            'optimal_k_silhouette': best_sil_k
        }
    
    def perform_clustering(self, n_clusters):
        """Perform K-means clustering"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.tfidf_matrix)
        
        # Add cluster labels to blog posts
        for i, post in enumerate(self.blog_posts):
            post['cluster'] = cluster_labels[i]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top terms for each cluster
        cluster_terms = {}
        for cluster_id in range(n_clusters):
            # Get centroid for this cluster
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Get top terms (highest TF-IDF scores)
            top_indices = centroid.argsort()[-20:][::-1]  # Top 20 terms
            top_terms = [(feature_names[i], centroid[i]) for i in top_indices]
            
            cluster_terms[cluster_id] = top_terms
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_posts = [post for post in self.blog_posts if post['cluster'] == cluster_id]
            
            # Year distribution
            year_dist = Counter([post['year'] for post in cluster_posts])
            
            cluster_stats[cluster_id] = {
                'size': len(cluster_posts),
                'percentage': len(cluster_posts) / len(self.blog_posts) * 100,
                'year_distribution': dict(year_dist),
                'top_terms': cluster_terms[cluster_id]
            }
        
        self.cluster_results = {
            'n_clusters': n_clusters,
            'labels': cluster_labels,
            'kmeans': kmeans,
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette_score(self.tfidf_matrix, cluster_labels)
        }
        
        return self.cluster_results
    
    def visualize_clusters(self, n_components=2):
        """Visualize clusters using PCA"""
        if not self.cluster_results:
            print("No clustering results available. Run perform_clustering() first.")
            return
        
        print("Creating cluster visualization...")
        
        # Perform PCA for visualization
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.tfidf_matrix.toarray())
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': self.cluster_results['labels'],
            'Year': [post['year'] for post in self.blog_posts]
        })
        
        plt.figure(figsize=(15, 10))
        
        # Main cluster plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(plot_df['PC1'], plot_df['PC2'], 
                            c=plot_df['Cluster'], cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Blog Posts Clusters (PCA Visualization)')
        
        # Cluster sizes
        plt.subplot(2, 2, 2)
        cluster_sizes = [self.cluster_results['cluster_stats'][i]['size'] 
                        for i in range(self.cluster_results['n_clusters'])]
        plt.bar(range(len(cluster_sizes)), cluster_sizes)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Posts')
        plt.title('Cluster Sizes')
        
        # Year distribution across clusters
        plt.subplot(2, 2, 3)
        years = sorted(plot_df['Year'].unique())
        cluster_year_matrix = np.zeros((self.cluster_results['n_clusters'], len(years)))
        
        for i, cluster_id in enumerate(range(self.cluster_results['n_clusters'])):
            year_dist = self.cluster_results['cluster_stats'][cluster_id]['year_distribution']
            for j, year in enumerate(years):
                cluster_year_matrix[i, j] = year_dist.get(year, 0)
        
        im = plt.imshow(cluster_year_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im)
        plt.yticks(range(self.cluster_results['n_clusters']), 
                  [f'Cluster {i}' for i in range(self.cluster_results['n_clusters'])])
        plt.xticks(range(len(years)), years, rotation=45)
        plt.xlabel('Year')
        plt.ylabel('Cluster')
        plt.title('Posts Distribution by Year and Cluster')
        
        # Silhouette score info
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"Number of Clusters: {self.cluster_results['n_clusters']}", 
                fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f"Silhouette Score: {self.cluster_results['silhouette_score']:.3f}", 
                fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f"Total Posts: {len(self.blog_posts)}", 
                fontsize=12, transform=plt.gca().transAxes)
        
        # Show top terms for each cluster
        y_pos = 0.4
        for cluster_id in range(min(3, self.cluster_results['n_clusters'])):  # Show top 3 clusters
            top_terms = [term[0] for term in self.cluster_results['cluster_stats'][cluster_id]['top_terms'][:5]]
            plt.text(0.1, y_pos, f"Cluster {cluster_id}: {', '.join(top_terms)}", 
                    fontsize=10, transform=plt.gca().transAxes)
            y_pos -= 0.08
        
        plt.axis('off')
        plt.title('Clustering Summary')
        
        plt.tight_layout()
        plt.show()
    
    def print_cluster_summary(self):
        """Print detailed cluster summary"""
        if not self.cluster_results:
            print("No clustering results available. Run perform_clustering() first.")
            return
        
        print(f"\n{'='*50}")
        print(f"CLUSTER ANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Number of clusters: {self.cluster_results['n_clusters']}")
        print(f"Total posts analyzed: {len(self.blog_posts)}")
        print(f"Silhouette score: {self.cluster_results['silhouette_score']:.3f}")
        
        for cluster_id in range(self.cluster_results['n_clusters']):
            stats = self.cluster_results['cluster_stats'][cluster_id]
            print(f"\n{'-'*40}")
            print(f"CLUSTER {cluster_id}")
            print(f"{'-'*40}")
            print(f"Size: {stats['size']} posts ({stats['percentage']:.1f}%)")
            
            print(f"\nTop terms:")
            for i, (term, score) in enumerate(stats['top_terms'][:10], 1):
                print(f"  {i:2d}. {term} ({score:.3f})")
            
            print(f"\nYear distribution:")
            year_dist = stats['year_distribution']
            for year in sorted(year_dist.keys()):
                print(f"  {year}: {year_dist[year]} posts")
    
    def save_results(self, output_file='clustering_results.csv'):
        """Save clustering results to CSV"""
        if not self.cluster_results:
            print("No clustering results available. Run perform_clustering() first.")
            return

        # Prepare data for CSV
        results_data = []
        for post in self.blog_posts:
            results_data.append({
                'title': post['title'],
                'year': post['year'],
                'month': post['month'],
                'cluster': post['cluster'],
                'file': post['file']
            })

        df = pd.DataFrame(results_data)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
            
        # Save cluster summaries
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"CLUSTER ANALYSIS SUMMARY\n")
            f.write(f"{'='*50}\n")
            f.write(f"Number of clusters: {self.cluster_results['n_clusters']}\n")
            f.write(f"Total posts analyzed: {len(self.blog_posts)}\n")
            f.write(f"Silhouette score: {self.cluster_results['silhouette_score']:.3f}\n\n")
            
            for cluster_id in range(self.cluster_results['n_clusters']):
                stats = self.cluster_results['cluster_stats'][cluster_id]
                f.write(f"CLUSTER {cluster_id}\n")
                f.write(f"{'-'*40}\n")
                f.write(f"Size: {stats['size']} posts ({stats['percentage']:.1f}%)\n\n")
                
                f.write(f"Top terms:\n")
                for i, (term, score) in enumerate(stats['top_terms'][:15], 1):
                    f.write(f"  {i:2d}. {term} ({score:.3f})\n")
                
                f.write(f"\nYear distribution:\n")
                year_dist = stats['year_distribution']
                for year in sorted(year_dist.keys()):
                    f.write(f"  {year}: {year_dist[year]} posts\n")
                f.write(f"\n")
        
        print(f"Cluster summary saved to {summary_file}")

# Main execution
def main():
    # Initialize the clustering analysis
    analyzer = BlogTopicClustering(base_path="lw_csv")
    
    # Load CSV files
    success = analyzer.load_csv_files(start_year=2016, end_year=2025)
    if not success:
        return
    
    # Create TF-IDF matrix
    analyzer.create_tfidf_matrix(max_features=3000, min_df=3, max_df=0.9)
    
    # Perform elbow test
    #elbow_results = analyzer.elbow_test(k_range=range(3, 16), n_samples=2000)  # Sample for speed
    
    # Use the suggested optimal number of clusters
    # 'optimal_k_silhouette' or 'optimal_k_elbow'
    # optimal_k = elbow_results['optimal_k_silhouette']
    optimal_k = 10
    print(f"\nUsing k={optimal_k} clusters based on silhouette score")
    
    # Perform clustering
    results = analyzer.perform_clustering(n_clusters=optimal_k)
    
    # Print summary
    analyzer.print_cluster_summary()
    
    # Visualize results
    analyzer.visualize_clusters()
    
    # Save results
    analyzer.save_results('blog_clustering_results.csv')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()