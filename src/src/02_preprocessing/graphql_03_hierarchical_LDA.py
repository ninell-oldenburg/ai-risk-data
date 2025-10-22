import tomotopy as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import nltk, re, os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class OptimizedHLDA:
    """
    True Hierarchical LDA using tomotopy.
    Discovers a topic tree instead of flat topics.
    """

    def __init__(self, platform):
        self.platform = 'lesswrong' if platform == 'lw' else 'alignment_forum'
        self.base_path = f"src/processed_data/{self.platform}/02_with_links_and_gender"
        self.blog_posts = []
        self.results = {}
        self.dictionary = None
        self.processed_texts = None

    # -------------------- Loading & preprocessing --------------------
    def load_csv_files(self, max_posts=None):
        all_posts = []
        years = sorted([
            int(name) for name in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, name)) and name.isdigit()
        ])
        for year in years:
            year_path = Path(self.base_path) / str(year)
            if not year_path.exists():
                continue
            for month in range(1, 13):
                csv_path = year_path / f"{year}-{month:02d}.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    text = (str(row.get("title", "")) + " " +
                            str(row.get("cleaned_htmlBody", ""))).strip()
                    if text:
                        all_posts.append(text)
                    if max_posts and len(all_posts) >= max_posts:
                        break
                if max_posts and len(all_posts) >= max_posts:
                    break
            if max_posts and len(all_posts) >= max_posts:
                break
        self.blog_posts = all_posts
        print(f"Loaded {len(self.blog_posts)} posts.")
        return True

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'): return wordnet.ADJ
        if tag.startswith('V'): return wordnet.VERB
        if tag.startswith('N'): return wordnet.NOUN
        if tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        pos_tags = nltk.pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(t, self.get_wordnet_pos(p))
            for t, p in pos_tags
            if t not in ENGLISH_STOP_WORDS and len(t) > 2
        ]

    def prepare_corpus(self, min_df=3, max_df=0.9, keep_n=3000):
        print("Preparing corpus...")
        texts = [self.preprocess_text(t) for t in self.blog_posts]
        texts = [t for t in texts if len(t) >= 10]
        dictionary = Dictionary(texts)
        dictionary.filter_extremes(no_below=min_df, no_above=max_df, keep_n=keep_n)
        self.dictionary, self.processed_texts = dictionary, texts
        print(f"Processed {len(texts)} documents with {len(dictionary)} unique terms.")
        return texts

    # -------------------- Modeling --------------------
    def train_hlda(self, depth=2, alpha=10.0, eta=0.1, gamma=1.0, iterations=500, 
                   burn_in=100, save_to_results=True):
        """
        Train a single hLDA model.
        depth: number of tree levels (e.g. 2–4)
        alpha, eta, gamma: model hyperparameters
        iterations: number of training iterations
        burn_in: number of burn-in iterations before collecting statistics
        save_to_results: whether to save this model to self.results
        """
        print(f"Training hLDA(depth={depth}, alpha={alpha}, eta={eta}, gamma={gamma})")
        mdl = tp.HLDAModel(depth=depth, alpha=alpha, eta=eta, gamma=gamma)

        for doc in self.processed_texts:
            mdl.add_doc(doc)

        mdl.burn_in = burn_in
        mdl.train(iterations)
        print(f"Log-likelihood: {mdl.ll_per_word:.4f}")

        # Extract topics by level
        topics_by_level = {level: [] for level in range(depth)}
        for topic_id in range(mdl.k):
            if mdl.is_live_topic(topic_id):
                lvl = mdl.level(topic_id)
                if lvl < depth:
                    words = mdl.get_topic_words(topic_id, top_n=15)
                    topics_by_level[lvl].append((topic_id, words))

        # Compute coherence using gensim
        # FIXED: Properly extract words from topics_by_level and filter empty topics
        top_words_all = []
        for level in sorted(topics_by_level.keys()):
            for topic_id, words in topics_by_level[level]:
                # Extract just the word strings (first element of each tuple)
                word_list = [word for word, prob in words if word]  # Filter empty strings
                # Only add if we have at least 2 words (coherence needs multiple words)
                if len(word_list) >= 2:
                    top_words_all.append(word_list)

        if len(top_words_all) > 0:
            try:
                cm = CoherenceModel(
                    topics=top_words_all,
                    texts=self.processed_texts,
                    dictionary=self.dictionary,
                    coherence='c_v'
                )
                coherence = cm.get_coherence()
            except (ValueError, ZeroDivisionError) as e:
                print(f"Warning: Coherence calculation failed: {e}")
                coherence = 0.0
        else:
            coherence = 0.0
            print("Warning: No valid topics found for coherence calculation!")
        
        print(f"Coherence: {coherence:.4f}")

        result = {
            "model": mdl,
            "topics_by_level": topics_by_level,
            "coherence": coherence,
            "ll_per_word": mdl.ll_per_word,
            "params": {"depth": depth, "alpha": alpha, "eta": eta, "gamma": gamma}
        }
        
        if save_to_results:
            self.results = result
            
        return mdl, coherence, result

    def find_best_hlda(self,
                       depths=[2, 3],
                       alphas=[5.0, 10.0],
                       gammas=[0.5, 1.0, 2.0],
                       eta=0.1,
                       iterations=500):
        """
        Grid search for best hyperparameters by coherence.
        Returns the best model and saves it to self.results.
        """
        results = []
        best_coh, best_result = -1, None
        
        total_runs = len(depths) * len(alphas) * len(gammas)
        run_count = 0
        
        for depth in depths:
            for alpha in alphas:
                for gamma in gammas:
                    run_count += 1
                    print(f"\n{'='*60}")
                    print(f"Run {run_count}/{total_runs}")
                    print(f"{'='*60}")
                    
                    # Train model but don't save to self.results yet
                    mdl, coh, result = self.train_hlda(
                        depth=depth, alpha=alpha, eta=eta, gamma=gamma,
                        iterations=iterations, save_to_results=False
                    )
                    
                    results.append({
                        "depth": depth, 
                        "alpha": alpha, 
                        "gamma": gamma,
                        "eta": eta,
                        "coherence": coh,
                        "ll_per_word": mdl.ll_per_word,
                        "num_topics": len([t for t in range(mdl.k) if mdl.is_live_topic(t)])
                    })
                    
                    # Track best model
                    if coh > best_coh:
                        best_coh = coh
                        best_result = result
                        print(f"★ New best coherence: {coh:.4f}")
        
        # Save best model to self.results
        if best_result:
            self.results = best_result
            
        df = pd.DataFrame(results).sort_values('coherence', ascending=False)
        print("\n" + "="*60)
        print("GRID SEARCH RESULTS (sorted by coherence)")
        print("="*60)
        print(df.to_string(index=False))
        print("\nBest hLDA params:", best_result['params'])
        
        return best_result['model'], best_result['params'], df

    # -------------------- Visualization --------------------
    def visualize_tree(self, top_n=10):
        """Print the hierarchical topic tree."""
        if not self.results:
            print("Train model first.")
            return
        mdl = self.results["model"]
        print("\n" + "="*60)
        print("HIERARCHICAL TOPIC TREE")
        print("="*60)
        
        # Build tree structure manually
        depth = self.results['params']['depth']
        
        # Get all live topics with their levels and parents
        topic_info = []
        for topic_id in range(mdl.k):
            if mdl.is_live_topic(topic_id):
                level = mdl.level(topic_id)
                parent = mdl.parent_topic(topic_id) if level > 0 else None
                words = mdl.get_topic_words(topic_id, top_n=top_n)
                topic_info.append({
                    'id': topic_id,
                    'level': level,
                    'parent': parent,
                    'words': words
                })
        
        # Sort by level then by id
        topic_info.sort(key=lambda x: (x['level'], x['id']))
        
        # Print tree
        for topic in topic_info:
            indent = "  " * topic['level']
            word_str = ", ".join([f"{w}" for w, p in topic['words'][:top_n]])
            parent_str = f" (parent: {topic['parent']})" if topic['parent'] is not None else ""
            print(f"{indent}├─ Topic {topic['id']} [L{topic['level']}]{parent_str}")
            print(f"{indent}   {word_str}")
            print()
    
    def print_topic_summary(self, top_n=10):
        """Print topics organized by hierarchy level."""
        if not self.results:
            print("Train model first.")
            return
            
        topics_by_level = self.results["topics_by_level"]
        mdl = self.results["model"]
        
        print("\n" + "="*60)
        print("TOPICS BY HIERARCHY LEVEL")
        print("="*60)
        
        for level in sorted(topics_by_level.keys()):
            print(f"\n--- Level {level} ---")
            for topic_id, words in topics_by_level[level]:
                word_str = ", ".join([f"{w}({p:.3f})" for w, p in words[:top_n]])
                print(f"Topic {topic_id}: {word_str}")
    
    def analyze_document_paths(self, num_docs=5):
        """Show example document paths through the topic hierarchy."""
        if not self.results:
            print("Train model first.")
            return
            
        mdl = self.results["model"]
        print("\n" + "="*60)
        print(f"SAMPLE DOCUMENT PATHS (first {num_docs} documents)")
        print("="*60)
        
        for doc_id in range(min(num_docs, len(mdl.docs))):
            doc = mdl.docs[doc_id]
            path = doc.path
            print(f"\nDocument {doc_id}:")
            for level, topic_id in enumerate(path):
                words = mdl.get_topic_words(topic_id, top_n=5)
                word_str = ", ".join([w for w, _ in words])
                print(f"  Level {level} → Topic {topic_id}: {word_str}")
    
    def plot_tree_matplotlib(self, top_n=5, figsize=(16, 10), save_path=None):
        """Create a visual tree plot using matplotlib."""
        if not self.results:
            print("Train model first.")
            return
            
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        
        mdl = self.results["model"]
        
        # Collect all live topics
        topics = []
        for topic_id in range(mdl.k):
            if mdl.is_live_topic(topic_id):
                level = mdl.level(topic_id)
                parent = mdl.parent_topic(topic_id) if level > 0 else None
                words = mdl.get_topic_words(topic_id, top_n=top_n)
                word_str = ", ".join([w for w, _ in words])
                topics.append({
                    'id': topic_id,
                    'level': level,
                    'parent': parent,
                    'label': f"T{topic_id}\n{word_str}"
                })
        
        if not topics:
            print("No live topics to visualize")
            return
        
        # Organize by level
        levels = {}
        for t in topics:
            if t['level'] not in levels:
                levels[t['level']] = []
            levels[t['level']].append(t)
        
        # Calculate positions
        fig, ax = plt.subplots(figsize=figsize)
        positions = {}
        max_level = max(levels.keys())
        
        for level in sorted(levels.keys()):
            n_topics = len(levels[level])
            y = max_level - level
            for i, topic in enumerate(levels[level]):
                x = (i + 1) / (n_topics + 1)
                positions[topic['id']] = (x, y)
        
        # Draw edges
        for topic in topics:
            if topic['parent'] is not None and topic['parent'] in positions:
                x1, y1 = positions[topic['parent']]
                x2, y2 = positions[topic['id']]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1.5)
        
        # Draw nodes
        colors = plt.cm.Set3(np.linspace(0, 1, max_level + 1))
        for topic in topics:
            x, y = positions[topic['id']]
            color = colors[topic['level']]
            
            # Draw box
            box = FancyBboxPatch((x - 0.08, y - 0.15), 0.16, 0.3,
                                boxstyle="round,pad=0.01",
                                edgecolor='black', facecolor=color,
                                linewidth=2, alpha=0.8)
            ax.add_patch(box)
            
            # Add text
            ax.text(x, y, topic['label'], ha='center', va='center',
                   fontsize=8, wrap=True, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, max_level + 0.5)
        ax.axis('off')
        ax.set_title('Hierarchical Topic Tree', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='black',
                                         label=f'Level {i}')
                          for i in range(max_level + 1)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree plot saved to {save_path}")
        
        #plt.show()
        return fig
    
    def create_interactive_tree_html(self, top_n=5, save_path='hlda_tree.html'):
        """Create an interactive D3.js tree visualization."""
        if not self.results:
            print("Train model first.")
            return
            
        mdl = self.results["model"]
        
        # Build tree data structure
        nodes = []
        for topic_id in range(mdl.k):
            if mdl.is_live_topic(topic_id):
                level = mdl.level(topic_id)
                parent = mdl.parent_topic(topic_id) if level > 0 else None
                words = mdl.get_topic_words(topic_id, top_n=top_n)
                word_str = ", ".join([w for w, _ in words])
                
                nodes.append({
                    'id': topic_id,
                    'parent': parent if parent is not None else '',
                    'level': level,
                    'words': word_str
                })
        
        if not nodes:
            print("No live topics to visualize")
            return
        
        # Convert to hierarchical structure
        import json
        
        def build_tree(nodes, parent_id=''):
            children = [n for n in nodes if n['parent'] == parent_id]
            result = []
            for child in children:
                node = {
                    'name': f"Topic {child['id']}",
                    'words': child['words'],
                    'level': child['level']
                }
                child_nodes = build_tree(nodes, child['id'])
                if child_nodes:
                    node['children'] = child_nodes
                result.append(node)
            return result
        
        tree_data = {
            'name': 'Root',
            'children': build_tree(nodes)
        }
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Hierarchical LDA Topic Tree</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .node circle {{
            fill: #fff;
            stroke: steelblue;
            stroke-width: 3px;
            cursor: pointer;
        }}
        .node text {{
            font: 12px sans-serif;
        }}
        .link {{
            fill: none;
            stroke: #ccc;
            stroke-width: 2px;
        }}
        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 10px;
            font: 12px sans-serif;
            background: lightsteelblue;
            border: 1px solid #999;
            border-radius: 5px;
            pointer-events: none;
            opacity: 0;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
    </style>
</head>
<body>
    <h1>Hierarchical LDA Topic Tree</h1>
    <div id="tree"></div>
    <div class="tooltip"></div>
    
    <script>
        const data = {json.dumps(tree_data)};
        
        const width = 1200;
        const height = 800;
        
        const svg = d3.select("#tree")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", "translate(100,50)");
        
        const tree = d3.tree().size([height - 100, width - 200]);
        const root = d3.hierarchy(data);
        tree(root);
        
        // Add links
        svg.selectAll(".link")
            .data(root.links())
            .enter()
            .append("path")
            .attr("class", "link")
            .attr("d", d3.linkHorizontal()
                .x(d => d.y)
                .y(d => d.x));
        
        // Add nodes
        const node = svg.selectAll(".node")
            .data(root.descendants())
            .enter()
            .append("g")
            .attr("class", "node")
            .attr("transform", d => `translate(${{d.y}},${{d.x}})`);
        
        const colorScale = d3.scaleOrdinal(d3.schemeSet3);
        
        node.append("circle")
            .attr("r", 8)
            .style("fill", d => colorScale(d.data.level || 0));
        
        node.append("text")
            .attr("dy", ".35em")
            .attr("x", d => d.children ? -13 : 13)
            .style("text-anchor", d => d.children ? "end" : "start")
            .text(d => d.data.name);
        
        // Tooltip
        const tooltip = d3.select(".tooltip");
        
        node.on("mouseover", function(event, d) {{
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
            tooltip.html(`<strong>${{d.data.name}}</strong><br/>${{d.data.words || ''}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }})
        .on("mouseout", function(d) {{
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        }});
    </script>
</body>
</html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive tree saved to {save_path}")
        print(f"Open the file in a web browser to view the interactive visualization")
        return save_path
    
    def save_results_to_txt(self, save_path='hlda_results.txt', top_n=15, num_sample_docs=10):
        """Save comprehensive results to a text file."""
        if not self.results:
            print("Train model first.")
            return
            
        mdl = self.results["model"]
        params = self.results["params"]
        coherence = self.results["coherence"]
        topics_by_level = self.results["topics_by_level"]
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("HIERARCHICAL LDA RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Model parameters
            f.write("MODEL PARAMETERS\n")
            f.write("-"*80 + "\n")
            f.write(f"Depth: {params['depth']}\n")
            f.write(f"Alpha: {params['alpha']}\n")
            f.write(f"Eta: {params['eta']}\n")
            f.write(f"Gamma: {params['gamma']}\n")
            f.write(f"Coherence Score: {coherence:.4f}\n")
            f.write(f"Log-likelihood per word: {self.results['ll_per_word']:.4f}\n")
            f.write(f"Number of documents: {len(self.processed_texts)}\n")
            f.write(f"Vocabulary size: {len(self.dictionary)}\n")
            
            # Count live topics
            live_topics = sum(1 for tid in range(mdl.k) if mdl.is_live_topic(tid))
            f.write(f"Number of live topics: {live_topics}\n\n")
            
            # Topics organized by level
            f.write("="*80 + "\n")
            f.write("TOPICS BY HIERARCHY LEVEL\n")
            f.write("="*80 + "\n\n")
            
            for level in sorted(topics_by_level.keys()):
                f.write(f"{'='*80}\n")
                f.write(f"LEVEL {level}\n")
                f.write(f"{'='*80}\n\n")
                
                for topic_id, words in topics_by_level[level]:
                    parent = mdl.parent_topic(topic_id) if level > 0 else None
                    parent_str = f" (Parent: Topic {parent})" if parent is not None else ""
                    
                    f.write(f"Topic {topic_id}{parent_str}\n")
                    f.write("-"*80 + "\n")
                    
                    # Write words with probabilities
                    for i, (word, prob) in enumerate(words[:top_n], 1):
                        f.write(f"  {i:2d}. {word:20s} (prob: {prob:.4f})\n")
                    f.write("\n")
            
            # Hierarchical tree structure
            f.write("="*80 + "\n")
            f.write("HIERARCHICAL TREE STRUCTURE\n")
            f.write("="*80 + "\n\n")
            
            # Collect and sort topics
            topic_info = []
            for topic_id in range(mdl.k):
                if mdl.is_live_topic(topic_id):
                    level = mdl.level(topic_id)
                    parent = mdl.parent_topic(topic_id) if level > 0 else None
                    words = mdl.get_topic_words(topic_id, top_n=5)
                    topic_info.append({
                        'id': topic_id,
                        'level': level,
                        'parent': parent,
                        'words': words
                    })
            
            topic_info.sort(key=lambda x: (x['level'], x['id']))
            
            for topic in topic_info:
                indent = "  " * topic['level']
                word_str = ", ".join([w for w, _ in topic['words']])
                parent_str = f" [Parent: {topic['parent']}]" if topic['parent'] is not None else ""
                f.write(f"{indent}├─ Topic {topic['id']} (Level {topic['level']}){parent_str}\n")
                f.write(f"{indent}   {word_str}\n\n")
            
            # Sample document paths
            f.write("="*80 + "\n")
            f.write(f"SAMPLE DOCUMENT PATHS (first {num_sample_docs} documents)\n")
            f.write("="*80 + "\n\n")
            
            for doc_id in range(min(num_sample_docs, len(mdl.docs))):
                doc = mdl.docs[doc_id]
                path = doc.path
                f.write(f"Document {doc_id}:\n")
                f.write("-"*80 + "\n")
                for level, topic_id in enumerate(path):
                    words = mdl.get_topic_words(topic_id, top_n=5)
                    word_str = ", ".join([w for w, _ in words])
                    f.write(f"  Level {level} → Topic {topic_id}: {word_str}\n")
                f.write("\n")
            
            # Topic statistics
            f.write("="*80 + "\n")
            f.write("TOPIC STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            # Count documents per topic
            topic_doc_counts = {tid: 0 for tid in range(mdl.k) if mdl.is_live_topic(tid)}
            for doc in mdl.docs:
                for topic_id in doc.path:
                    if topic_id in topic_doc_counts:
                        topic_doc_counts[topic_id] += 1
            
            f.write("Documents assigned to each topic:\n")
            f.write("-"*80 + "\n")
            for topic_id in sorted(topic_doc_counts.keys()):
                level = mdl.level(topic_id)
                count = topic_doc_counts[topic_id]
                pct = (count / len(mdl.docs)) * 100
                f.write(f"  Topic {topic_id:3d} (Level {level}): {count:5d} docs ({pct:5.2f}%)\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Results saved to {save_path}")
        return save_path


# -------------------- Example usage --------------------
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

    analyzer = OptimizedHLDA(platform='lw')
    analyzer.load_csv_files(max_posts=None)  # Start with subset for testing
    analyzer.prepare_corpus()

    # Option 1: Train a single model
    # print("\n" + "="*60)
    # print("TRAINING SINGLE MODEL")
    # print("="*60)
    # model, coherence, _ = analyzer.train_hlda(depth=3, alpha=10.0, gamma=1.0)
    # analyzer.visualize_tree()
    # analyzer.print_topic_summary()
    # analyzer.analyze_document_paths()
    
    # # Visualizations
    # print("\nCreating visualizations...")
    # analyzer.plot_tree_matplotlib(top_n=5, save_path='hlda_tree.png')
    # analyzer.create_interactive_tree_html(top_n=8, save_path='hlda_tree.html')

    # Option 2: Find best parameters (comment out above and uncomment below)
    print("\n" + "="*60)
    print("GRID SEARCH FOR BEST PARAMETERS")
    print("="*60)
    model, best_params, df = analyzer.find_best_hlda(
        depths=[2, 3],
        alphas=[5.0, 10.0],
        gammas=[0.5, 1.0],
        iterations=300
    )
    analyzer.visualize_tree()
    analyzer.print_topic_summary()
    analyzer.save_results_to_txt(f'src/metadata/clustering_results/{analyzer.platform}/hdla_summary.txt', top_n=15, num_sample_docs=10)
    analyzer.plot_tree_matplotlib(top_n=3, save_path=f'src/metadata/img/{analyzer.platform}/hlda_tree.pdf')
    analyzer.create_interactive_tree_html(top_n=10, save_path=f'src/metadata/img/{analyzer.platform}/hlda_tree.html')