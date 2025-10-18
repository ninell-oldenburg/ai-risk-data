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

    # -------------------- Loading & preprocessing --------------------
    def load_csv_files(self, start_year=2015, end_year=2024, max_posts=None):
        all_posts = []
        for year in range(start_year, end_year + 1):
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
        return texts

    # -------------------- Modeling --------------------
    def train_hlda(self, depth=2, alpha=10.0, eta=0.1, gamma=1.0, iterations=500):
        """
        Train a single hLDA model.
        depth: number of tree levels (e.g. 2–4)
        alpha, eta, gamma: model hyperparameters
        """
        print(f"Training hLDA(depth={depth}, alpha={alpha}, eta={eta}, gamma={gamma})")
        mdl = tp.HLDAModel(depth=depth, alpha=alpha, eta=eta, gamma=gamma)

        for doc in self.processed_texts:
            mdl.add_doc(doc)

        mdl.burn_in = 100
        mdl.train(iterations)
        print(f"Log-likelihood: {mdl.ll_per_word:.4f}")

        # Extract topics by level
        topics_by_level = {}
        for level in range(depth):
            topics = mdl.get_topic_words_by_level(level, top_n=15)
            topics_by_level[level] = topics

        # Compute coherence using gensim
        # Convert top words per level to gensim format
        top_words_all = []
        for level_topics in topics_by_level.values():
            for tw in level_topics:
                top_words_all.append([w for w, _ in tw])

        cm = CoherenceModel(
            topics=top_words_all,
            texts=self.processed_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
        print(f"Coherence: {coherence:.4f}")

        self.results = {
            "model": mdl,
            "topics_by_level": topics_by_level,
            "coherence": coherence,
            "params": {"depth": depth, "alpha": alpha, "eta": eta, "gamma": gamma}
        }
        return mdl, coherence

    def find_best_hlda(self,
                       depths=[2],
                       alphas=[5.0, 10.0],
                       gammas=[0.5, 1.0, 2.0],
                       eta=0.1):
        """
        Simple grid search for best hyperparameters by coherence
        """
        results = []
        best_coh, best_model, best_params = -1, None, None
        for depth in depths:
            for alpha in alphas:
                for gamma in gammas:
                    mdl, coh = self.train_hlda(depth=depth, alpha=alpha,
                                               eta=eta, gamma=gamma)
                    results.append({
                        "depth": depth, "alpha": alpha, "gamma": gamma, "coherence": coh
                    })
                    if coh > best_coh:
                        best_coh, best_model = coh, mdl
                        best_params = {"depth": depth, "alpha": alpha,
                                       "eta": eta, "gamma": gamma}
        df = pd.DataFrame(results)
        print("\nBest hLDA params:", best_params)
        print(df)
        return best_model, best_params, df

    # -------------------- Visualization --------------------
    def visualize_tree(self, top_n=10):
        if not self.results:
            print("Train model first.")
            return
        mdl = self.results["model"]
        print("\nTopic tree:\n")
        mdl.print_tree(topic_word_top_n=top_n)


# -------------------- Example usage --------------------
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

    analyzer = OptimizedHLDA(platform='lw')
    analyzer.load_csv_files(max_posts=None)
    analyzer.prepare_corpus()

    # Find best depth and params
    model, best_params, df = analyzer.find_best_hlda(depths=[2],
                                                     alphas=[5.0,10.0],
                                                     gammas=[0.5,1.0])
    analyzer.visualize_tree()
