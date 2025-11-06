"""
Validation script for AI Ethics + Safety corpus topic selection.
Tests different sampling strategies against a curated validation set.

Usage:
    python validate_topic_selection.py
"""

import time
from typing import List, Dict, Tuple
import requests
import json

# ==============================================================================
# TOPIC CONFIGURATIONS
# ==============================================================================

TOPIC_CONFIGS = {
    'core': {
        'name': 'Core topics only',
        'topics': ['T10883', 'T11689', 'T12026'],
        'description': 'Ethics & Social Impacts, Adversarial Robustness, Explainable AI'
    },
    
    # High-value individual topics
    'core_topic': {
        'name': 'Core + Topic Modeling',
        'topics': ['T10883', 'T11689', 'T12026', 'T10028'],
        'description': 'Core + Topic Modeling'
    },
    'core_nlp': {
        'name': 'Core + NLP',
        'topics': ['T10883', 'T11689', 'T12026', 'T10181'],
        'description': 'Core + NLP Techniques'
    },
    'core_hate': {
        'name': 'Core + Hate Speech',
        'topics': ['T10883', 'T11689', 'T12026', 'T12262'],
        'description': 'Core + Hate Speech Detection'
    },
    'core_rl': {
        'name': 'Core + RL',
        'topics': ['T10883', 'T11689', 'T12026', 'T10462'],
        'description': 'Core + RL in Robotics'
    },
    
    # Low-value topics (to show why excluded)
    'core_game': {
        'name': 'Core + Game Theory',
        'topics': ['T10883', 'T11689', 'T12026', 'T11252'],
        'description': 'Core + Game Theory & Cooperation'
    },
    'core_comp': {
        'name': 'Core + Computability',
        'topics': ['T10883', 'T11689', 'T12026', 'T12002'],
        'description': 'Core + Computability & Logic'
    },
    
    # Combinations
    'core_topic_nlp': {
        'name': 'Core + Topic + NLP',
        'topics': ['T10883', 'T11689', 'T12026', 'T10028', 'T10181'],
        'description': 'Core + Topic Modeling + NLP'
    },
    'core_topic_hate_rl': {
        'name': 'Core + Topic + Hate + RL',
        'topics': ['T10883', 'T11689', 'T12026', 'T10028', 'T12262', 'T10462'],
        'description': 'Core + Topic Modeling + Hate Speech + RL (FINAL CHOICE)'
    },
}

# ==============================================================================
# KEYWORD CONFIGURATIONS
# ==============================================================================

# Comprehensive keyword list (230 keywords)
COMPREHENSIVE_KEYWORDS = [
    # Safety & Alignment
    'safety', 'safe', 'alignment', 'aligned', 'value alignment',
    'ai safety', 'safe ai', 'machine ethics', 'beneficial ai',
    'corrigibility', 'corrigible', 'reward hacking', 'wireheading',
    'mesa-optimization', 'mesa-optimizer', 'inner alignment', 'outer alignment',
    'value learning', 'inverse reinforcement learning', 'irl',
    'preference learning', 'human feedback', 'rlhf',
    'iterated amplification', 'debate', 'recursive reward modeling',
    'cooperative ai', 'cooperation', 'multi-agent', 'coordination',
    
    # Fairness & Bias
    'fairness', 'fair', 'bias', 'biased', 'discrimination',
    'algorithmic fairness', 'algorithmic bias',
    'demographic parity', 'equalized odds', 'disparate impact',
    'counterfactual fairness', 'protected attribute',
    'bias mitigation', 'debiasing', 'trustworthy',
    'marginalized', 'harms', 'societal impact',
    
    # Interpretability
    'interpretability', 'interpretable', 'explainability', 'explainable',
    'transparency', 'transparent', 'xai',
    'feature attribution', 'saliency', 'attention visualization',
    'lime', 'shap', 'circuits', 'mechanistic interpretability',
    'model card', 'datasheet', 'documentation',
    
    # Robustness
    'robust', 'robustness', 'adversarial', 'adversarial example',
    'adversarial attack', 'adversarial training', 'certified defense',
    'poisoning attack', 'backdoor attack',
    'distributional shift', 'out-of-distribution',
    
    # Governance & Risk
    'governance', 'regulation', 'policy', 'risk',
    'existential risk', 'x-risk', 'catastrophic risk',
    'ai governance', 'ai policy', 'responsible ai',
    'misuse', 'dual use', 'malicious use',
    'model evaluation', 'red teaming',
    
    # LLM Safety
    'llm safety', 'prompt injection', 'jailbreak', 'jailbreaking',
    'toxic', 'toxicity', 'hate speech', 'misinformation',
    'hallucination', 'helpfulness', 'harmlessness',
    'stochastic parrot', 'language model', 'foundation model',
]

# Boolean search terms
AI_TERMS = [
    'artificial intelligence', 'machine learning', 'deep learning',
    'neural network', 'reinforcement learning', 'language model',
    'ai', 'ml', 'llm', 'nlp',
]

SAFETY_TERMS = [
    'safety', 'alignment', 'fairness', 'bias',
    'interpretability', 'explainability', 'robustness',
    'adversarial', 'ethics', 'governance', 'risk',
    'trustworthy', 'responsible', 'cooperation',
]

# Targeted keywords for hybrid approach
TARGETED_KEYWORDS = [
    'alignment problem', 'cooperative ai', 'malicious use',
    'model evaluation', 'extreme risk', 'circuits',
    'mechanistic interpretability', 'feature visualization',
    'existential risk', 'benchmark', 'mesa-optimization',
    'value alignment', 'catastrophic risk'
]

# Validation set - known AI safety papers that SHOULD be captured
VALIDATION_PAPERS = {
    # ========================================
    # TECHNICAL ALIGNMENT (17 papers)
    # ========================================
    "10.48550/arxiv.2401.10899": "Concrete Problems in AI Safety (Amodei)",
    "10.48550/arxiv.1706.03741": "Deep RL from Human Preferences (Christiano)",
    "10.48550/arxiv.1711.09883": "AI Safety Gridworlds (Leike)",
    "10.48550/arxiv.1906.01820": "Risks from Learned Optimization (Hubinger)",
    "10.48550/arxiv.1811.07871": "Scalable agent alignment via reward modeling (Leike)",
    "10.48550/arxiv.2112.00861": "A General Language Assistant as a Laboratory for Alignment",
    "10.48550/arxiv.2203.02155": "Training language models to follow instructions with human feedback",
    "10.48550/arxiv.2204.05862": "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback",
    "10.48550/arxiv.2212.08073": "Constitutional AI: Harmlessness from AI Feedback",
    "10.48550/arxiv.1805.00899": "AI Safety via Debate (Irving)",
    "10.48550/arxiv.1810.08575": "Amplification (Christiano)",
    "10.48550/arxiv.2209.00626": "The Alignment Problem from a Deep Learning Perspective",
    "10.1038/d41586-021-01170-0": "Cooperative AI: machines must learn to find common ground",
    "10.48550/arxiv.2012.08630": "Open Problems in Cooperative AI",
    "10.18653/v1/2022.acl-long.229": "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
    "10.18653/v1/2022.emnlp-main.225": "Red Teaming Language Models with Language Models",
    "10.48550/arxiv.2303.12712": "Sparks of Artificial General Intelligence: Early experiments with GPT-4",
    
    # ========================================
    # INTERPRETABILITY & TRANSPARENCY (10 papers)
    # ========================================
    "10.23915/distill.00007": "Feature Visualization (Olah)",
    "10.23915/distill.00010": "The Building Blocks of Interpretability (Olah)",
    "10.23915/distill.00024": "Thread: Circuits",
    "10.48550/arxiv.2211.00593": "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small",
    "10.48550/arxiv.1602.04938": "Why Should I Trust You? LIME",
    "10.48550/arxiv.1705.07874": "A Unified Approach to Interpreting Model Predictions (SHAP)",
    "10.48550/arxiv.2210.12440": "BERT: Pre-training of Deep Bidirectional Transformers",
    "10.48550/arxiv.2005.14165": "Language Models are Few-Shot Learners (GPT-3)",
    "10.48550/arxiv.1902.10186": "Attention is not Explanation",
    "10.48550/arxiv.2202.05262": "Locating and Editing Factual Associations in GPT",
    
    # ========================================
    # ADVERSARIAL ROBUSTNESS (7 papers)
    # ========================================
    "10.48550/arxiv.1312.6199": "Intriguing properties of neural networks (Szegedy)",
    "10.48550/arxiv.1412.6572": "Explaining and Harnessing Adversarial Examples (Goodfellow)",
    "10.48550/arxiv.1706.06083": "Towards Deep Learning Models Resistant to Adversarial Attacks",
    "10.48550/arxiv.1801.02774": "Adversarial Spheres",
    "10.48550/arxiv.2307.15043": "Universal and Transferable Adversarial Attacks on Aligned LLMs",
    "10.48550/arxiv.1908.07125": "Universal Adversarial Triggers for Attacking and Analyzing NLP",
    "10.48550/arxiv.2307.02483": "Jailbroken: How Does LLM Safety Training Fail?",
    
    # ========================================
    # CRITICAL AI ETHICS / FAIRNESS (28 papers)
    # ========================================
    "10.1145/3442188.3445922": "On the Dangers of Stochastic Parrots (Bender et al.)",
    "10.1145/3287560.3287598": "Fairness and Abstraction in Sociotechnical Systems",
    "10.1145/3351095.3372862": "Garbage in, garbage out?",
    "10.1145/3278721.3278729": "Measuring and Mitigating Unintended Bias in Text Classification",
    "10.1145/3287560.3287596": "Model Cards for Model Reporting",
    "10.1146/annurev-statistics-042720-125902": "Algorithmic Fairness: Choices, Assumptions, and Definitions",
    "10.48550/arxiv.2111.15366": "AI and the Everything in the Whole Wide World Benchmark",
    "10.1145/2783258.2783311": "Certifying and Removing Disparate Impact",
    "10.48550/arxiv.2112.04359": "Ethical and social risks of harm from Language Models",
    "10.48550/arxiv.2009.11462": "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models",
    "10.48550/arxiv.1911.03891": "Social Bias Frames: Reasoning about Social and Power Implications of Language",
    "10.1145/2090236.2090255": "Fairness Through Awareness (Dwork)",
    "10.1145/3287560.3287572": "Bias in Bios",
    "10.1145/3461702.3462557": "Measuring Model Biases in the Absence of Ground Truth",
    "10.1126/science.aax2342": "Dissecting racial bias in an algorithm used to manage the health of populations",
    "10.1145/3097983.3098095": "Algorithmic Decision Making and the Cost of Fairness",
    "10.18653/v1/d17-1323": "Men Also Like Shopping (gender bias in word embeddings)",
    "10.1145/3306618.3314244": "Actionable Auditing",
    "10.1145/3287560.3287561": "From Soft Classifiers to Hard Decisions",
    "10.1145/3442188.3445924": "Bold",
    "10.1145/3442188.3445901": "Measurement and Fairness",
    "10.1177/20539517211035955": "On the Genealogy of Machine Learning Datasets",
    "10.48550/arxiv.1906.02243": "Energy and Policy Considerations for Deep Learning in NLP",
    "10.48550/arxiv.2112.04359": "Ethical and social risks of harm from LLMs (Weidinger)",
    "10.18653/v1/2021.findings-emnlp.210": "Challenges in Detoxifying Language Models",
    "10.18653/v1/2021.emnlp-main.98": "Documenting Large Webtext Corpora (C4)",
    "10.48550/arxiv.2211.09110": "Holistic Evaluation of Language Models",
    "10.48550/arxiv.2305.15717": "The False Promise of Imitating Proprietary LLMs",
    
    # ========================================
    # AI GOVERNANCE & POLICY (5 papers)
    # ========================================
    "10.1007/978-3-030-35746-7_3": "Malicious use of AI (Brundage)",
    "10.1038/s41746-024-01232-3": "Navigating the EU AI Act: implications for regulated digital medical products",
    "10.48550/arxiv.2305.15324": "Model Evaluation for Extreme Risks",
    "10.1007/s43681-024-00624-1": "Democratizing value alignment: from authoritarian to democratic AI ethics",
    "10.1016/j.techsoc.2024.102747": "Balancing the tradeoff between regulation and innovation for artificial intelligence: An analysis of top-down command and control and bottom-up self-regulatory approaches",
    }

# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def test_topic_coverage(topic_ids: List[str], email: str = "your-email@example.com") -> Tuple[float, List[str]]:
    """
    Test coverage of a topic configuration against validation set.
    
    Args:
        topic_ids: List of OpenAlex topic IDs (e.g., ['T10883', 'T11689'])
        email: Email for OpenAlex API (polite pool access)
    
    Returns:
        (coverage_rate, list_of_missing_papers)
    """
    base_url = "https://api.openalex.org"
    headers = {'User-Agent': f'mailto:{email}'}
    session = requests.Session()
    session.headers.update(headers)
    
    found = 0
    missing = []
    
    print(f"  Testing {len(topic_ids)} topics...")
    
    for doi, paper_name in VALIDATION_PAPERS.items():
        clean_doi = doi.replace('https://doi.org/', '')
        url = f"{base_url}/works/doi:{clean_doi}"
        
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 404:
                continue
            
            response.raise_for_status()
            paper = response.json()
            
            paper_topics = [t['id'].split('/')[-1] for t in paper.get('topics', [])]
            
            if any(tid in paper_topics for tid in topic_ids):
                found += 1
            else:
                missing.append(paper_name)
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"    Error checking {paper_name}: {e}")
            continue
    
    coverage = found / len(VALIDATION_PAPERS)
    print(f"    Coverage: {coverage:.1%} ({found}/{len(VALIDATION_PAPERS)} papers)")
    
    return coverage, missing


def test_keyword_coverage(keywords: List[str], email: str = "your-email@example.com") -> Tuple[float, List[str]]:
    """
    Test coverage using keyword matching in title, abstract, and concepts.
    
    Args:
        keywords: List of keywords to search for
        email: Email for OpenAlex API
    
    Returns:
        (coverage_rate, list_of_missing_papers)
    """
    base_url = "https://api.openalex.org"
    headers = {'User-Agent': f'mailto:{email}'}
    session = requests.Session()
    session.headers.update(headers)
    
    found = 0
    missing = []
    
    print(f"  Testing {len(keywords)} keywords...")
    
    for doi, paper_name in VALIDATION_PAPERS.items():
        clean_doi = doi.replace('https://doi.org/', '')
        url = f"{base_url}/works/doi:{clean_doi}"
        
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 404:
                continue
            
            response.raise_for_status()
            paper = response.json()
            
            # Get searchable text
            title = paper.get('title', '').lower()
            abstract_inv = paper.get('abstract_inverted_index', {})
            abstract = ''
            if abstract_inv:
                words = [(pos, word.lower()) for word, positions in abstract_inv.items() for pos in positions]
                words.sort()
                abstract = ' '.join([w[1] for w in words])
            concepts = ' '.join([c['display_name'].lower() for c in paper.get('concepts', [])])
            
            text = f"{title} {abstract} {concepts}"
            
            # Check if any keyword matches
            if any(kw in text for kw in keywords):
                found += 1
            else:
                missing.append(paper_name)
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"    Error checking {paper_name}: {e}")
            continue
    
    coverage = found / len(VALIDATION_PAPERS)
    print(f"    Coverage: {coverage:.1%} ({found}/{len(VALIDATION_PAPERS)} papers)")
    
    return coverage, missing


def test_boolean_coverage(ai_terms: List[str], safety_terms: List[str], 
                         email: str = "your-email@example.com") -> Tuple[float, List[str]]:
    """
    Test coverage using Boolean search: (AI terms) AND (Safety terms).
    
    Args:
        ai_terms: List of AI-related terms
        safety_terms: List of safety-related terms
        email: Email for OpenAlex API
    
    Returns:
        (coverage_rate, list_of_missing_papers)
    """
    base_url = "https://api.openalex.org"
    headers = {'User-Agent': f'mailto:{email}'}
    session = requests.Session()
    session.headers.update(headers)
    
    found = 0
    missing = []
    
    print(f"  Testing Boolean: {len(ai_terms)} AI terms AND {len(safety_terms)} safety terms...")
    
    for doi, paper_name in VALIDATION_PAPERS.items():
        clean_doi = doi.replace('https://doi.org/', '')
        url = f"{base_url}/works/doi:{clean_doi}"
        
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 404:
                continue
            
            response.raise_for_status()
            paper = response.json()
            
            # Get searchable text
            title = paper.get('title', '').lower()
            abstract_inv = paper.get('abstract_inverted_index', {})
            abstract = ''
            if abstract_inv:
                words = [(pos, word.lower()) for word, positions in abstract_inv.items() for pos in positions]
                words.sort()
                abstract = ' '.join([w[1] for w in words])
            concepts = ' '.join([c['display_name'].lower() for c in paper.get('concepts', [])])
            
            text = f"{title} {abstract} {concepts}"
            
            # Check Boolean: has AI term AND has safety term
            has_ai = any(term in text for term in ai_terms)
            has_safety = any(term in text for term in safety_terms)
            
            if has_ai and has_safety:
                found += 1
            else:
                missing.append(paper_name)
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"    Error checking {paper_name}: {e}")
            continue
    
    coverage = found / len(VALIDATION_PAPERS)
    print(f"    Coverage: {coverage:.1%} ({found}/{len(VALIDATION_PAPERS)} papers)")
    
    return coverage, missing


def test_hybrid_coverage(topic_ids: List[str], keywords: List[str], 
                        email: str = "your-email@example.com") -> Tuple[float, Dict]:
    """
    Test hybrid approach: topics OR keywords.
    
    Returns:
        (coverage_rate, details_dict)
    """
    base_url = "https://api.openalex.org"
    headers = {'User-Agent': f'mailto:{email}'}
    session = requests.Session()
    session.headers.update(headers)
    
    found_by_topics = 0
    found_by_keywords = 0
    missing = []
    
    print(f"  Testing hybrid: {len(topic_ids)} topics + {len(keywords)} keywords...")
    
    for doi, paper_name in VALIDATION_PAPERS.items():
        clean_doi = doi.replace('https://doi.org/', '')
        url = f"{base_url}/works/doi:{clean_doi}"
        
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 404:
                continue
            
            response.raise_for_status()
            paper = response.json()
            
            # Check topics
            paper_topics = [t['id'].split('/')[-1] for t in paper.get('topics', [])]
            has_topic = any(tid in paper_topics for tid in topic_ids)
            
            # Check keywords
            title = paper.get('title', '').lower()
            abstract_inv = paper.get('abstract_inverted_index', {})
            abstract = ''
            if abstract_inv:
                words = [(pos, word.lower()) for word, positions in abstract_inv.items() for pos in positions]
                words.sort()
                abstract = ' '.join([w[1] for w in words])
            concepts = ' '.join([c['display_name'].lower() for c in paper.get('concepts', [])])
            text = f"{title} {abstract} {concepts}"
            has_keyword = any(kw in text for kw in keywords)
            
            # Classify
            if has_topic:
                found_by_topics += 1
            elif has_keyword:
                found_by_keywords += 1
            else:
                missing.append(paper_name)
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"    Error checking {paper_name}: {e}")
            continue
    
    total_found = found_by_topics + found_by_keywords
    coverage = total_found / len(VALIDATION_PAPERS)
    
    details = {
        'coverage': coverage,
        'found_by_topics': found_by_topics,
        'found_by_keywords': found_by_keywords,
        'missing': missing
    }
    
    print(f"    Coverage: {coverage:.1%} ({total_found}/{len(VALIDATION_PAPERS)} papers)")
    print(f"      Via topics: {found_by_topics}")
    print(f"      Via keywords: {found_by_keywords}")
    
    return coverage, details


# ==============================================================================
# MAIN VALIDATION
# ==============================================================================

def main():
    """Run full validation comparison."""
    print("""
    ═══════════════════════════════════════════════════════════════
    AI Safety Corpus Validation
    ═══════════════════════════════════════════════════════════════
    
    This script validates different sampling strategies against
    67 canonical AI safety papers.
    
    Please provide your email for OpenAlex API access:
    """)
    
    email = input("Email: ").strip()
    if not email:
        print("Email required for OpenAlex API. Exiting.")
        return
    
    results = {}
    
    # Test topic-based approaches
    print("\n" + "="*70)
    print("PART 1: TOPIC-BASED APPROACHES")
    print("="*70 + "\n")
    
    for key, config in TOPIC_CONFIGS.items():
        print(f"{config['name']}:")
        print(f"  Description: {config['description']}")
        coverage, missing = test_topic_coverage(config['topics'], email)
        results[key] = {
            'approach': 'Topic-based',
            'name': config['name'],
            'coverage': coverage,
            'missing_count': len(missing),
            'missing_papers': missing
        }
        print()
    
    # Test keyword-based approaches
    print("\n" + "="*70)
    print("PART 2: KEYWORD-BASED APPROACHES")
    print("="*70 + "\n")
    
    print("Comprehensive Keywords (~230 keywords):")
    coverage, missing = test_keyword_coverage(COMPREHENSIVE_KEYWORDS, email)
    results['keywords_comprehensive'] = {
        'approach': 'Keyword-based',
        'name': 'Comprehensive Keywords',
        'coverage': coverage,
        'missing_count': len(missing),
        'missing_papers': missing
    }
    print()
    
    print("Boolean Search (AI AND Safety):")
    coverage, missing = test_boolean_coverage(AI_TERMS, SAFETY_TERMS, email)
    results['keywords_boolean'] = {
        'approach': 'Keyword-based',
        'name': 'Boolean Search',
        'coverage': coverage,
        'missing_count': len(missing),
        'missing_papers': missing
    }
    print()
    
    # Test hybrid approach
    print("\n" + "="*70)
    print("PART 3: HYBRID APPROACH")
    print("="*70 + "\n")
    
    print("Hybrid (Topics + Targeted Keywords):")
    hybrid_coverage, hybrid_details = test_hybrid_coverage(
        TOPIC_CONFIGS['core_topic_hate_rl']['topics'],
        TARGETED_KEYWORDS,
        email
    )
    results['hybrid'] = {
        'approach': 'Hybrid',
        'name': 'Topics + Targeted Keywords',
        'coverage': hybrid_coverage,
        'found_by_topics': hybrid_details['found_by_topics'],
        'found_by_keywords': hybrid_details['found_by_keywords'],
        'missing_count': len(hybrid_details['missing']),
        'missing_papers': hybrid_details['missing']
    }
    print()
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    print(f"{'Approach':<20} {'Name':<30} {'Coverage':>10} {'Missing':>10}")
    print("-"*70)
    for key, result in results.items():
        print(f"{result['approach']:<20} {result['name']:<30} {result['coverage']:>9.1%} {result['missing_count']:>10}")
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to validation_results.json")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    print(f"""
Topic-based approaches:
  • Core only:              {results['core']['coverage']:.1%} coverage
  • Core + Topic Modeling:  {results['core_topic']['coverage']:.1%} coverage
  • Core + Topic + H + RL:  {results['core_topic_hate_rl']['coverage']:.1%} coverage

Keyword-based approaches:
  • Comprehensive keywords: {results['keywords_comprehensive']['coverage']:.1%} coverage
  • Boolean (AI AND Safety): {results['keywords_boolean']['coverage']:.1%} coverage

Hybrid approach:
  • Topics + Targeted KW:   {results['hybrid']['coverage']:.1%} coverage
    - Via topics: {results['hybrid']['found_by_topics']}/{len(VALIDATION_PAPERS)} papers
    - Via keywords: {results['hybrid']['found_by_keywords']}/{len(VALIDATION_PAPERS)} papers

Key findings:
  • Keywords achieve higher coverage but require justifying selection
  • Topics are algorithmic and reproducible
  • Hybrid balances coverage with reproducibility
    """)


if __name__ == "__main__":
    main()