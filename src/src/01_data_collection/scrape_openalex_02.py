import os
import time
from datetime import datetime
from typing import List, Dict, Tuple
import requests
import pandas as pd
import json
from collections import Counter
from itertools import combinations

# ==============================================================================
# TOPIC DEFINITIONS
# ==============================================================================
safety_alignment_keywords = [
    'safety', 'safe', 'alignment', 'aligned', 'value alignment',
    'ai safety', 'safe ai', 'machine ethics', 'beneficial ai',
    'corrigibility', 'corrigible', 'reward hacking', 'wireheading',
    'mesa-optimization', 'mesa-optimizer', 'inner alignment', 'outer alignment',
    'value learning', 'inverse reinforcement learning', 'irl',
    'preference learning', 'human feedback', 'rlhf', 'reinforcement learning from human',
    'iterated amplification', 'debate', 'recursive reward modeling',
    'cooperative inverse', 'assistance game'
]

fairness_bias_keywords = [
    'fairness', 'fair', 'bias', 'biased', 'discrimination', 'discriminatory',
    'algorithmic fairness', 'algorithmic bias', 'algorithmic discrimination',
    'demographic parity', 'equalized odds', 'calibration',
    'disparate impact', 'treatment disparity', 'counterfactual fairness',
    'individual fairness', 'group fairness', 'protected attribute',
    'sensitive attribute', 'fairness metric', 'bias mitigation',
    'debiasing', 'fairness intervention'
]

interpretability_keywords = [
    'interpretability', 'interpretable', 'explainability', 'explainable',
    'transparency', 'transparent', 'xai', 'explainable ai',
    'interpretable machine learning', 'model interpretation',
    'feature attribution', 'saliency', 'attention visualization',
    'lime', 'shap', 'grad-cam', 'activation maximization',
    'circuit', 'mechanistic interpretability', 'neural circuit'
]

robustness_keywords = [
    'robust', 'robustness', 'adversarial', 'adversarial example',
    'adversarial attack', 'adversarial training', 'certified defense',
    'adversarial robustness', 'poisoning attack', 'backdoor attack',
    'distributional shift', 'distribution shift', 'out-of-distribution',
    'ood detection', 'covariate shift'
]

governance_risk_keywords = [
    'governance', 'regulation', 'policy', 'risk', 'existential risk',
    'x-risk', 'catastrophic risk', 'ai governance', 'ai policy',
    'ai regulation', 'responsible ai', 'responsible artificial intelligence',
    'trustworthy ai', 'ai ethics', 'ethical ai', 'ai risk',
    'misuse', 'dual use', 'malicious use', 'arms race',
    'compute governance', 'model evaluation', 'red teaming'
]

llm_safety_keywords = [
    'language model safety', 'llm safety', 'prompt injection',
    'jailbreak', 'jailbreaking', 'harmful content', 'toxic',
    'toxicity', 'hate speech', 'misinformation', 'hallucination',
    'instruction following', 'helpfulness', 'harmlessness',
    'constitutional ai', 'rlhf', 'instruct', 'chat model'
]

# Combine all
all_keywords = (
    safety_alignment_keywords + 
    fairness_bias_keywords + 
    interpretability_keywords + 
    robustness_keywords + 
    governance_risk_keywords + 
    llm_safety_keywords
)

# Core AI safety topics - these are unambiguously relevant
CORE_SAFETY_TOPICS = {
    "Ethics & Social Impacts of AI": "T10883",
    "Adversarial Robustness in ML": "T11689",
    "Explainable AI (XAI)": "T12026",
}

CRITICAL_ETHICS_TOPICS = {
    "Hate Speech and Cyberbullying Detection": "T12262",  # Bias/harm detection
    "Innovative Human-Technology Interaction": "T10803",  # HCI + ethics
}

# Extended topics - contain substantial AI safety work
EXTENDED_TOPICS = {
    "Topic Modeling": "T10028",
    "Reinforcement Learning in Robotics": "T10462",
    "Neural Networks and Applications": "T10320",
    "Natural Language Processing Techniques": "T10181",
    "Computability, Logic, AI Algorithms": "T12002",
    "Decision-Making and Behavioral Economics": "T10315",
    "Evolutionary Game Theory and Cooperation": "T11252",
    "Neural dynamics and brain function": "T10581"
}

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

# Safety-specific keywords for optional filtering
SAFETY_KEYWORDS = {
    'ai safety', 'ai alignment', 'value alignment', 
    'machine ethics', 'ai risk', 'existential risk',
    'beneficial ai', 'corrigibility', 'reward hacking',
    'mesa-optimization', 'inner alignment', 'outer alignment',
    'safe reinforcement learning', 'ai governance',
    'interpretable ai', 'transparent ai'
}


class AIScholarshipAnalyzer:
    def __init__(self, email: str, topic_mode: str = "core"):
        self.base_url = "https://api.openalex.org"
        self.email = email
        self.headers = {'User-Agent': f'mailto:{email}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.topic_mode = topic_mode
        
        if topic_mode == "core":
            self.topics = CORE_SAFETY_TOPICS
        elif topic_mode == "extended":
            self.topics = {**CORE_SAFETY_TOPICS, **EXTENDED_TOPICS}
        elif topic_mode == "ethics":
            self.topics = {**CORE_SAFETY_TOPICS, **CRITICAL_ETHICS_TOPICS}
        elif topic_mode == "all":
            self.topics = {**CORE_SAFETY_TOPICS, **CRITICAL_ETHICS_TOPICS, **EXTENDED_TOPICS}
        else:
            raise ValueError(f"Invalid topic_mode: {topic_mode}")
        
        self.topic_ids = list(self.topics.values())
    
    def _build_topic_filter(self) -> str:
        return '|'.join(self.topic_ids)
    
    def validate_coverage(self, validation_dois: Dict[str, str] = None) -> Tuple[float, List[Dict]]:
        if validation_dois is None:
            validation_dois = VALIDATION_PAPERS
        
        print(f"\nValidating coverage on {len(validation_dois)} papers...")
        
        found = 0
        missing = []
        not_in_openalex = []
        
        for doi, paper_name in validation_dois.items():
            clean_doi = doi.replace('https://doi.org/', '')
            url = f"{self.base_url}/works/doi:{clean_doi}"
            
            try:
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 404:
                    not_in_openalex.append({'doi': doi, 'name': paper_name})
                    continue
                
                response.raise_for_status()
                paper = response.json()
                
                paper_topics = [t['id'].split('/')[-1] for t in paper.get('topics', [])]
                
                if any(topic_id in paper_topics for topic_id in self.topic_ids):
                    found += 1
                else:
                    missing.append({
                        'doi': doi,
                        'name': paper_name,
                        'topics': [t['display_name'] for t in paper.get('topics', [])]
                    })
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"  Error checking {paper_name}: {e}")
                continue
        
        total_checkable = len(validation_dois) - len(not_in_openalex)
        coverage = found / total_checkable if total_checkable > 0 else 0
        
        print(f"Coverage: {coverage:.1%} ({found}/{total_checkable} papers)")
        print(f"Not in OpenAlex: {len(not_in_openalex)}")
        
        return coverage, missing


def test_topic_set_fast(topic_ids: List[str]) -> float:
    """Fast coverage test using full validation set."""
    analyzer = AIScholarshipAnalyzer.__new__(AIScholarshipAnalyzer)
    analyzer.base_url = "https://api.openalex.org"
    analyzer.email = "ninelloldenburg@gmail.com"
    analyzer.headers = {'User-Agent': f'mailto:{analyzer.email}'}
    analyzer.session = requests.Session()
    analyzer.session.headers.update(analyzer.headers)
    analyzer.topic_ids = topic_ids
    
    found = 0
    total = 0
    
    for doi, paper_name in VALIDATION_PAPERS.items():
        clean_doi = doi.replace('https://doi.org/', '')
        url = f"{analyzer.base_url}/works/doi:{clean_doi}"
        
        try:
            response = analyzer.session.get(url, timeout=10)
            if response.status_code == 404:
                continue
            
            response.raise_for_status()
            paper = response.json()
            
            paper_topics = [t['id'].split('/')[-1] for t in paper.get('topics', [])]
            
            if any(topic_id in paper_topics for topic_id in topic_ids):
                found += 1
            
            total += 1
            time.sleep(0.02)
            
        except:
            continue
    
    coverage = found / total if total > 0 else 0
    return coverage


def estimate_paper_count(topic_ids: List[str], start_year: int = 2015, end_year: int = 2025) -> int:
    """Estimate total papers for topic combination."""
    analyzer = AIScholarshipAnalyzer.__new__(AIScholarshipAnalyzer)
    analyzer.base_url = "https://api.openalex.org"
    analyzer.email = "ninelloldenburg@gmail.com"
    analyzer.headers = {'User-Agent': f'mailto:{analyzer.email}'}
    analyzer.session = requests.Session()
    analyzer.session.headers.update(analyzer.headers)
    
    topic_filter = '|'.join(topic_ids)
    
    url = f"{analyzer.base_url}/works"
    params = {
        'filter': f'publication_year:{start_year}-{end_year},type:article,topics.id:{topic_filter}',
        'per-page': 1
    }
    
    try:
        response = analyzer.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        count = data.get('meta', {}).get('count', 0)
        return count
    except Exception as e:
        print(f" (Error: {e})", end='')
        return 0


def estimate_application_filter_rate(topic_ids: List[str], sample_size: int = 100) -> float:
    """Estimate what % of papers would be filtered as applications."""
    analyzer = AIScholarshipAnalyzer.__new__(AIScholarshipAnalyzer)
    analyzer.base_url = "https://api.openalex.org"
    analyzer.email = "ninelloldenburg@gmail.com"
    analyzer.headers = {'User-Agent': f'mailto:{analyzer.email}'}
    analyzer.session = requests.Session()
    analyzer.session.headers.update(analyzer.headers)
    
    topic_filter = '|'.join(topic_ids)
    
    url = f"{analyzer.base_url}/works"
    params = {
        'filter': f'publication_year:2020-2024,type:article,topics.id:{topic_filter}',
        'per-page': sample_size,
        'select': 'title,concepts'
    }
    
    try:
        response = analyzer.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        papers = response.json().get('results', [])
        
        if not papers:
            return 0.0
        
        application_domains = {
            'medicine', 'medical', 'healthcare', 'clinical', 'diagnosis', 'patient',
            'disease', 'hospital', 'health', 'therapy', 'treatment',
            'education', 'student', 'teaching', 'learning analytics',
            'finance', 'financial', 'trading', 'stock',
            'agriculture', 'crop', 'farming',
            'manufacturing', 'industrial',
            'traffic', 'transportation', 'autonomous vehicle',
        }
        
        safety_keywords = {
            'safety', 'fairness', 'bias', 'robustness', 'interpretability',
            'explainability', 'transparency', 'adversarial', 'ethics',
            'governance', 'alignment', 'risk', 'harmful'
        }
        
        application_count = 0
        
        for paper in papers:
            title = paper.get('title', '').lower()
            concepts = ' '.join([c['display_name'].lower() for c in paper.get('concepts', [])])
            text = title + ' ' + concepts
            
            has_application = any(domain in text for domain in application_domains)
            has_safety = any(keyword in text for keyword in safety_keywords)
            
            if has_application and not has_safety:
                application_count += 1
        
        filter_rate = application_count / len(papers)
        return filter_rate
        
    except Exception as e:
        return 0.0


def check_sample_quality(topic_ids: List[str], sample_size: int = 50) -> float:
    """FIXED: Check quality of papers (1-10 scale)."""
    analyzer = AIScholarshipAnalyzer.__new__(AIScholarshipAnalyzer)
    analyzer.base_url = "https://api.openalex.org"
    analyzer.email = "ninelloldenburg@gmail.com"
    analyzer.headers = {'User-Agent': f'mailto:{analyzer.email}'}
    analyzer.session = requests.Session()
    analyzer.session.headers.update(analyzer.headers)
    
    topic_filter = '|'.join(topic_ids)
    
    url = f"{analyzer.base_url}/works"
    params = {
        'filter': f'publication_year:2020-2024,type:article,topics.id:{topic_filter}',
        'per-page': sample_size,
        'select': 'title,concepts'
    }
    
    try:
        response = analyzer.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        papers = response.json().get('results', [])
        
        if not papers:
            return 5.0
        
        safety_focused = 0
        
        for paper in papers:
            title = paper.get('title', '').lower()
            concepts = ' '.join([c['display_name'].lower() for c in paper.get('concepts', [])])
            text = f"{title} {concepts}"
            
            safety_keywords = [
                'safety', 'safe', 'alignment', 'fairness', 'fair', 'bias',
                'robustness', 'robust', 'adversarial', 'interpretability',
                'explainability', 'ethics', 'ethical', 'governance', 'risk'
            ]
            
            application_keywords = [
                'medical', 'healthcare', 'clinical', 'patient', 'disease',
                'education', 'student', 'teaching',
                'financial', 'stock', 'trading',
                'agriculture', 'crop', 'farming',
                'manufacturing', 'industrial'
            ]
            
            has_safety = any(kw in text for kw in safety_keywords)
            has_application = any(kw in text for kw in application_keywords)
            
            # Score: pure safety = 1, safety+application = 0.5, application only = 0
            if has_safety and not has_application:
                safety_focused += 1
            elif has_safety and has_application:
                safety_focused += 0.5
        
        # Quality = (safety_focused / total) * 10
        quality = (safety_focused / len(papers)) * 10
        return min(10.0, max(0.0, quality))
        
    except Exception as e:
        return 5.0


def test_every_topic_combination():
    """Test EVERY combination exhaustively."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     Testing EVERY Topic Combination (Exhaustive Search)              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    core_topics = {
        "T10883": "Ethics & Social Impacts of AI",
        "T11689": "Adversarial Robustness in ML",
        "T12026": "Explainable AI (XAI)",
    }
    
    optional_topics = {
        "T12262": "Hate Speech Detection",
        "T10803": "Human-Technology Interaction",
        "T10028": "Topic Modeling",
        "T10462": "RL in Robotics",
        "T10320": "Neural Networks",
        "T10181": "NLP Techniques",
        "T12002": "Computability & Logic",
        "T10315": "Decision-Making & Economics",
        "T11252": "Game Theory & Cooperation",
        "T10581": "Neural Dynamics",
    }
    
    # Generate all combinations
    all_combinations = [[]]  # Core only
    for r in range(1, len(optional_topics) + 1):
        for combo in combinations(optional_topics.keys(), r):
            all_combinations.append(list(combo))
    
    print(f"\nTesting {len(all_combinations)} combinations (~{len(all_combinations) * 3 / 60:.0f} minutes)\n")
    
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        return
    
    results = []
    
    for i, optional_combo in enumerate(all_combinations, 1):
        full_topics = list(core_topics.keys()) + optional_combo
        
        if not optional_combo:
            name = "Core only"
        else:
            optional_names = [optional_topics[t].split()[0] for t in optional_combo]
            if len(optional_names) <= 3:
                name = "Core + " + "+".join(optional_names)
            else:
                name = f"Core + {len(optional_names)} topics"
        
        if i % 50 == 0:
            print(f"\nProgress: {i}/{len(all_combinations)} ({i/len(all_combinations)*100:.1f}%)")
        
        print(f"  [{i:3d}] {name} ({len(full_topics)} topics)...", end='')
        
        # Test coverage
        coverage = test_topic_set_fast(full_topics)
        
        # Only estimate papers/quality for promising combinations
        if coverage >= 0.80:
            paper_count = estimate_paper_count(full_topics)
            filter_rate = estimate_application_filter_rate(full_topics, sample_size=50)
            papers_filtered = int(paper_count * (1 - filter_rate))
            quality = check_sample_quality(full_topics, sample_size=30)
        else:
            paper_count = 0
            papers_filtered = 0
            filter_rate = 0
            quality = 0
        
        print(f" {coverage:.1%} cov, {papers_filtered:,} papers, {quality:.1f} qual")
        
        results.append({
            'name': name,
            'optional_topics': optional_combo,
            'all_topics': full_topics,
            'topic_count': len(full_topics),
            'coverage': coverage,
            'papers_no_filter': paper_count,
            'papers_filtered': papers_filtered,
            'filter_rate': filter_rate,
            'quality': quality,
        })
        
        time.sleep(0.3)
    
    # Save results
    with open('all_topic_combinations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved to all_topic_combinations.json")
    
    # Print analysis
    print_results_analysis(results, optional_topics)
    
    return results


def print_results_analysis(results: List[Dict], optional_topics: Dict):
    """Print comprehensive analysis of results."""
    
    print(f"\n\n{'='*80}")
    print("TOP 20 COMBINATIONS BY COVERAGE")
    print(f"{'='*80}\n")
    
    print(f"{'Rank':<5} {'Topics':>6} {'Cov':>7} {'Papers':>10} {'Quality':>8} {'Name':<40}")
    print(f"{'-'*80}")
    
    top_20 = sorted(results, key=lambda x: x['coverage'], reverse=True)[:20]
    for i, r in enumerate(top_20, 1):
        print(f"{i:<5} {r['topic_count']:>6} {r['coverage']:>6.1%} {r['papers_filtered']:>10,} "
              f"{r['quality']:>7.1f} {r['name']:<40}")
    
    # Pareto frontier
    print(f"\n\n{'='*80}")
    print("PARETO OPTIMAL SOLUTIONS")
    print(f"{'='*80}\n")
    
    pareto = []
    for r in results:
        dominated = False
        for other in results:
            if (other['coverage'] >= r['coverage'] and 
                other['papers_filtered'] <= r['papers_filtered'] and
                other['papers_filtered'] > 0 and  # Exclude broken ones
                (other['coverage'] > r['coverage'] or other['papers_filtered'] < r['papers_filtered'])):
                dominated = True
                break
        if not dominated and r['coverage'] >= 0.70 and r['papers_filtered'] > 0:
            pareto.append(r)
    
    pareto_sorted = sorted(pareto, key=lambda x: x['papers_filtered'])
    
    print(f"{'Topics':>6} {'Coverage':>9} {'Papers':>10} {'Quality':>8}")
    print(f"{'-'*60}")
    
    for r in pareto_sorted:
        print(f"{r['topic_count']:>6} {r['coverage']:>8.1%} {r['papers_filtered']:>10,} {r['quality']:>7.1f}")
    
    # Best by criteria
    print(f"\n\n{'='*80}")
    print("BEST COMBINATIONS")
    print(f"{'='*80}\n")
    
    candidates = [r for r in results if r['coverage'] >= 0.85 and r['papers_filtered'] > 0]
    
    if candidates:
        best_coverage = max(candidates, key=lambda x: x['coverage'])
        best_quality = max(candidates, key=lambda x: x['quality'])
        best_small = min(candidates, key=lambda x: x['papers_filtered'])
        
        print(f"🏆 HIGHEST COVERAGE: {best_coverage['name']}")
        print(f"   {best_coverage['coverage']:.1%}, {best_coverage['papers_filtered']:,} papers, quality {best_coverage['quality']:.1f}/10")
        print(f"   Topics: {[optional_topics[t] for t in best_coverage['optional_topics']]}")
        
        print(f"\n⭐ HIGHEST QUALITY: {best_quality['name']}")
        print(f"   {best_quality['coverage']:.1%}, {best_quality['papers_filtered']:,} papers, quality {best_quality['quality']:.1f}/10")
        
        print(f"\n📊 SMALLEST: {best_small['name']}")
        print(f"   {best_small['coverage']:.1%}, {best_small['papers_filtered']:,} papers, quality {best_small['quality']:.1f}/10")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test-every":
        test_every_topic_combination()
    else:
        # Quick test
        analyzer = AIScholarshipAnalyzer("ninelloldenburg@gmail.com", topic_mode="all")
        analyzer.validate_coverage()