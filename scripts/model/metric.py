import json
import re
from typing import Dict, List, Any , Tuple
from collections import Counter
import os
# --- External Metric Libraries ---
import evaluate
from nltk.tokenize import word_tokenize

# --- Configuration ---
RESULTS_JSON_PATH = "qa_inference_results.json" # Path to your inference results file

# Initialize Hugging Face Evaluators
try:
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
except Exception as e:
    print(f"Error loading Hugging Face metrics: {e}")
    print("Please ensure you have run: pip install evaluate rouge_score")
    sys.exit(1)


# --- Normalization Function (Standard for QA Metrics) ---
def normalize_answer(s: str) -> str:
    """Lowercases, removes punctuation, and strips extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# --- F1 and Exact Match (EM) Calculation ---
def calculate_f1_em(prediction: str, ground_truth: str) -> Tuple[float, float]:
    """Calculates token-level F1 and Exact Match (EM)."""
    
    # Apply normalization to both strings
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # 1. Exact Match (EM)
    em = 1.0 if normalized_prediction == normalized_ground_truth else 0.0

    # Handle cases where either string is empty after normalization
    if normalized_prediction == normalized_ground_truth == "":
        return 1.0, 1.0
    if normalized_prediction == "" or normalized_ground_truth == "":
        return 0.0, em

    # 2. F1 Score (Token-based)
    # Using simple split tokenization after normalization
    pred_tokens = normalized_prediction.split()
    truth_tokens = normalized_ground_truth.split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0, em

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, em


# --- Main Evaluation Logic ---
def run_evaluation(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes all relevant metrics from the list of results."""
    
    # Prepare lists for batch processing
    predictions = []
    references = []
    
    # Running total for averaged metrics
    total_em = 0.0
    total_f1 = 0.0
    
    for item in results:
        pred = item.get("predicted_answer", "")
        gt = item.get("ground_truth_answer", "")

        # 1. EM and F1 (Question Answering Metrics)
        f1, em = calculate_f1_em(pred, gt)
        total_em += em
        total_f1 += f1
        
        # 2. Prepare for Generative Metrics (BLEU/ROUGE)
        # Note: BLEU/ROUGE usually compare the prediction against a *list* of references.
        # Here we use the single ground truth answer string as the reference.
        predictions.append(pred)
        references.append([gt])

    N = len(results)
    
    # --- Final Metric Calculation ---
    
    # 1. Averaged QA Metrics
    avg_em = total_em / N
    avg_f1 = total_f1 / N
    
    # 2. BLEU Score
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    
    # 3. ROUGE Score
    # We focus on ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (LCS) F-scores
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    
    
    final_metrics = {
        "N_Samples": N,
        "QA_Exact_Match (EM)": round(avg_em, 4),
        "QA_F1_Score": round(avg_f1, 4),
        "BLEU-4": round(bleu_results["bleu"], 4),
        "ROUGE-1_F1": round(rouge_results["rouge1"], 4),
        "ROUGE-2_F1": round(rouge_results["rouge2"], 4),
        "ROUGE-L_F1": round(rouge_results["rougeL"], 4),
        # NOTE: BERTScore requires its own setup and is complex to include directly here.
    }
    
    return final_metrics


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(RESULTS_JSON_PATH):
        print(f"Error: Results file not found at {RESULTS_JSON_PATH}")
        sys.exit(1)
        
    print(f"Reading results from: {RESULTS_JSON_PATH}")
    
    try:
        with open(RESULTS_JSON_PATH, 'r', encoding='utf-8') as f:
            qa_results = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing JSON file: {e}")
        sys.exit(1)
        
    if not qa_results:
        print("No results found in the JSON file. Exiting.")
        sys.exit(1)

    metrics = run_evaluation(qa_results)
    
    print("\n" + "="*50)
    print("✨ GENERATIVE QA EVALUATION RESULTS ✨")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:<30}: {value}")
    print("="*50)