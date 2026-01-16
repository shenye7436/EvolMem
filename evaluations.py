# encoding: utf-8
import os
import json
import re
import glob
import argparse
import traceback
import threading
import ast
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ============================================================
# 1. Utilities
# ============================================================
log_lock = threading.Lock()

def write_log(log_path: str, msg: str):
    with log_lock:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

def write_error(log_path: str, file_path: str, exc: Exception, context: str = ""):
    with log_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"ERROR | {file_path} | {context} | {repr(exc)}\n")
            f.write("Detailed traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*80 + "\n")

def LLM_backend(api_key: str, messages: List[Dict[str, str]], model_name: str, log_path: str = None) -> str:
    """LLM"""
    try:
        client = OpenAI(
            base_url="your url", #Please revise to your own base url
            api_key=api_key
        )
        resp = client.chat.completions.create(model=model_name, messages=messages)
        return resp.choices[0].message.content
    except Exception as e:
        if log_path:
            write_log(log_path, f"LLM_backend | model={model_name} | ERROR: {repr(e)}")
        raise

# ============================================================
# 2. ARC 
# ============================================================
def normalize_arc_answer(text: str) -> List[str]:
    """
    ARC 
    """
    if not text:
        return []
    return re.findall(r"\d+", text)

# ============================================================
# 3. Similarity & Ordering
# ============================================================
def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def compute_lcs(seq, correct_len):
    """LCS"""
    n, m = len(seq), correct_len
    if n == 0 or m == 0: return 0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if seq[i] == j:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def find_best_matching_steps(correct_steps, generation_steps):
    """TF-IDF """
    if not correct_steps or not generation_steps:
        return [], [], []
    vectorizer = TfidfVectorizer().fit(correct_steps + generation_steps)
    cor_vec = vectorizer.transform(correct_steps)
    gen_vec = vectorizer.transform(generation_steps)
    sim_matrix = cosine_similarity(gen_vec, cor_vec)
    matched, sims, all_sims, used = [], [], [], set()

    for i in range(len(generation_steps)):
        candidates = [j for j in range(len(correct_steps)) if j not in used]
        best = max(candidates, key=lambda j: sim_matrix[i][j]) if candidates else sim_matrix[i].argmax()
        used.add(best)
        matched.append(int(best))
        sims.append(float(sim_matrix[i][best]))
        all_sims.append(sim_matrix[i].tolist())
    return matched, sims, all_sims

def match_model_to_correct(model_steps, correct_steps):
    """Sequence Matcher"""
    matched_sequence, best_similarities, all_similarities = [], [], []
    for m_step in model_steps:
        sims = [similarity(m_step, c_step) for c_step in correct_steps]
        if sims:
            best_index = sims.index(max(sims))
            matched_sequence.append(best_index)
            best_similarities.append(max(sims))
        else:
            matched_sequence.append(-1)
            best_similarities.append(0)
        all_similarities.append(sims)
    return matched_sequence, best_similarities, all_similarities

# ============================================================
# 4. Procedure Extraction
# ============================================================
def extract_steps_ordered(api_key: str, model_name: str, text: str) -> List[str]:
    """PROCEDURAL_LEARNING"""
    if not text: return []
    prompt = f"Extract all procedural steps from the text below.\nFollow the exact order of appearance. Do not merge or invent steps.\nText:\n{text}"
    messages = [{"role": "system", "content": "You are a strict procedural step extractor."}, {"role": "user", "content": prompt}]
    try:
        response = LLM_backend(api_key, messages, model_name)
        steps = []
        for line in response.strip().split("\n"):
            line = re.sub(r'^step\s*\d+[:\-]?\s*', '', line, flags=re.I).strip()
            if len(line) > 3: steps.append(line)
        return steps
    except: return []

def extract_steps_via_llm_and_regex(api_key: str, judge_model: str, text: str, log_path: str) -> Tuple[List[str], bool]:
    """ STEP_BY_STEP_INVERSION """
    if not text: return [], False
    prompt = f"""Extract all steps from the following text IN ORDER. Return each step on its own line. DO NOT add new information. Remove numbering like Step X. Additionally, first check whether the steps are in reverse order. If they appear reversed, indicate that clearly as: "STEPS_REVERSED: YES" at the top. Otherwise, indicate: "STEPS_REVERSED: NO"\nText:\n{text}"""
    messages = [{"role": "system", "content": "You are a strict step extractor."}, {"role": "user", "content": prompt}]
    raw = LLM_backend(api_key, messages, judge_model, log_path)
    is_reversed, steps = False, []
    for line in raw.splitlines():
        if "STEPS_REVERSED:" in line.upper():
            is_reversed = "YES" in line.upper()
            continue
        l = re.sub(r'^(step\s*\d+[:\.\)]\s*|\d+[)\.]\s*|["\'])', '', line.strip(), flags=re.IGNORECASE)
        if l: steps.append(l)
    return steps, is_reversed

def extract_correct_steps_from_text_or_dict(text: str) -> List[str]:
    """Extract Correct Procedures"""
    if not text: return []
    try:
        d = ast.literal_eval(text.strip())
        if isinstance(d, dict): return list(d.values())
    except: pass
    pattern = re.compile(r"Step\s*\d+\s*:\s*['\"]?(.*?)['\"]?(?=\s*Step\s*\d+\s*:|$)", re.IGNORECASE | re.DOTALL)
    return [m.strip() for m in pattern.findall(text)]

def vt8k_extract_final_answer(api_key: str, judge_model: str, final_question: str, model_answer: str, log_path: str) -> str:
    """Extract Model Answers in VT-8K """
    prompt = f"""Please read the Final Question, Model Answer carefully and extract the final variable name(s) only (NOT their values or numeric equivalents). Output them as a single string only, following the exact same format as the example.

#Requirements:

The output must contain only one string, with answers separated by spaces if there are multiple parts.

Do not add any explanations, labels, or extra text.

Keep the same order and formatting as they appear in the Model Answer.

The final output must look like the example format:
"XJQAH JDFHV HLOLF ABC SJ"

You should remember that VAR stands for assignment, not an answer.

#Input:
Final Question:
{final_question}

Model Answer:
{model_answer}

#Output:
"A B C D"
"""
    messages = [{"role": "system", "content": "You are a strict answer extractor."}, {"role": "user", "content": prompt}]
    return LLM_backend(api_key, messages, judge_model, log_path).strip()

# ============================================================
# 5.  Task-specific Logic
# ============================================================
def vt8k_metrics(correct_answer: str, model_answer: str) -> Dict[str, float]:
    """VT-8K Evaluation Metric"""
    c_tokens, m_tokens = correct_answer.strip().split(), model_answer.strip().split()
    c_set, m_set = set(c_tokens), set(m_tokens)
    precision = len(c_set & m_set) / len(m_tokens) if m_tokens else 0.0
    return {"precision": round(precision, 4)}

def judge_pipeline_unified(api_key: str, judge_model: str, correct_answer: str, model_answer: str, pipeline: str, log_path: str) -> float:
    prompt = f"""
You are an expert evaluator. Please score the model answer according to the correct answer and evaluation pipeline that are provided.

# Evaluation_Pipeline
{pipeline}

# Correct Answer
{correct_answer}

# Model Answer
{model_answer}

Return JSON only:
{{ "overall_score": <float between 0 and 1> }}
"""
    messages = [{"role": "system", "content": "You are an objective evaluator. Output only JSON."}, {"role": "user", "content": prompt}]
    raw = LLM_backend(api_key, messages, judge_model, log_path)
    try:
        return float(json.loads(raw).get("overall_score", 0.0))
    except: return 0.0

# ============================================================
# 6. Single File Evaluation
# ============================================================
def evaluate_single_file(file_path, api_key, judge_model, gen_model, eval_root, skip_existing=False):
    log_path = os.path.join(eval_root, "log.txt")
    base = os.path.basename(file_path)
    m = re.match(r"gen_(\d+)\.json$", base)
    suffix = m.group(1) if m else base.replace("_gen.json", "").replace(".json", "")
    out_path = os.path.join(eval_root, f"eval_{suffix}.json")

    if skip_existing and os.path.exists(out_path): return {"file": file_path, "status": "skipped"}

    try:
        with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
        source = (data.get("source") or "").strip().lower()
        fn_name = (data.get("function_name") or "").strip()
        model_ans = data.get("model_answer", "")
        correct_ans = data.get("correct_answer", "")
        result = {"file": file_path, "source": source, "final_metric": 0.0}

        # ---------------- 1. ARC   ----------------
        if source == "arc":
            model_nums = normalize_arc_answer(model_ans)
            correct_nums = normalize_arc_answer(correct_ans)
            score = 1 if model_nums == correct_nums else 0
            result.update({
                "final_metric": score,
                "model_answer": model_nums,
                "correct_answer": correct_nums
            })

        # ---------------- 2. VT_8K  ----------------
        elif source.startswith("vt_8k"):
            ext_ans = vt8k_extract_final_answer(api_key, judge_model, data.get("final_question", ""), model_ans, log_path)
            metrics = vt8k_metrics(correct_ans, ext_ans)
            result.update({
                "final_metric": metrics["precision"], 
                "model_answer": ext_ans,
                "correct_answer":"correct_ans"
                })

        # ---------------- 3. PROCEDURAL_LEARNING  ----------------
        elif fn_name == "PROCEDURAL_LEARNING":
            gen_steps = extract_steps_ordered(api_key, judge_model, model_ans)
            cor_steps = extract_steps_ordered(api_key, judge_model, correct_ans)
            matched, sims, _ = find_best_matching_steps(cor_steps, gen_steps)
            lcs_len = compute_lcs(matched, len(cor_steps))
            score = round(lcs_len / len(cor_steps), 3) if cor_steps else 0.0
            result.update({
                "model_answer_steps": gen_steps,
                "correct_answer_steps": cor_steps, 
                "matched_sequence": matched, 
                "final_metric": score})

        # ---------------- 4. STEP_BY_STEP_INVERSION  ----------------
        elif fn_name == "STEP_BY_STEP_INVERSION":
            m_steps, is_rev = extract_steps_via_llm_and_regex(api_key, judge_model, model_ans, log_path)
            if not is_rev:
                result.update({"final_metric": 0.0, "notes": "Non-reversed"})
            else:
                c_steps = extract_correct_steps_from_text_or_dict(correct_ans)
                matched, _, _ = match_model_to_correct(m_steps, c_steps)
                lcs_len = compute_lcs(matched, len(c_steps))
                result.update({"final_metric": round(lcs_len/len(c_steps), 3) if c_steps else 0})

        # ---------------- 5. Other pipelines ----------------
        else:
            score = judge_pipeline_unified(api_key, judge_model, correct_ans, model_ans, data.get("evaluation_pipeline", ""), log_path)
            if fn_name in {"STYLE_TEMPLATE_CONSTRAINT", "FORMAT_CONSTRAINT", "TRANSLATION"}:
                score = 1.0 if score > 0.7 else 0.0
            result.update({"final_metric": score})

        with open(out_path, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
        write_log(log_path, f"SUCCESS | {file_path} | score={result['final_metric']}")
        return result
    except Exception as e:
        write_error(log_path, file_path, e, "Evaluation")
        return {"file": file_path, "status": "error"}

# ============================================================
# 7. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--gen_model", required=True)
    parser.add_argument("--judge_model", default="gpt-4.1")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--input_dir", default="./generations")
    parser.add_argument("--save_root", default="./evaluations")
    args = parser.parse_args()

    gen_root = os.path.join(args.input_dir, args.gen_model)
    eval_root = os.path.join(args.save_root, args.gen_model)
    os.makedirs(eval_root, exist_ok=True)

    gen_files = sorted(glob.glob(os.path.join(gen_root, "**", "*.json"), recursive=True))
    print(f"🚀 Found {len(gen_files)} files in {gen_root}")

    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(evaluate_single_file, f, args.api_key, args.judge_model, args.gen_model, eval_root) for f in gen_files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            results.append(fut.result())

    print(f"✅ Done. Results saved to {eval_root}")

if __name__ == "__main__":
    main()