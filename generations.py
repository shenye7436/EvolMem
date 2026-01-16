# encoding: utf-8
import os
import json
import argparse
from pathlib import Path
from typing import List, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import ast
import re
from tqdm import tqdm

# -------------------- Function --------------------
def safe_strip(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(safe_strip(v) for v in x)
    return str(x).strip()

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_final_question(dialogue_obj: Dict[str, Any]) -> str:
    if isinstance(dialogue_obj, dict):
        for k, v in dialogue_obj.items():
            if k == "Final_Question":
                return safe_strip(v)
            r = extract_final_question(v)
            if r:
                return r
    elif isinstance(dialogue_obj, list):
        for item in dialogue_obj:
            r = extract_final_question(item)
            if r:
                return r
    return ""

def extract_key(dialogue_obj: Dict[str, Any], keyname: str) -> str:
    if isinstance(dialogue_obj, dict):
        for k, v in dialogue_obj.items():
            if k == keyname:
                return safe_strip(v)
            r = extract_key(v, keyname)
            if r:
                return r
    elif isinstance(dialogue_obj, list):
        for item in dialogue_obj:
            r = extract_key(item, keyname)
            if r:
                return r
    return ""

def dialogue_to_messages(dialogue_obj: Dict[str, Any], is_vt: bool = False) -> List[Dict[str, str]]:
    """
    Convert structured dialogue JSON into messages.
    """
    messages = []
    if is_vt:
        sessions = [v for k, v in dialogue_obj.items() if k.startswith("session_")]
        for session in sessions:
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                for uk in ["User","User_1","User_2","User_3","User_4","User_5","User_6"]:
                    if uk in turn and safe_strip(turn[uk]):
                        messages.append({"role":"user","content":safe_strip(turn[uk])})
                        break
                for ak in ["AI","AI_1","AI_2","AI_3","AI_4","AI_5","AI_6"]:
                    if ak in turn and safe_strip(turn[ak]):
                        messages.append({"role":"assistant","content":safe_strip(turn[ak])})
                        break
    else:
        keys = sorted([k for k in dialogue_obj.keys() if k.startswith("Dialogue_")])
        for k in keys:
            item = dialogue_obj[k]
            if not isinstance(item, dict):
                continue
            for uk in ["User","User_1","User_2","User_3","User_4","User_5","User_6"]:
                if uk in item and safe_strip(item[uk]):
                    messages.append({"role":"user","content":safe_strip(item[uk])})
                    break
            for ak in ["AI","AI_1","AI_2","AI_3","AI_4","AI_5","AI_6"]:
                if ak in item and safe_strip(item[ak]):
                    messages.append({"role":"assistant","content":safe_strip(item[ak])})
                    break
    return messages

def parse_file_source(path: Path) -> Dict[str, Any]:
    """
    Parse the file path to determine:
    - data source type (model / vt_8k / arc)
    - metadata such as dia_model, function_name, topic, num

    This function relies on directory structure under `dialogues/`.
    """
    parts = path.as_posix().split("/")
    try:
        idx = parts.index("dialogues")
    except ValueError:
        return {}

    # vt_8k and arc
    if parts[idx+1] in ["vt_8k", "arc"]:
        filename = parts[-1]
        num = filename.replace("dialogue_","").replace(".json","")
        return {
            "is_vt": True,
            "source": parts[idx+1],
            "num": num
        }

    # original model data
    dia_model = parts[idx+1]
    function_name = parts[idx+2]
    filename = parts[-1]
    topic = "_".join(filename.split("_")[:-1])
    num = filename.split("_")[-1].replace(".json","")
    return {
        "is_vt": False,
        "source": "model",
        "dia_model": dia_model,
        "function_name": function_name,
        "topic": topic,
        "num": num
    }

def safe_parse_json(raw_dialogue: Any) -> Dict[str, Any]:
    if isinstance(raw_dialogue, dict):
        return raw_dialogue
    if isinstance(raw_dialogue, str):
        try:
            return json.loads(raw_dialogue)
        except json.JSONDecodeError:
            s = raw_dialogue.replace('\\"','"').replace("\n"," ")
            s = re.sub(r",\s*}", "}", s)
            s = re.sub(r",\s*]", "]", s)
            try:
                return json.loads(s)
            except:
                try:
                    return ast.literal_eval(raw_dialogue)
                except:
                    return {}
    return {}

# -------------------- LLM Call --------------------
def LLM_backend(api_key: str, messages: list, model_name: str):
    client = OpenAI(
        base_url="your url", #Please revise to your own base url
        api_key=api_key
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    content = resp.choices[0].message.content
    pt = getattr(resp.usage,"prompt_tokens",0)
    ct = getattr(resp.usage,"completion_tokens",0)
    return content, pt, ct

# -------------------- Output Idex --------------------
def get_next_gen_index(output_root: Path, gen_model: str) -> int:
    save_dir = output_root / gen_model
    if not save_dir.exists():
        return 0

    max_id = -1
    for p in save_dir.glob("gen_*.json"):
        try:
            idx = int(p.stem.replace("gen_", ""))
            max_id = max(max_id, idx)
        except:
            pass
    return max_id + 1

# -------------------- Check Generated Files --------------------
def collect_existing_files(output_root: Path, gen_model: str):
    existing = {"model": set(), "vt_8k": set(), "arc": set()}
    save_dir = output_root / gen_model
    if not save_dir.exists():
        return existing

    for p in save_dir.glob("gen_*.json"):
        try:
            obj = load_json(p)
            src = obj.get("source")
            if src == "model":
                key = f"{src}|{obj.get('dia_model')}|{obj.get('function_name')}|{obj.get('topic')}|{obj.get('num')}"
                existing["model"].add(key)
            elif src in ["vt_8k", "arc"]:
                key = f"{src}|{obj.get('num')}"
                existing[src].add(key)
        except:
            continue
    return existing

# -------------------- Collect Files --------------------
def collect_all_files(input_root: Path) -> list:
    return list(input_root.rglob("*.json"))

def collect_arc_files(input_root: Path) -> list:
    return [p for p in input_root.rglob("*.json") if "dialogues/arc" in p.as_posix()]

# -------------------- Single File Process --------------------
def process_single_file(file_path: Path, api_key: str, gen_model: str, output_root: Path, file_idx: int):
    data = load_json(file_path)
    info = parse_file_source(file_path)
    raw_dialogue = data.get("dialogue_response")
    dialogue = safe_parse_json(raw_dialogue)

    save_dir = output_root / gen_model
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"gen_{file_idx}.json"
    log_path = save_dir / "log.txt"

    if not dialogue:
        msg = f"❌ Unable to parse dialogue_response，skip: {file_path}"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return

    messages = dialogue_to_messages(dialogue, is_vt=info.get("is_vt", False))
    final_question = extract_final_question(dialogue)
    correct_answer = extract_key(dialogue, "Correct_Answer")

    if not final_question:
        msg = f"⚠️ Lose Final_Question: {file_path}"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return

    messages.append({"role":"user","content":final_question})
    model_output, ptok, ctok = LLM_backend(api_key, messages, gen_model)

    # Decide the content structure of json according to "source"
    if info["source"] == "model" or info["source"] == "vt_8k":
        output_obj = {
            "gen_model": gen_model,
            "source": info["source"],
            "dia_model": info.get("dia_model"),
            "function_name": info.get("function_name"),
            "topic": info.get("topic"),
            "num": info["num"],
            "evaluation_pipeline": extract_key(dialogue, "Evaluation_Pipeline"),
            "final_question": final_question,
            "model_answer": model_output,
            "correct_answer": correct_answer,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "messages": messages
        }
    elif info["source"] == "arc":
        output_obj = {
            "gen_model": gen_model,
            "source": "arc",
            "num": info["num"],
            "final_question": final_question,
            "model_answer": model_output,
            "correct_answer": correct_answer,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "messages": messages
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    msg = f"✅ saved to {save_path}"
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# -------------------- Main Process --------------------
def run_parallel_tests(api_key: str, gen_model: str, input_root: Path, output_root: Path, max_workers: int = 8):
    all_files = collect_all_files(input_root)
    existing_files = collect_existing_files(output_root, gen_model)

    # Build the list of files to be generated.
    pending_files = []
    for f in all_files:
        info = parse_file_source(f)
        if info.get("source") == "model":
            key = f"{info['source']}|{info['dia_model']}|{info['function_name']}|{info['topic']}|{info['num']}"
            if key not in existing_files["model"]:
                pending_files.append(f)
        elif info.get("source") in ["vt_8k", "arc"]:
            key = f"{info['source']}|{info['num']}"
            if key not in existing_files[info["source"]]:
                pending_files.append(f)

    total_files = len(all_files)
    print(f"Sum: {total_files}, Generated: {total_files - len(pending_files)}")
    print(f"To be generated: {len(pending_files)}")

    start_idx = get_next_gen_index(output_root, gen_model)
    print(f"From gen_{start_idx}.json to generate（model + vt_8k + ARC）")

    # Multiple Pipelines
    tasks = []
    current_idx = start_idx
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for f in pending_files:
            tasks.append(executor.submit(process_single_file, f, api_key, gen_model, output_root, current_idx))
            current_idx += 1
        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Processing files"):
            pass

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--gen_model", type=str, required=True)
    parser.add_argument("--input_root", type=str, default="dialogues")
    parser.add_argument("--output_root", type=str, default="generations")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    run_parallel_tests(
        api_key=args.api_key,
        gen_model=args.gen_model,
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        max_workers=args.num_workers
    )
