"""
SteerOps Deep Evaluation v2 — Tamil Control Experiment
========================================================
N=40 evaluation prompts, bootstrap 95% CIs, crash-safe checkpointing.
Designed for Kaggle T4 GPU (30h/week free, 12h sessions).

Two genuine research questions:
  Q2: Is Tamil degradation vector-origin-specific or circuit-level?
  Q5: Are Dravidian/Indo-Aryan honorifics in orthogonal subspaces?

Phases (each saves a checkpoint, script resumes from last completed phase):
  1. Model load + scan + compute 4 vectors (hindi, bengali, tamil_native, hindi_for_cross)
  2. Hindi N=40 sweep
  3. Bengali N=40 sweep
  4. Tamil with Hindi vector (cross-lingual condition)
  5. Tamil with Tamil-native vector (native condition)
  6. Layer responsiveness (4 conditions x 6 layers x 10 prompts)
  7. Cross-vector analysis + final merged report

Usage (Kaggle):
  !pip install transformers accelerate bitsandbytes sentence-transformers scipy scikit-learn loguru aiosqlite textblob
  import sys; sys.path.append('/kaggle/input/steerops-backend/backend')
  !python steerops_deep_eval.py
"""

import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.loader import ModelManager
from app.core.engine import SteeringEngine
from app.core.scanner import LayerScanner
from app.core.vector_calculator import VectorCalculator
from app.core.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("steerops_deep_eval")


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "sarvamai/sarvam-1"
MAX_TOKENS = 60
STRENGTH_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_BOOTSTRAP = 2000
CI_ALPHA = 0.05
LAYER_PROBE_POSITIONS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]

# Auto-detect environment for checkpoint path
if os.path.exists("/kaggle/working"):
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
elif os.path.exists("/content"):
    CHECKPOINT_DIR = "/content/drive/MyDrive/steerops_checkpoints"
else:
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")


# ============================================================
# Load evaluation prompts
# ============================================================
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "data", "deep_eval_prompts.json")
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    EVAL_PROMPTS = json.load(f)

HINDI_PROMPTS = EVAL_PROMPTS["hindi"]
BENGALI_PROMPTS = EVAL_PROMPTS["bengali"]
TAMIL_PROMPTS = EVAL_PROMPTS["tamil"]

logger.info(f"Loaded prompts: Hindi={len(HINDI_PROMPTS)}, Bengali={len(BENGALI_PROMPTS)}, Tamil={len(TAMIL_PROMPTS)}")


# ============================================================
# Honorific pattern matchers (reused from v1)
# ============================================================
POLITE_PATTERNS = {
    "आप": re.compile(r"आप"), "जी": re.compile(r"जी"), "कृपया": re.compile(r"कृपया"),
    "धन्यवाद": re.compile(r"धन्यवाद"), "महोदय": re.compile(r"महोदय"), "श्रीमान": re.compile(r"श्रीमान"),
    "नमस्कार": re.compile(r"नमस्कार"), "आदरणीय": re.compile(r"आदरणीय"),
    "আপনি": re.compile(r"আপনি"), "দয়া করে": re.compile(r"দয়া করে"),
    "ধন্যবাদ": re.compile(r"ধন্যবাদ"), "মহাশয়": re.compile(r"মহাশয়"),
    "নমস্কার_bn": re.compile(r"নমস্কার"), "অনুগ্রহ": re.compile(r"অনুগ্রহ"),
    "நீங்கள்": re.compile(r"நீங்கள்"), "தயவுசெய்து": re.compile(r"தயவுசெய்து"),
    "நன்றி": re.compile(r"நன்றி"), "ஐயா": re.compile(r"ஐயா"),
    "வணக்கம்": re.compile(r"வணக்கம்"), "பணிவான": re.compile(r"பணிவான"),
    "aap": re.compile(r"\baap\b", re.IGNORECASE), "ji": re.compile(r"\bji\b", re.IGNORECASE),
    "please": re.compile(r"\bplease\b", re.IGNORECASE), "thank": re.compile(r"\bthank\b", re.IGNORECASE),
    "sir": re.compile(r"\bsir\b", re.IGNORECASE),
}

RUDE_PATTERNS = {
    "तू": re.compile(r"तू"), "तुम": re.compile(r"तुम"), "अबे": re.compile(r"अबे"),
    "ओए": re.compile(r"ओए"), "বোকা": re.compile(r"বোকা"),
    "তুই": re.compile(r"তুই"), "তোর": re.compile(r"তোর"),
    "நீ": re.compile(r"நீ(?!ங்கள்)"), "போடா": re.compile(r"போடா"),
    "முட்டாள்": re.compile(r"முட்டாள்"),
}

def count_honorifics(text):
    polite = sum(len(p.findall(text)) for p in POLITE_PATTERNS.values())
    rude = sum(len(p.findall(text)) for p in RUDE_PATTERNS.values())
    total = polite + rude
    return {"polite_count": polite, "rude_count": rude, "total_markers": total,
            "polite_ratio": round(polite / total, 4) if total > 0 else 0.5}

def measure_token_fertility(text, tokenizer):
    words = text.split()
    if not words:
        return 0.0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return round(len(token_ids) / len(words), 4)


# ============================================================
# Statistics: NaN-safe + Bootstrap CIs
# ============================================================
def safe_mean(vals):
    clean = [v for v in vals if v is not None and not math.isnan(v) and not math.isinf(v)]
    return float(np.mean(clean)) if clean else 0.0

def safe_std(vals):
    clean = [v for v in vals if v is not None and not math.isnan(v) and not math.isinf(v)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0

def cohens_d(baseline, steered):
    b = [v for v in baseline if v is not None and not math.isnan(v)]
    s = [v for v in steered if v is not None and not math.isnan(v)]
    if len(b) < 2 or len(s) < 2:
        return 0.0
    mb, ms = np.mean(b), np.mean(s)
    sb, ss = np.std(b, ddof=1), np.std(s, ddof=1)
    pooled = math.sqrt(((len(b)-1)*sb**2 + (len(s)-1)*ss**2) / (len(b)+len(s)-2))
    return round(float((ms - mb) / pooled), 4) if pooled > 1e-10 else 0.0

def paired_t_test(baseline, steered):
    b = [v for v in baseline if v is not None and not math.isnan(v)]
    s = [v for v in steered if v is not None and not math.isnan(v)]
    n = min(len(b), len(s))
    if n < 3:
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False}
    t, p = scipy_stats.ttest_rel(b[:n], s[:n])
    if math.isnan(p):
        p = 1.0
    return {"t_stat": round(float(t), 4), "p_value": round(float(p), 4), "significant": p < 0.05}

def bootstrap_cohens_d_ci(baseline, steered, n_bootstrap=N_BOOTSTRAP, alpha=CI_ALPHA):
    """BCa-lite: percentile bootstrap CI on Cohen's d."""
    b = np.array([v for v in baseline if v is not None and not math.isnan(v)])
    s = np.array([v for v in steered if v is not None and not math.isnan(v)])
    n = min(len(b), len(s))
    if n < 5:
        return {"ci_lower": 0.0, "ci_upper": 0.0, "ci_width": 0.0}
    b, s = b[:n], s[:n]
    d_boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        d_boot.append(cohens_d(b[idx].tolist(), s[idx].tolist()))
    d_boot.sort()
    lo = d_boot[int(alpha / 2 * n_bootstrap)]
    hi = d_boot[int((1 - alpha / 2) * n_bootstrap)]
    return {"ci_lower": round(lo, 4), "ci_upper": round(hi, 4), "ci_width": round(hi - lo, 4)}


# ============================================================
# Checkpoint I/O
# ============================================================
def save_checkpoint(name, data):
    path = os.path.join(CHECKPOINT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"  Checkpoint saved: {path}")

def load_checkpoint(name):
    path = os.path.join(CHECKPOINT_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"  Checkpoint loaded: {path}")
        return data
    return None

def save_vectors(vectors_dict):
    path = os.path.join(CHECKPOINT_DIR, "vectors.pt")
    torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in vectors_dict.items()}, path)
    logger.info(f"  Vectors saved: {path}")

def load_vectors():
    path = os.path.join(CHECKPOINT_DIR, "vectors.pt")
    if os.path.exists(path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"  Vectors loaded: {path}")
        return data
    return None


# ============================================================
# Phase 1: Model + Vectors
# ============================================================
def phase1_setup():
    logger.info("=" * 80)
    logger.info("PHASE 1: Model load + scan + vector computation")
    logger.info("=" * 80)

    cached = load_vectors()
    if cached and all(k in cached for k in ["hindi_vec", "bengali_vec", "tamil_native_vec", "hindi_for_tamil_vec", "scan_results", "routing_layer"]):
        logger.info("  All vectors found in cache. Skipping Phase 1.")
        return cached

    t0 = time.time()
    mm = ModelManager.get_instance()
    mm.load(MODEL_NAME, device_preference="auto", quantize=True)
    logger.info(f"  Model loaded in {time.time()-t0:.1f}s")

    # Sarvam-1 is a base model — disable chat template
    if hasattr(mm.tokenizer, 'chat_template') and mm.tokenizer.chat_template:
        logger.info("  Chat template FOUND — disabling for base model test")
        mm.tokenizer.chat_template = None

    scanner = LayerScanner(mm)
    t1 = time.time()
    scan_results = scanner.scan()
    logger.info(f"  Scan completed in {time.time()-t1:.1f}s")

    calculator = VectorCalculator(max_prompts=50)
    n_layers = mm.num_layers or 28

    # Route to best layer — match v1 logic: sort by confidence, take highest
    candidates = [r for r in scan_results if r.get("category") == "knowledge_retrieval"]
    candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    if candidates:
        routing_layer = candidates[0].get("layer_index", 11)
    else:
        routing_layer = 11  # fallback
    logger.info(f"  Routing layer: {routing_layer} (from {len(candidates)} candidates)")

    # Compute 4 vectors
    vectors = {}
    for concept_id, label in [("hindi_honorific", "hindi_vec"), ("bengali_honorific", "bengali_vec"),
                               ("tamil_honorific", "tamil_native_vec")]:
        t = time.time()
        vec_result = calculator.compute_vector(mm.model, mm.tokenizer, concept_id, routing_layer)
        vec = vec_result.get("direction_vector")  # This is a list, not a tensor
        mag = vec_result.get("magnitude", 0.0)
        dim = vec_result.get("dimension", 0)
        vectors[label] = vec
        vectors[f"{label}_magnitude"] = mag
        logger.info(f"  {label}: dim={dim}, mag={mag:.4f} ({time.time()-t:.1f}s)")

    # The Hindi vector that will be applied cross-lingually to Tamil prompts
    vectors["hindi_for_tamil_vec"] = list(vectors["hindi_vec"]) if vectors["hindi_vec"] is not None else None
    vectors["hindi_for_tamil_vec_magnitude"] = vectors.get("hindi_vec_magnitude", 0.0)

    vectors["scan_results"] = scan_results
    vectors["routing_layer"] = routing_layer
    vectors["n_layers"] = n_layers
    vectors["model_load_time"] = time.time() - t0

    save_vectors(vectors)
    return vectors


# ============================================================
# Strength sweep runner (used by Phases 2-5)
# ============================================================
def run_strength_sweep(mm, engine, evaluator, direction_vector, prompts, concept_label,
                       routing_layer, strengths=STRENGTH_SWEEP):
    """Run a full strength sweep for one concept/vector/prompt combination."""
    sweep_results = []

    for strength in strengths:
        t0 = time.time()
        steering_configs = [{
            "layer": routing_layer,
            "strength": strength,
            "direction_vector": direction_vector,
        }]

        results = evaluator.evaluate(
            model=mm.model, tokenizer=mm.tokenizer, engine=engine,
            test_prompts=prompts, steering_configs=steering_configs,
            max_tokens=MAX_TOKENS, target_concept=concept_label,
        )

        # Extract per-prompt metrics
        baseline_sims, steered_sims, shifts, perps = [], [], [], []
        polite_ratios, fertilities = [], []

        for comp in results.get("comparisons", []):
            m = comp.get("metrics", {})
            baseline_sims.append(m.get("baseline_concept_alignment", 0.0))
            steered_sims.append(m.get("steered_concept_alignment", 0.0))
            shifts.append(m.get("semantic_shift", 0.0))
            perps.append(m.get("perplexity_ratio", 1.0))

            steered_text = comp.get("steered", "")
            hon = count_honorifics(steered_text)
            polite_ratios.append(hon["polite_ratio"])
            fertilities.append(measure_token_fertility(steered_text, mm.tokenizer))

        # Statistics
        d = cohens_d(baseline_sims, steered_sims)
        t_test = paired_t_test(baseline_sims, steered_sims)
        ci = bootstrap_cohens_d_ci(baseline_sims, steered_sims)
        deltas = [s - b for b, s in zip(baseline_sims, steered_sims)]

        entry = {
            "strength": strength,
            "concept_alignment_delta": {"mean": round(safe_mean(deltas), 4), "std": round(safe_std(deltas), 4)},
            "semantic_shift": {"mean": round(safe_mean(shifts), 4), "std": round(safe_std(shifts), 4)},
            "perplexity_ratio": {"mean": round(safe_mean(perps), 4)},
            "statistical_test": {
                "cohens_d": d,
                "cohens_d_ci_95": ci,
                "paired_t_test": {"p_value": t_test["p_value"], "significant": t_test["significant"]},
            },
            "honorific_metrics": {
                "steered_polite_ratio": {"mean": round(safe_mean(polite_ratios), 4)},
                "token_fertility_steered": {"mean": round(safe_mean(fertilities), 4)},
            },
            "n_prompts": len(prompts),
            "eval_time_s": round(time.time() - t0, 1),
        }
        sweep_results.append(entry)

        sig = "***" if t_test["significant"] else "   "
        logger.info(f"    α={strength:.1f} | d={d:+.4f} [{ci['ci_lower']:+.4f}, {ci['ci_upper']:+.4f}] | "
                     f"p={t_test['p_value']:.4f} {sig} | PPL={safe_mean(perps):.4f} | "
                     f"polite={safe_mean(polite_ratios):.2f} | {time.time()-t0:.0f}s")

    return sweep_results


# ============================================================
# Phases 2-5: Per-concept sweeps
# ============================================================
def run_concept_phase(phase_name, concept_label, direction_vector, prompts, vectors_data):
    logger.info("=" * 80)
    logger.info(f"PHASE: {phase_name} (N={len(prompts)})")
    logger.info("=" * 80)

    cached = load_checkpoint(phase_name)
    if cached:
        logger.info(f"  Checkpoint found for {phase_name}. Skipping.")
        return cached

    mm = ModelManager.get_instance()
    if not mm.loaded:
        mm.load(MODEL_NAME, device_preference="auto", quantize=True)
    engine = SteeringEngine.get_instance(mm)
    evaluator = Evaluator()
    routing_layer = vectors_data["routing_layer"]

    t0 = time.time()
    sweep = run_strength_sweep(mm, engine, evaluator, direction_vector, prompts,
                                concept_label, routing_layer)

    result = {
        "concept": concept_label,
        "phase": phase_name,
        "routing_layer": routing_layer,
        "n_prompts": len(prompts),
        "sweep": sweep,
        "total_time_s": round(time.time() - t0, 1),
    }
    save_checkpoint(phase_name, result)
    return result


# ============================================================
# Phase 6: Layer responsiveness
# ============================================================
def phase6_layer_responsiveness(vectors_data):
    logger.info("=" * 80)
    logger.info("PHASE 6: Layer Responsiveness")
    logger.info("=" * 80)

    cached = load_checkpoint("layer_responsiveness")
    if cached:
        logger.info("  Checkpoint found. Skipping.")
        return cached

    mm = ModelManager.get_instance()
    if not mm.loaded:
        mm.load(MODEL_NAME, device_preference="auto", quantize=True)
    engine = SteeringEngine.get_instance(mm)
    evaluator = Evaluator()
    calculator = VectorCalculator(max_prompts=50)
    n_layers = vectors_data.get("n_layers", 28)

    probe_layers = sorted(set(min(int(pos * n_layers), n_layers - 1) for pos in LAYER_PROBE_POSITIONS))

    all_results = {}
    conditions = [
        ("hindi_honorific", "hindi_vec", HINDI_PROMPTS[:10]),
        ("bengali_honorific", "bengali_vec", BENGALI_PROMPTS[:10]),
        ("tamil_honorific", "tamil_native_vec", TAMIL_PROMPTS[:10]),
        ("tamil_cross", "hindi_for_tamil_vec", TAMIL_PROMPTS[:10]),
    ]

    for concept_id, vec_key, prompts in conditions:
        logger.info(f"  Layer scan: {concept_id} ({vec_key})")
        layer_results = []
        for layer_idx in probe_layers:
            try:
                # Recompute vector at this specific layer for native conditions
                if vec_key in ("hindi_vec", "bengali_vec", "tamil_native_vec"):
                    real_concept = concept_id if concept_id != "tamil_cross" else "tamil_honorific"
                    vec_result = calculator.compute_vector(mm.model, mm.tokenizer, real_concept, layer_idx)
                    vec = vec_result.get("direction_vector")
                else:
                    # Cross-lingual: compute hindi vector at this layer
                    vec_result = calculator.compute_vector(mm.model, mm.tokenizer, "hindi_honorific", layer_idx)
                    vec = vec_result.get("direction_vector")

                if vec is None:
                    continue

                steering_configs = [{"layer": layer_idx, "strength": 1.5, "direction_vector": vec}]
                results = evaluator.evaluate(
                    model=mm.model, tokenizer=mm.tokenizer, engine=engine,
                    test_prompts=prompts, steering_configs=steering_configs,
                    max_tokens=MAX_TOKENS, target_concept=concept_id,
                )
                s = [c["metrics"].get("semantic_shift", 0.0) for c in results.get("comparisons", [])]
                p = [c["metrics"].get("perplexity_ratio", 1.0) for c in results.get("comparisons", [])]
                entry = {
                    "layer": layer_idx, "relative_depth": round(layer_idx / n_layers, 2),
                    "semantic_shift": round(safe_mean(s), 4), "perplexity_ratio": round(safe_mean(p), 4),
                }
                layer_results.append(entry)
                logger.info(f"    L{layer_idx:2d} ({entry['relative_depth']:.0%}) | "
                             f"shift={entry['semantic_shift']:.4f} | PPL={entry['perplexity_ratio']:.4f}")
            except Exception as e:
                logger.warning(f"    L{layer_idx}: failed — {e}")

        all_results[f"{concept_id}__{vec_key}"] = layer_results

    save_checkpoint("layer_responsiveness", all_results)
    return all_results


# ============================================================
# Phase 7: Cross-vector analysis + report
# ============================================================
def phase7_analysis(vectors_data, hindi_res, bengali_res, tamil_cross_res, tamil_native_res, layer_res):
    logger.info("=" * 80)
    logger.info("PHASE 7: Cross-vector analysis + final report")
    logger.info("=" * 80)

    # --- Q5: Subspace orthogonality ---
    hindi_vec = vectors_data.get("hindi_vec")
    bengali_vec = vectors_data.get("bengali_vec")
    tamil_vec = vectors_data.get("tamil_native_vec")

    # Convert lists to tensors for cosine similarity
    def to_tensor(v):
        if v is None:
            return None
        if isinstance(v, list):
            return torch.tensor(v, dtype=torch.float32)
        return v.float()

    hindi_vec_t = to_tensor(hindi_vec)
    bengali_vec_t = to_tensor(bengali_vec)
    tamil_vec_t = to_tensor(tamil_vec)

    cross_vector_analysis = {}
    if hindi_vec_t is not None and tamil_vec_t is not None:
        cos_ht = torch.cosine_similarity(hindi_vec_t.unsqueeze(0), tamil_vec_t.unsqueeze(0)).item()
        cross_vector_analysis["cos_hindi_tamil"] = round(cos_ht, 4)
        logger.info(f"  Q5: cos(hindi_vec, tamil_native_vec) = {cos_ht:.4f}")

    if bengali_vec_t is not None and tamil_vec_t is not None:
        cos_bt = torch.cosine_similarity(bengali_vec_t.unsqueeze(0), tamil_vec_t.unsqueeze(0)).item()
        cross_vector_analysis["cos_bengali_tamil"] = round(cos_bt, 4)
        logger.info(f"  Q5: cos(bengali_vec, tamil_native_vec) = {cos_bt:.4f}")

    if hindi_vec_t is not None and bengali_vec_t is not None:
        cos_hb = torch.cosine_similarity(hindi_vec_t.unsqueeze(0), bengali_vec_t.unsqueeze(0)).item()
        cross_vector_analysis["cos_hindi_bengali"] = round(cos_hb, 4)
        logger.info(f"  Q5: cos(hindi_vec, bengali_vec) = {cos_hb:.4f}")

    # --- Q2: Tamil control comparison ---
    tamil_control = {"vector_origin_hypothesis": None}

    # Find best strength for each Tamil condition (lowest PPL that's significant)
    def best_entry(sweep, max_ppl=5.0):
        sig = [e for e in sweep if e["statistical_test"]["paired_t_test"]["significant"] and e["perplexity_ratio"]["mean"] < max_ppl]
        if sig:
            return min(sig, key=lambda e: e["perplexity_ratio"]["mean"])
        return min(sweep, key=lambda e: e["perplexity_ratio"]["mean"]) if sweep else None

    best_cross = best_entry(tamil_cross_res.get("sweep", []))
    best_native = best_entry(tamil_native_res.get("sweep", []))

    if best_cross and best_native:
        cross_ppl = best_cross["perplexity_ratio"]["mean"]
        native_ppl = best_native["perplexity_ratio"]["mean"]
        tamil_control["cross_lingual_best_ppl"] = cross_ppl
        tamil_control["native_best_ppl"] = native_ppl
        tamil_control["ppl_ratio_cross_vs_native"] = round(cross_ppl / native_ppl, 4) if native_ppl > 0 else float("inf")

        if native_ppl < 2.0 and cross_ppl > 5.0:
            tamil_control["vector_origin_hypothesis"] = "CONFIRMED"
            tamil_control["conclusion"] = ("Tamil degradation is vector-origin-specific. "
                "The Tamil-native vector steers without catastrophic PPL, confirming that "
                "the L4/L8 circuit CAN process Tamil honorifics. The Indo-Aryan vector "
                "encodes morphological transformations incompatible with Tamil's agglutinative system.")
        elif native_ppl > 5.0 and cross_ppl > 5.0:
            tamil_control["vector_origin_hypothesis"] = "REJECTED"
            tamil_control["conclusion"] = ("Tamil degradation is circuit-level, not vector-origin-specific. "
                "Both native and cross-lingual vectors produce catastrophic PPL, indicating that "
                "the shared L4/L8 circuit cannot tolerate perturbation for Tamil honorific steering. "
                "Sarvam-1's Tamil representation at this depth is too fragile or too sparse.")
        else:
            tamil_control["vector_origin_hypothesis"] = "INCONCLUSIVE"
            tamil_control["conclusion"] = (f"Results are ambiguous. Native PPL={native_ppl:.2f}, "
                f"Cross PPL={cross_ppl:.2f}. Neither clearly confirms nor rejects the hypothesis.")

        logger.info(f"  Q2 Verdict: {tamil_control['vector_origin_hypothesis']}")
        logger.info(f"  Q2: {tamil_control['conclusion']}")

    # --- Q5 verdict ---
    q5_verdict = {}
    cos_ht = cross_vector_analysis.get("cos_hindi_tamil", None)
    cos_hb = cross_vector_analysis.get("cos_hindi_bengali", None)
    if cos_ht is not None:
        if abs(cos_ht) < 0.1:
            q5_verdict["subspace_orthogonality"] = "CONFIRMED"
            q5_verdict["conclusion"] = (f"Hindi and Tamil honorific vectors are near-orthogonal "
                f"(cos={cos_ht:.4f} < 0.1), confirming that Indo-Aryan and Dravidian honorific "
                f"systems occupy distinct subspaces within the shared circuit.")
        elif abs(cos_ht) > 0.3:
            q5_verdict["subspace_orthogonality"] = "REJECTED"
            q5_verdict["conclusion"] = (f"Hindi and Tamil vectors share significant subspace overlap "
                f"(cos={cos_ht:.4f} > 0.3). The degradation is not explained by orthogonality.")
        else:
            q5_verdict["subspace_orthogonality"] = "PARTIAL"
            q5_verdict["conclusion"] = (f"Hindi-Tamil alignment is moderate (cos={cos_ht:.4f}). "
                f"Partial subspace separation exists but is not decisive.")

        if cos_hb is not None:
            q5_verdict["indo_aryan_alignment"] = (f"Hindi-Bengali cos={cos_hb:.4f} vs Hindi-Tamil cos={cos_ht:.4f}. "
                f"{'Indo-Aryan languages are more aligned with each other than with Dravidian.' if abs(cos_hb) > abs(cos_ht) else 'No clear family-level separation.'}")

        logger.info(f"  Q5 Verdict: {q5_verdict.get('subspace_orthogonality', 'N/A')}")

    # --- Build final report ---
    report = {
        "title": "SteerOps Deep Evaluation v2 — Sarvam-1 Tamil Control Experiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_NAME,
        "evaluation_config": {
            "n_eval_prompts": len(HINDI_PROMPTS),
            "n_bootstrap": N_BOOTSTRAP,
            "ci_alpha": CI_ALPHA,
            "strengths": STRENGTH_SWEEP,
        },
        "vector_magnitudes": {
            "hindi": vectors_data.get("hindi_vec_magnitude", 0.0),
            "bengali": vectors_data.get("bengali_vec_magnitude", 0.0),
            "tamil_native": vectors_data.get("tamil_native_vec_magnitude", 0.0),
        },
        "results": {
            "hindi_honorific": hindi_res,
            "bengali_honorific": bengali_res,
            "tamil_cross_lingual": tamil_cross_res,
            "tamil_native": tamil_native_res,
        },
        "layer_responsiveness": layer_res,
        "research_questions": {
            "Q2_vector_origin": tamil_control,
            "Q5_subspace_orthogonality": {
                "cross_vector_cosine_similarities": cross_vector_analysis,
                **q5_verdict,
            },
        },
    }

    # Save final report
    report_path = os.path.join(CHECKPOINT_DIR, "steerops_v2_deep_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"  Final report saved: {report_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("  DEEP EVALUATION COMPLETE — SUMMARY")
    print("=" * 80)
    print(f"  Model: {MODEL_NAME}")
    print(f"  N per condition: {len(HINDI_PROMPTS)}")
    print(f"  Vector magnitudes: Hindi={vectors_data.get('hindi_vec_magnitude',0):.2f}, "
          f"Bengali={vectors_data.get('bengali_vec_magnitude',0):.2f}, "
          f"Tamil(native)={vectors_data.get('tamil_native_vec_magnitude',0):.2f}")
    print(f"\n  Q2 (Vector Origin): {tamil_control.get('vector_origin_hypothesis', 'N/A')}")
    print(f"     {tamil_control.get('conclusion', 'N/A')}")
    print(f"\n  Q5 (Subspace Orthogonality): {q5_verdict.get('subspace_orthogonality', 'N/A')}")
    print(f"     {q5_verdict.get('conclusion', 'N/A')}")
    print("=" * 80)

    return report


# ============================================================
# Main
# ============================================================
def main():
    overall_t0 = time.time()
    print("=" * 80)
    print("  SteerOps Deep Evaluation v2 — Tamil Control Experiment")
    print(f"  N={len(HINDI_PROMPTS)} | Bootstrap CIs | Kaggle-optimized")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print("=" * 80)

    # Phase 1
    vectors_data = phase1_setup()
    routing_layer = vectors_data["routing_layer"]

    # Ensure model is loaded for subsequent phases
    mm = ModelManager.get_instance()
    if not mm.loaded:
        mm.load(MODEL_NAME, device_preference="auto", quantize=True)

    # Vectors are lists (not tensors) — no GPU transfer needed.
    # The evaluator/engine handles list-to-tensor conversion internally.

    # Phase 2: Hindi
    hindi_res = run_concept_phase("hindi_n40", "hindi_honorific",
                                   vectors_data["hindi_vec"], HINDI_PROMPTS, vectors_data)

    # Phase 3: Bengali
    bengali_res = run_concept_phase("bengali_n40", "bengali_honorific",
                                     vectors_data["bengali_vec"], BENGALI_PROMPTS, vectors_data)

    # Phase 4: Tamil with Hindi vector (cross-lingual condition)
    tamil_cross_res = run_concept_phase("tamil_cross_n40", "tamil_honorific",
                                         vectors_data["hindi_for_tamil_vec"], TAMIL_PROMPTS, vectors_data)

    # Phase 5: Tamil with Tamil-native vector
    tamil_native_res = run_concept_phase("tamil_native_n40", "tamil_honorific",
                                          vectors_data["tamil_native_vec"], TAMIL_PROMPTS, vectors_data)

    # Phase 6: Layer responsiveness
    layer_res = phase6_layer_responsiveness(vectors_data)

    # Phase 7: Analysis + report
    report = phase7_analysis(vectors_data, hindi_res, bengali_res,
                              tamil_cross_res, tamil_native_res, layer_res)

    elapsed = time.time() - overall_t0
    logger.info(f"Total evaluation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
