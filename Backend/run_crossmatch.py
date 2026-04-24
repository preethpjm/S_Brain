"""
run_crossmatch.py — Run ONLY the semantic cross-matcher
Use this when sbrain_core.py has already finished and you just need
to (re)build the crossmatch index without reprocessing all the PDFs.

Usage:
    python run_crossmatch.py
"""

import os
import json
from pathlib import Path
from sbrain_crossmatch import SemanticCrossMatcher

# ── Paths — adjust if your layout differs ─────────────────────────────────────
OUTPUT_DIR   = "./output"          # where sbrain_core.py saved its JSONs
CROSSMATCH_DIR = "./output/crossmatch"

INPUT_DIRS = {
    "S1000D": "Inputs/S1000D",
    "S2000M": "Inputs/S2000M",
    "S3000L": "Inputs/S3000L",
}
SX000I_DIR = "Inputs/SX000i"      # set to None if you don't have it

# ── Model — uses your existing local model, no internet needed ────────────────
# Points directly at the local folder sbrain_core already used successfully.
MODEL_PATH = "./models/all-MiniLM-L6-v2"   # relative to Backend folder


# ── Collect extracted JSONs sbrain_core already produced ──────────────────────
extracted_jsons = {}
for std in ["S1000D", "S2000M", "S3000L", "SX000i"]:
    p = Path(OUTPUT_DIR) / f"{std}_extracted.json"
    if p.exists():
        extracted_jsons[std] = str(p)
        print(f"  Found: {p}")
    else:
        print(f"  Missing (skipping): {p}")

# ── Load XMI models that core already parsed ──────────────────────────────────
# We re-parse XMI here since we don't have core's in-memory objects.
# This is fast (seconds) — XMIs are small compared to PDFs.
from sbrain_core import MultiStandardXSDParser
print("\n  Loading XSD/XMI schemas...")
xsd_parser = MultiStandardXSDParser()
all_dirs = dict(INPUT_DIRS)
if SX000I_DIR and Path(SX000I_DIR).exists():
    all_dirs["SX000i"] = SX000I_DIR
for std, path in all_dirs.items():
    if Path(path).exists():
        xsd_parser.load_standard(std, path)

xmi_models = {
    std: xsd_parser.xmi_models.get(std, {})
    for std in all_dirs
}

# ── Ontology merged JSON ───────────────────────────────────────────────────────
ontology_json = str(Path(OUTPUT_DIR) / "ontology_merged.json")
if not Path(ontology_json).exists():
    print(f"  Warning: ontology_merged.json not found at {ontology_json}")
    ontology_json = None

# ── Build crossmatch ───────────────────────────────────────────────────────────
print(f"\n  Building Semantic Cross-Matcher → {CROSSMATCH_DIR}")
print(f"  Using model: {MODEL_PATH}\n")

standard_dirs = {std: path for std, path in INPUT_DIRS.items() if Path(path).exists()}
sx_dir = SX000I_DIR if SX000I_DIR and Path(SX000I_DIR).exists() else None

matcher = SemanticCrossMatcher(
    output_dir=CROSSMATCH_DIR,
    model_name=MODEL_PATH,       # local path, no download needed
)
matcher.build(
    standard_dirs=standard_dirs,
    extracted_jsons=extracted_jsons,
    SX000i_dir=sx_dir,
    xmi_models=xmi_models,
    ontology_json=ontology_json,
)

print(f"\n  Done. Crossmatch index saved to: {CROSSMATCH_DIR}")
print(f"  Links: {len(matcher.links)}")