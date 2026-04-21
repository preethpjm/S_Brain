"""
S-Series Unified Data Platform — Backend v2.1
Fixes:
  - _load_extracted() now correctly parses DocumentResult JSON structure
    (entities[], sections[], schemas[], tables[] — not flat tag lists)
  - SX000i casing preserved throughout (lowercase 'i', never uppercased)
  - Graph node IDs built from XSD schema element names to match crossmatch source_tag
  - Fallback node generation from entities[] and sections[] when schemas empty
  - /crossmatch-groups union-find works with real node IDs
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
from pathlib import Path
import shutil
import json
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from sbrain_translator import OntologyDrivenTranslator
from sbrain_learning_memory import SBrainLearningMemory

app = FastAPI(title="S-Series Unified Data Platform v2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path("./output")
INPUTS_DIR     = Path("./Inputs")
CROSSMATCH_DIR = OUTPUT_DIR / "crossmatch"

# SX000i must keep lowercase 'i' — NEVER call .upper() on this list
STANDARDS = ["S1000D", "S2000M", "S3000L", "SX000i"]
_STD_CANON: Dict[str, str] = {s.upper(): s for s in STANDARDS}

def _canon_std(s: str) -> str:
    """Return canonical standard name (correct casing) from any input."""
    return _STD_CANON.get(s.upper(), s)

STD_COLORS = {
    "S1000D":  "#3B82F6",
    "S2000M":  "#22C55E",
    "S3000L":  "#F97316",
    "SX000i": "#A855F7",
}

_node_overrides: Dict[str, dict] = {}

translator = OntologyDrivenTranslator()
memory     = SBrainLearningMemory()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_crossmatch_links() -> list:
    for candidate in [CROSSMATCH_DIR / "concept_links.json",
                      OUTPUT_DIR / "concept_links.json"]:
        raw = _load_json(candidate)
        if raw is not None:
            for lnk in raw:
                lnk["source_std"] = _canon_std(lnk.get("source_std", ""))
                lnk["target_std"] = _canon_std(lnk.get("target_std", ""))
            return raw
    return []


def _parse_document_result(raw: Any, std: str) -> List[dict]:
    """
    Unpack a DocumentResult JSON (from sbrain_core.py save_all()) into flat node records.

    DocumentResult has:
      standard, pdf_path, extracted_at, total_pages,
      sections:  list of section dicts
      entities:  list of {name, entity_type, description, page, confidence, ...}
      rules:     list of rule dicts
      schemas:   list of {element, standard, children, source_file, source}
      tables:    list of table dicts
      metadata:  {hierarchy: [...], ...}

    Priority order for node generation:
      1. schemas[].element  (XSD names — these MATCH crossmatch source_tag values)
      2. entities[].name    (extracted concepts)
      3. sections[].title   (fallback only when nothing else present)
    """
    if not raw:
        return []
    if isinstance(raw, list):
        nodes = []
        for item in raw:
            nodes.extend(_parse_document_result(item, std))
        return nodes
    if not isinstance(raw, dict):
        return []

    std = _canon_std(raw.get("standard", std))
    nodes: List[dict] = []
    seen: set = set()

    # 1. XSD schema elements
    for sch in raw.get("schemas", []):
        tag = (sch.get("element") or sch.get("entity_name") or "").strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        nodes.append({
            "tag_name":    tag,
            "standard":    std,
            "source":      "xsd_schema",
            "children":    sch.get("children", []),
            "source_file": sch.get("source_file", ""),
            "description": f"XSD element — {sch.get('source_file', '')}",
        })

    # 2. Extracted entities
    for ent in raw.get("entities", []):
        name = (ent.get("name") or "").strip()
        if not name or name in seen or len(name) < 3 or name.isdigit():
            continue
        seen.add(name)
        nodes.append({
            "tag_name":    name,
            "standard":    std,
            "source":      "entity",
            "entity_type": ent.get("entity_type", ""),
            "description": (ent.get("description") or "")[:300],
            "page":        ent.get("page", 0),
            "confidence":  ent.get("confidence", 1.0),
        })

    # 3. Section titles (only if we got nothing from schemas/entities)
    if not nodes:
        for sec in raw.get("sections", []):
            title = (sec.get("title") or "").strip()
            if not title or title == "[PREAMBLE]" or title in seen or len(title) < 3:
                continue
            seen.add(title)
            nodes.append({
                "tag_name":    title,
                "standard":    std,
                "source":      "section",
                "description": (sec.get("content_preview") or "")[:200],
                "page":        sec.get("page_start", 0),
            })

    return nodes


def _load_all_nodes() -> Dict[str, List[dict]]:
    result: Dict[str, List[dict]] = {}
    for std in STANDARDS:
        p = OUTPUT_DIR / f"{std}_extracted.json"
        raw = _load_json(p)
        result[std] = _parse_document_result(raw, std)
    return result


def _load_ontology() -> dict:
    return _load_json(OUTPUT_DIR / "ontology_merged.json") or {}


def _build_graph(active_standards: Optional[List[str]] = None) -> dict:
    stds = [_canon_std(s) for s in (active_standards or STANDARDS)]
    all_nodes_by_std = _load_all_nodes()
    links = _load_crossmatch_links()

    nodes: List[dict] = []
    node_id_set: set = set()

    for std in stds:
        for item in all_nodes_by_std.get(std, []):
            tag = item.get("tag_name", "")
            if not tag:
                continue
            nid = f"{std}::{tag}"
            if nid in node_id_set:
                continue
            node_id_set.add(nid)
            merged = {**item, **_node_overrides.get(nid, {})}
            nodes.append({
                "id":       nid,
                "tag":      tag,
                "standard": std,
                "color":    STD_COLORS.get(std, "#888"),
                "source":   item.get("source", ""),
                "data":     merged,
            })

    edges: List[dict] = []
    link_set: set = set()
    for lnk in links:
        src_std = lnk.get("source_std", "")
        tgt_std = lnk.get("target_std", "")
        if src_std not in stds or tgt_std not in stds:
            continue
        src_id = f"{src_std}::{lnk['source_tag']}"
        tgt_id = f"{tgt_std}::{lnk['target_tag']}"
        if src_id not in node_id_set or tgt_id not in node_id_set:
            continue
        ek = (src_id, tgt_id)
        if ek in link_set:
            continue
        link_set.add(ek)
        edges.append({
            "id":                f"e::{src_id}::{tgt_id}",
            "source":            src_id,
            "target":            tgt_id,
            "score":             lnk.get("score", 0),
            "match_type":        lnk.get("match_type", ""),
            "relationship_type": lnk.get("relationship_type", "unknown"),
            "evidence":          lnk.get("evidence", ""),
        })

    return {"nodes": nodes, "edges": edges}


def _rag_search(query: str, top_k: int = 30) -> list:
    try:
        import numpy as np
        import faiss as faiss_lib
        from sentence_transformers import SentenceTransformer

        idx_p  = OUTPUT_DIR / "rag_index.faiss"
        docs_p = OUTPUT_DIR / "rag_docs.npy"
        meta_p = OUTPUT_DIR / "rag_meta.npy"
        if not all(p.exists() for p in [idx_p, docs_p, meta_p]):
            raise FileNotFoundError("RAG files missing")

        idx   = faiss_lib.read_index(str(idx_p))
        docs  = np.load(str(docs_p), allow_pickle=True)
        meta  = np.load(str(meta_p), allow_pickle=True)
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        qvec  = model.encode([query]).astype(np.float32)
        faiss_lib.normalize_L2(qvec)
        D, I = idx.search(qvec, min(top_k, len(docs)))

        results = []
        for score, i in zip(D[0], I[0]):
            if i < 0 or i >= len(meta):
                continue
            m = meta[i]
            m_dict = m.item() if hasattr(m, "item") else (m if isinstance(m, dict) else {})
            # Normalise standard casing in meta
            if "standard" in m_dict:
                m_dict["standard"] = _canon_std(m_dict["standard"])
            results.append({"score": float(score), "text": str(docs[i])[:400], "meta": m_dict})
        return results

    except Exception:
        q = query.lower()
        results = []
        for std, items in _load_all_nodes().items():
            for item in items:
                tag  = (item.get("tag_name") or "").lower()
                desc = (item.get("description") or "").lower()
                if q in tag or q in desc:
                    results.append({
                        "score": 1.0 if q in tag else 0.6,
                        "text":  item.get("description") or tag,
                        "meta":  {"standard": std, "tag": item.get("tag_name", ""),
                                  "type": item.get("source", "")},
                    })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def _get_input_tree() -> dict:
    if not INPUTS_DIR.exists():
        return {"name": "Inputs", "type": "folder", "children": []}

    def walk(path: Path) -> dict:
        if path.is_dir():
            cname = _canon_std(path.name)
            display = cname if cname in STANDARDS else path.name
            return {
                "name":     display,
                "type":     "folder",
                "path":     str(path),
                "standard": display if display in STANDARDS else None,
                "children": [walk(c) for c in sorted(path.iterdir())],
            }
        return {
            "name": path.name, "type": "file",
            "path": str(path), "ext": path.suffix,
            "standard": None, "children": [],
        }
    return walk(INPUTS_DIR)


# ── Original endpoints (preserved) ────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "S-Series Unified Data Platform v2.1"}


@app.post("/translate")
async def translate(
    file: UploadFile = File(...),
    from_std: str = Form(...),
    to_std: str = Form(...),
):
    temp_path = Path(f"temp_{file.filename}")
    with temp_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        result = translator.translate(
            xml_input=str(temp_path),
            from_std=_canon_std(from_std),
            to_std=_canon_std(to_std),
        )
        return {"xml": result["xml_string"], "coverage": result.get("coverage", 0),
                "log": result.get("log", [])[:30]}
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/confirm-mapping")
async def confirm_mapping(
    from_std: str = Form(...), original_tag: str = Form(...),
    new_tag: str = Form(...), to_std: str = Form(...),
    example_value: str = Form(None),
):
    memory.record_correction(_canon_std(from_std), original_tag, new_tag,
                             _canon_std(to_std), example_value)
    return {"status": "learned", "message": "Mapping permanently stored"}


@app.get("/memory-stats")
async def memory_stats():
    return memory.get_stats()


# ── New endpoints ──────────────────────────────────────────────────────────────

@app.get("/load-data")
async def load_data():
    all_nodes = _load_all_nodes()
    ontology  = _load_ontology()
    links     = _load_crossmatch_links()

    node_counts = {std: len(items) for std, items in all_nodes.items()}
    total_nodes = sum(node_counts.values())

    rel_types: Dict[str, int] = {}
    for lnk in links:
        rt = lnk.get("relationship_type", "unknown")
        rel_types[rt] = rel_types.get(rt, 0) + 1

    source_breakdown: Dict[str, Dict[str, int]] = {}
    for std, items in all_nodes.items():
        s: Dict[str, int] = {}
        for it in items:
            src = it.get("source", "unknown")
            s[src] = s.get(src, 0) + 1
        source_breakdown[std] = s

    return {
        "standards":              STANDARDS,
        "std_colors":             STD_COLORS,
        "node_counts":            node_counts,
        "total_nodes":            total_nodes,
        "total_links":            len(links),
        "relationship_types":     rel_types,
        "source_breakdown":       source_breakdown,
        "ontology_entity_count":  len(ontology.get("entities", {})),
        "ontology_cross_refs":    len(ontology.get("cross_references", [])),
        "input_tree":             _get_input_tree(),
        "memory_stats":           memory.get_stats(),
    }


@app.get("/graph")
async def get_graph(standards: Optional[str] = Query(None)):
    if standards:
        active = [_canon_std(s.strip()) for s in standards.split(",")]
        active = [s for s in active if s in STANDARDS]
    else:
        active = None
    graph = _build_graph(active)
    return {"nodes": graph["nodes"], "edges": graph["edges"],
            "node_count": len(graph["nodes"]), "edge_count": len(graph["edges"])}


@app.get("/node/{node_id:path}")
async def get_node(node_id: str):
    if "::" not in node_id:
        raise HTTPException(400, "node_id must be 'STD::tagName'")
    std, tag = node_id.split("::", 1)
    std = _canon_std(std)
    nid = f"{std}::{tag}"

    all_nodes = _load_all_nodes()
    base = next((n for n in all_nodes.get(std, []) if n.get("tag_name") == tag),
                {"tag_name": tag, "standard": std, "source": "unknown"})
    merged = {**base, **_node_overrides.get(nid, {})}

    links = _load_crossmatch_links()
    linked = [lnk for lnk in links
              if (lnk.get("source_tag") == tag and lnk.get("source_std") == std)
              or (lnk.get("target_tag") == tag and lnk.get("target_std") == std)]

    return {"id": nid, "standard": std, "tag": tag, "data": merged,
            "color": STD_COLORS.get(std, "#888"), "links": linked[:100],
            "override_applied": nid in _node_overrides}


@app.post("/update-node")
async def update_node(payload: dict):
    nid = payload.get("id", "")
    updates = payload.get("updates", {})
    if "::" not in nid:
        raise HTTPException(400, "id must be 'STD::tagName'")
    std, tag = nid.split("::", 1)
    std = _canon_std(std)
    nid = f"{std}::{tag}"

    existing = _node_overrides.get(nid, {})
    existing.update(updates)
    existing["_updated_at"] = datetime.now(timezone.utc).isoformat()
    _node_overrides[nid] = existing

    links = _load_crossmatch_links()
    impacted = set()
    for lnk in links:
        if lnk.get("source_tag") == tag and lnk.get("source_std") == std:
            impacted.add(f"{lnk['target_std']}::{lnk['target_tag']}")
        elif lnk.get("target_tag") == tag and lnk.get("target_std") == std:
            impacted.add(f"{lnk['source_std']}::{lnk['source_tag']}")

    return {"status": "updated", "node_id": nid, "impacted": list(impacted)[:50]}


@app.post("/export/{standard}")
async def export_standard(standard: str, file: UploadFile = File(...),
                          from_std: str = Form("S1000D")):
    standard = _canon_std(standard)
    from_std = _canon_std(from_std)
    if standard not in STANDARDS:
        raise HTTPException(400, f"Unknown standard: {standard}")
    if standard == from_std:
        raise HTTPException(400, "Source and target must differ")

    temp_path = Path(f"temp_export_{file.filename}")
    with temp_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)
    out_path = OUTPUT_DIR / f"export_{from_std}_to_{standard}.xml"
    try:
        result = translator.translate(xml_input=str(temp_path),
                                      from_std=from_std, to_std=standard,
                                      output_path=str(out_path))
        return Response(content=result["xml_string"], media_type="application/xml",
                        headers={"Content-Disposition": f'attachment; filename="{from_std}_to_{standard}.xml"',
                                 "X-Coverage": str(result.get("coverage", 0))})
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/search")
async def search(q: str = Query(..., min_length=1)):
    return {"query": q, "results": _rag_search(q), "count": len(_rag_search(q))}


@app.get("/crossmatch-groups")
async def crossmatch_groups(min_score: float = Query(0.85)):
    links = _load_crossmatch_links()
    high = [lnk for lnk in links
            if lnk.get("score", 0) >= min_score
            and lnk.get("relationship_type") in ("equivalent", "partial_equivalent")]

    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)
            x = parent.get(x, x)
        return x

    def union(a: str, b: str):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    all_in_links: set = set()
    for lnk in high:
        a = f"{lnk['source_std']}::{lnk['source_tag']}"
        b = f"{lnk['target_std']}::{lnk['target_tag']}"
        all_in_links.update([a, b])
        union(a, b)

    groups: Dict[str, list] = {}
    for node in all_in_links:
        groups.setdefault(find(node), []).append(node)

    result = [{"group_id": r, "members": m, "size": len(m)}
              for r, m in groups.items() if len(m) >= 2]
    result.sort(key=lambda g: g["size"], reverse=True)
    return {"groups": result[:300], "total": len(result)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)