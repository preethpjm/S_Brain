"""
sbrain_crossmatch.py — Semantic Cross-Standard Concept Matcher v2
Fixes:
  1. getparent() replaced with ET parent_map (stdlib compatible)
  2. S1000D element names harvested from ALL nested elements, not just top-level
  3. Synthetic definitions improved — camelCase split gives much richer signal
  4. Case-insensitive + camelCase-normalised lookup in get_best_match
  5. Direct name-similarity bonus to catch obvious aliases (partNumber ↔ partNumberValue)
  6. SX1000i treated as source standard too so its elements appear as match targets
"""

import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

XS_NS = "http://www.w3.org/2001/XMLSchema"


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class ConceptRecord:
    tag_name: str
    standard: str
    definition_text: str
    aliases: list = field(default_factory=list)
    source_file: str = ""
    xs_type: str = ""
    parent_elements: list = field(default_factory=list)
    source: str = ""

@dataclass
class ConceptLink:
    source_tag: str
    source_std: str
    target_tag: str
    target_std: str
    score: float
    match_type: str
    evidence: str = ""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _split_camel(name: str) -> str:
    """
    'partNumberValue' → 'part number value'
    'dmCode'          → 'dm code'
    'NSN'             → 'NSN'   (all-caps acronyms kept together)
    """
    # Insert space before uppercase letters that follow lowercase
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # Insert space before uppercase run followed by lowercase (acronym boundary)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


def _norm_key(name: str) -> str:
    """Normalise for lookup: remove separators, lowercase."""
    return re.sub(r'[\s\-_]', '', name.lower())


def _name_similarity(a: str, b: str) -> float:
    """
    Simple token-overlap similarity between two tag names after camelCase split.
    Returns 0.0–1.0.  Used as a bonus on top of cosine.
    """
    ta = set(_split_camel(a).split())
    tb = set(_split_camel(b).split())
    if not ta or not tb:
        return 0.0
    # Jaccard
    return len(ta & tb) / len(ta | tb)


# ─────────────────────────────────────────────
# DEFINITION HARVESTER
# ─────────────────────────────────────────────

class DefinitionHarvester:
    """
    Pulls definitions from every source available for a standard.
    Priority: XSD annotation > XMI note > PDF extraction > camelCase synthetic
    """

    def harvest_xsd(self, xsd_dir: str, standard: str) -> list:
        records = []
        seen = set()
        for xsd_file in sorted(Path(xsd_dir).glob("**/*.xsd")):
            try:
                tree = ET.parse(str(xsd_file))
                root = tree.getroot()
                # Build parent map for this file (stdlib compatible — no lxml needed)
                parent_map = {child: parent for parent in root.iter() for child in parent}
                records.extend(
                    self._parse_xsd_file(root, parent_map, xsd_file.name, standard, seen)
                )
            except Exception as e:
                print(f"    Warning XSD {xsd_file.name}: {e}")
        return records

    def _parse_xsd_file(self, root, parent_map, fname, standard, seen) -> list:
        records = []

        # ── Collect ALL elements (global + nested) ──
        for elem in root.findall(f".//{{{XS_NS}}}element"):
            name = elem.get("name")
            if not name or name in seen:
                continue
            seen.add(name)

            defn = self._extract_xsd_annotation(elem)
            xs_type = elem.get("type", "")
            parents = self._find_parents_via_map(elem, parent_map)

            # Use camelCase split to build a richer synthetic definition
            if not defn:
                defn = self._synthetic_definition(name, xs_type, parents)

            records.append(ConceptRecord(
                tag_name=name,
                standard=standard,
                definition_text=defn,
                source_file=fname,
                xs_type=xs_type,
                parent_elements=parents,
                source="xsd"
            ))

        # ── Global complexTypes (structural concepts) ──
        for ct in root.findall(f".//{{{XS_NS}}}complexType"):
            name = ct.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            defn = self._extract_xsd_annotation(ct)
            if defn:
                records.append(ConceptRecord(
                    tag_name=name,
                    standard=standard,
                    definition_text=defn,
                    source_file=fname,
                    source="xsd",
                ))
            else:
                # Still add with synthetic def so the name is in the index
                records.append(ConceptRecord(
                    tag_name=name,
                    standard=standard,
                    definition_text=self._synthetic_definition(name, "", []),
                    source_file=fname,
                    source="xsd",
                ))

        return records

    def _extract_xsd_annotation(self, node) -> str:
        """Pulls xs:documentation text — the richest definition source."""
        for ann in node.findall(f".//{{{XS_NS}}}documentation"):
            if ann.text and len(ann.text.strip()) > 10:
                return ann.text.strip()[:600]
        return ""

    def _find_parents_via_map(self, elem, parent_map: dict) -> list:
        """
        Walk up the parent_map chain to find the nearest named ancestor element.
        Fully stdlib-compatible — no lxml getparent() needed.
        """
        parents = []
        current = elem
        depth = 0
        while current in parent_map and depth < 6:
            p = parent_map[current]
            pname = p.get("name")
            if pname and pname not in parents:
                parents.append(pname)
                if len(parents) >= 3:
                    break
            current = p
            depth += 1
        return parents

    def _synthetic_definition(self, tag_name: str, xs_type: str = "", parents: List[str] = None) -> str:
            words = re.sub(r'([A-Z])', r' \1', tag_name).replace('_', ' ').strip().lower()
            extra = " identifier" if any(k in tag_name.lower() for k in ["code","ident","number","id"]) else ""
            if parents:
                extra += f" inside {parents[0]}"
            return f"This element represents a{extra} {words} in aerospace technical publications and data modules."
    
    def harvest_xmi(self, xmi_path: str, standard: str) -> list:
        """Parses Enterprise Architect XMI — Note field is the definition."""
        records = []
        try:
            tree = ET.parse(xmi_path)
            root = tree.getroot()
            for table in root.findall(".//Table[@name='t_object']"):
                for row in table.findall("Row"):
                    cols = {c.get("name"): c.get("value") for c in row.findall("Column")}
                    if cols.get("Object_Type") != "Class":
                        continue
                    name = cols.get("Name", "").strip()
                    note = cols.get("Note", "").strip()
                    if not name:
                        continue
                    defn = note[:600] if note and len(note) > 15 else self._synthetic_definition(name, "", [])
                    records.append(ConceptRecord(
                        tag_name=name,
                        standard=standard,
                        definition_text=defn,
                        source="xmi"
                    ))
        except Exception as e:
            print(f"    Warning XMI {xmi_path}: {e}")
        return records

    def harvest_sx1000i(self, sx1000i_dir: str) -> list:
        records = []
        sx_dir = Path(sx1000i_dir)
        seen = set()

        for xsd_file in sorted(sx_dir.glob("**/*.xsd")):
            try:
                tree = ET.parse(str(xsd_file))
                root = tree.getroot()
                parent_map = {child: parent for parent in root.iter() for child in parent}
                for elem in root.findall(f".//{{{XS_NS}}}element"):
                    name = elem.get("name")
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    defn = self._extract_xsd_annotation(elem)
                    cross_refs = self._extract_sx1000i_crossrefs(elem)
                    parents = self._find_parents_via_map(elem, parent_map)
                    records.append(ConceptRecord(
                        tag_name=name,
                        standard="SX1000i",
                        definition_text=defn or self._synthetic_definition(name, elem.get("type",""), parents),
                        aliases=cross_refs,
                        source_file=xsd_file.name,
                        source="sx1000i"
                    ))
            except Exception as e:
                print(f"    Warning SX1000i XSD {xsd_file.name}: {e}")

        for xmi_file in sx_dir.glob("**/*.xmi"):
            records.extend(self.harvest_xmi(str(xmi_file), "SX1000i"))

        print(f"    SX1000i: {len(records)} concept records harvested")
        return records

    def _extract_sx1000i_crossrefs(self, elem) -> list:
        aliases = []
        for ann in elem.findall(f".//{{{XS_NS}}}documentation"):
            if ann.text:
                for m in re.finditer(
                    r"(?:S1000D|S2000M|S3000L)\s*[:\-]?\s*([a-zA-Z][a-zA-Z0-9]{2,50})",
                    ann.text
                ):
                    aliases.append(m.group(1))
        return aliases

    def harvest_pdf_definitions(self, extracted_json: str, standard: str) -> list:
        records = []
        try:
            with open(extracted_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entity in data.get("entities", []):
                if entity.get("entity_type") in ("Definition", "StandardEntity"):
                    defn = entity.get("description", "").strip()
                    if len(defn) < 15:
                        continue
                    records.append(ConceptRecord(
                        tag_name=entity["name"],
                        standard=standard,
                        definition_text=defn[:600],
                        source_file=extracted_json,
                        source="pdf"
                    ))
        except Exception as e:
            print(f"    Warning PDF JSON {extracted_json}: {e}")
        return records


# ─────────────────────────────────────────────
# SEMANTIC INDEX
# ─────────────────────────────────────────────

class SemanticConceptIndex:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

        model_path = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
        print(f"Loading embedding model from: {model_path}")
        #self.model = SentenceTransformer(str(model_path))
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.records: list = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

    def build(self, records: list):
        self.records = records
        texts = [r.definition_text for r in records]
        print(f"    Embedding {len(texts)} concept records...")
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        self.embeddings = self.embeddings.astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"    Index built: {self.index.ntotal} vectors, dim={dim}")

    def search_by_text(self, text: str, top_k: int = 20,
                       exclude_standard: str = None) -> list:
        """Search by raw text — used for live translation lookups."""
        q_emb = self.model.encode([text]).astype(np.float32)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        k = min(top_k * 4, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.records):
                continue
            rec = self.records[idx]
            if exclude_standard and rec.standard == exclude_standard:
                continue
            results.append((rec, float(score)))
            if len(results) >= top_k:
                break
        return results

    def search(self, query_record, top_k: int = 10,
               exclude_standard: str = None) -> list:
        return self.search_by_text(
            query_record.definition_text, top_k, exclude_standard
        )

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "concept_index.faiss"))
        np.save(str(p / "concept_records.npy"),
                np.array([asdict(r) for r in self.records], dtype=object))

    def load(self, path: str) -> bool:
        p = Path(path)
        fi = p / "concept_index.faiss"
        nr = p / "concept_records.npy"
        if not (fi.exists() and nr.exists()):
            return False
        self.index = faiss.read_index(str(fi))
        raw = np.load(str(nr), allow_pickle=True).tolist()
        self.records = [ConceptRecord(**r) for r in raw]
        return True


# ─────────────────────────────────────────────
# STRUCTURAL SCORER
# ─────────────────────────────────────────────

class StructuralScorer:
    TYPE_FAMILIES = {
        "string":  {"string", "token", "normalizedString", "Name", "NCName", "NMTOKEN"},
        "numeric": {"integer", "decimal", "float", "double", "nonNegativeInteger", "positiveInteger"},
        "date":    {"date", "dateTime", "gYear", "gYearMonth"},
        "boolean": {"boolean"},
        "id":      {"ID", "IDREF", "IDREFS"},
    }

    def score(self, source, target) -> float:
        bonus = 0.0
        src_family = self._type_family(source.xs_type)
        tgt_family = self._type_family(target.xs_type)
        if src_family and src_family == tgt_family:
            bonus += 0.08
        src_parents = {_norm_key(p) for p in source.parent_elements}
        tgt_parents = {_norm_key(p) for p in target.parent_elements}
        if src_parents and tgt_parents:
            overlap = len(src_parents & tgt_parents) / max(len(src_parents | tgt_parents), 1)
            bonus += overlap * 0.06
        return min(bonus, 0.15)

    def _type_family(self, xs_type: str) -> Optional[str]:
        t = xs_type.split(":")[-1] if ":" in xs_type else xs_type
        for family, types in self.TYPE_FAMILIES.items():
            if t in types:
                return family
        return None


# ─────────────────────────────────────────────
# SX1000i BOOSTER
# ─────────────────────────────────────────────

class SX1000iBooster:
    def __init__(self, sx_records: list):
        self.confirmed: dict = {}
        for rec in sx_records:
            for alias in rec.aliases:
                key = _norm_key(alias)
                self.confirmed.setdefault(key, set()).add(_norm_key(rec.tag_name))

    def boost(self, source_tag: str, target_tag: str) -> float:
        sk = _norm_key(source_tag)
        tk = _norm_key(target_tag)
        if tk in self.confirmed.get(sk, set()) or sk in self.confirmed.get(tk, set()):
            return 0.15
        return 0.0


# ─────────────────────────────────────────────
# CROSS MATCHER
# ─────────────────────────────────────────────

class SemanticCrossMatcher:

    SCORE_THRESHOLDS = {
        "definition_match":  0.78,
        "structural_match":  0.68,
        "sx1000i_confirmed": 0.60,
        "name_match":        0.55,   # strong name similarity even if definition weak
        "alias_match":       0.50,
        "unmapped":          0.0,
    }

    def __init__(self, output_dir: str = "./output",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.output_dir = Path(output_dir)
        self.harvester = DefinitionHarvester()
        self.index = SemanticConceptIndex(model_name)
        self.structural = StructuralScorer()
        self.booster = None
        self.links: list = []
        # Fast lookup index built after load/build
        self._link_index: dict = {}   # (src_tag_norm, src_std, tgt_std) → [ConceptLink]

    def build(self, standard_dirs: dict, extracted_jsons: dict,
              sx1000i_dir: Optional[str] = None):
        all_records: list = []

        for std, xsd_dir in standard_dirs.items():
            print(f"\n  Harvesting {std}...")
            if Path(xsd_dir).exists():
                recs = self.harvester.harvest_xsd(xsd_dir, std)
                print(f"    XSD: {len(recs)} records")
                all_records.extend(recs)
                for xmi in Path(xsd_dir).glob("**/*.xmi"):
                    xmi_recs = self.harvester.harvest_xmi(str(xmi), std)
                    print(f"    XMI: {len(xmi_recs)} records")
                    all_records.extend(xmi_recs)
            if std in extracted_jsons and Path(extracted_jsons[std]).exists():
                pdf_recs = self.harvester.harvest_pdf_definitions(extracted_jsons[std], std)
                print(f"    PDF: {len(pdf_recs)} records")
                all_records.extend(pdf_recs)

        sx_records = []
        if sx1000i_dir and Path(sx1000i_dir).exists():
            print(f"\n  Harvesting SX1000i...")
            sx_records = self.harvester.harvest_sx1000i(sx1000i_dir)
            all_records.extend(sx_records)

        self.booster = SX1000iBooster(sx_records)

        all_records = self._dedup(all_records)
        print(f"\n  Total concept records after dedup: {len(all_records)}")
        by_std = {}
        for r in all_records:
            by_std.setdefault(r.standard, 0)
            by_std[r.standard] += 1
        for s, c in sorted(by_std.items()):
            print(f"    {s}: {c} records")

        self.index.build(all_records)

        print(f"\n  Computing cross-standard links...")
        self.links = self._compute_all_links(all_records, list(standard_dirs.keys()))

        self._build_link_index()
        self._save()
        print(f"\n  Cross-matcher ready: {len(self.links)} concept links")

    def _dedup(self, records: list) -> list:
        seen: dict = {}
        for r in records:
            key = f"{r.standard}::{r.tag_name}"
            if key not in seen or len(r.definition_text) > len(seen[key].definition_text):
                seen[key] = r
        return list(seen.values())

    def _compute_all_links(self, records: list, standards: list) -> list:
        links = []
        for rec in records:
            if rec.standard not in standards:
                continue
            neighbours = self.index.search(rec, top_k=8, exclude_standard=rec.standard)
            for neighbour, cos_score in neighbours:
                if neighbour.standard not in standards:
                    continue
                struct_bonus = self.structural.score(rec, neighbour)
                sx_boost = self.booster.boost(rec.tag_name, neighbour.tag_name) if self.booster else 0.0
                name_bonus = _name_similarity(rec.tag_name, neighbour.tag_name) * 0.12
                final_score = min(cos_score + struct_bonus + sx_boost + name_bonus, 1.0)
                match_type = self._classify(cos_score, struct_bonus, sx_boost, name_bonus)
                evidence = (
                    f"cos={cos_score:.3f} struct={struct_bonus:.3f} "
                    f"sx1000i={sx_boost:.3f} name={name_bonus:.3f} → final={final_score:.3f}"
                )
                links.append(ConceptLink(
                    source_tag=rec.tag_name,
                    source_std=rec.standard,
                    target_tag=neighbour.tag_name,
                    target_std=neighbour.standard,
                    score=round(final_score, 4),
                    match_type=match_type,
                    evidence=evidence,
                ))
        links.sort(key=lambda l: l.score, reverse=True)
        return links

    def _classify(self, cos: float, struct: float, sx: float, name: float) -> str:
        if sx > 0:
            return "sx1000i_confirmed"
        if name >= 0.08 and cos + name >= self.SCORE_THRESHOLDS["name_match"]:
            return "name_match"
        if cos >= self.SCORE_THRESHOLDS["definition_match"]:
            return "definition_match"
        if cos + struct >= self.SCORE_THRESHOLDS["structural_match"]:
            return "structural_match"
        if cos >= self.SCORE_THRESHOLDS["alias_match"]:
            return "alias_match"
        return "unmapped"

    def get_ontology_graph(self) -> Dict:
            """Returns full bidirectional ontology graph for any-to-any translation"""
            graph = {}
            for link in self.links:
                key = (link.source_std, link.source_tag)
                if key not in graph:
                    graph[key] = []
                graph[key].append({
                    "target_std": link.target_std,
                    "target_tag": link.target_tag,
                    "score": link.score,
                    "match_type": link.match_type
                })
            return graph

    def find_best_concept_in_target(self, source_tag: str, from_std: str, to_std: str) -> Optional[ConceptLink]:
        candidates = self.get_all_matches(source_tag, from_std, to_std, min_score=0.35)
        if candidates:
            return max(candidates, key=lambda x: x.score)

        # True fallback with rich context
        synth_def = self.harvester._synthetic_definition(source_tag)
        neighbours = self.index.search_by_text(synth_def, top_k=8, exclude_standard=from_std)
        
        best_link = None
        best_score = 0.0

        for rec, cos_score in neighbours:
            if rec.standard != to_std:
                continue
            name_bonus = _name_similarity(source_tag, rec.tag_name) * 0.25   # module-level function
            final_score = cos_score + name_bonus

            if final_score > best_score and final_score >= 0.32:   # MIN_RETURN_SCORE floor
                best_score = final_score
                best_link = ConceptLink(
                    source_tag=source_tag,
                    source_std=from_std,
                    target_tag=rec.tag_name,
                    target_std=to_std,
                    score=round(final_score, 4),
                    match_type="semantic_fallback",
                    evidence=f"cos={cos_score:.3f} + name={name_bonus:.3f}"
                )

        return best_link
    
    def _name_similarity(self, a: str, b: str) -> float:
        """Simple token overlap similarity for fallback"""
        ta = set(re.sub(r'([A-Z])', r' \1', a).lower().split())
        tb = set(re.sub(r'([A-Z])', r' \1', b).lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _build_link_index(self):
        """
        Build a fast dict for O(1) lookup during translation.
        Key: (normalised_tag, source_std, target_std)
        Also add normalised-key variants to catch case mismatches.
        """
        self._link_index = {}
        for lnk in self.links:
            # Exact key
            key = (_norm_key(lnk.source_tag), lnk.source_std, lnk.target_std)
            self._link_index.setdefault(key, []).append(lnk)
        # Sort each bucket by score descending
        for k in self._link_index:
            self._link_index[k].sort(key=lambda l: l.score, reverse=True)

    # ── Query API ──

    def get_best_match(self, tag: str, from_std: str, to_std: str, context: str = ""):
        norm = _norm_key(tag)

        # 1. Fast path — precomputed link index
        key = (norm, from_std, to_std)
        if key in self._link_index:
            return self._link_index[key][0]

        # 2. Domain enforcement
        DOMAIN_MAP = {
            "quantity": ["quantity", "qpa", "count", "amount"],
            "identity": ["partnumber", "pnr", "ident", "nomenclature"],
            "logistics": ["nsn", "stock", "natocode", "supply", "cage"],
            "pricing":   ["price", "cost", "currency", "unitprice"],
        }
        source_domain = None
        for domain, keys in DOMAIN_MAP.items():
            if any(k in norm for k in keys):
                source_domain = domain
                break

        # 3. Manual alias overrides
        AERO_ALIASES = {
            "manufacturerCodeValue": "ncageCode",
            "identName":             "itemNomenclature",
            "quantityPerAssembly":   "qpa",
        }
        if tag in AERO_ALIASES and to_std == "S2000M":
            for rec in self.index.records:          # FIXED: self.index.records
                if rec.tag_name == AERO_ALIASES[tag] and rec.standard == "S2000M":
                    return ConceptLink(
                        source_tag=tag,
                        source_std=from_std,        # FIXED: correct field name
                        target_tag=rec.tag_name,
                        target_std=to_std,          # FIXED: correct field name
                        score=1.0,
                        match_type="alias_match",
                        evidence="AERO_ALIAS_BOOST",
                    )

        # 4. Live vector search
        rich_query = f"{tag} {context}" if context else tag
        query_vec = self.index.model.encode(        # FIXED: self.index.model
            [rich_query]
        ).astype(np.float32)
        faiss.normalize_L2(query_vec)

        D, I = self.index.index.search(query_vec, 20)  # FIXED: self.index.index

        candidates = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.index.records):  # FIXED: self.index.records
                continue
            res = self.index.records[idx]                   # FIXED: self.index.records
            if res.standard != to_std:
                continue

            final_score = float(score)
            target_norm = _norm_key(res.tag_name)

            # Domain penalty
            if source_domain:
                domain_keys = DOMAIN_MAP[source_domain]
                target_def  = res.definition_text.lower()
                if not any(k in target_norm or k in target_def for k in domain_keys):
                    final_score -= 0.35

            # Name overlap bonus
            if norm in target_norm or target_norm in norm:
                final_score += 0.20

            candidates.append(ConceptLink(
                source_tag=tag,
                source_std=from_std,        # FIXED: correct field name
                target_tag=res.tag_name,
                target_std=to_std,          # FIXED: correct field name
                score=min(round(final_score, 4), 1.0),
                match_type="semantic_filtered",
                evidence=f"domain={source_domain} cos={score:.3f}",
            ))

        if candidates:
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[0]

        return None

    def get_all_matches(self, tag: str, from_std: str, to_std: str,
                        min_score: float = 0.40) -> list:
        key = (_norm_key(tag), from_std, to_std)
        return [l for l in self._link_index.get(key, []) if l.score >= min_score]

    def _save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index.save(str(self.output_dir))
        links_out = [asdict(l) for l in self.links]
        path = self.output_dir / "concept_links.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(links_out, f, indent=2)
        print(f"  Saved: {path} ({len(links_out)} links)")

    @classmethod
    def load(cls, output_dir: str = "./output",
             model_name: str = "all-MiniLM-L6-v2") -> "SemanticCrossMatcher":
        matcher = cls(output_dir, model_name)
        if matcher.index.load(output_dir):
            links_path = Path(output_dir) / "concept_links.json"
            if links_path.exists():
                with open(links_path) as f:
                    matcher.links = [ConceptLink(**l) for l in json.load(f)]
            matcher._build_link_index()
            print(f"  Cross-matcher loaded: {len(matcher.links)} links, "
                  f"{len(matcher._link_index)} unique lookup keys")
        else:
            print("  WARNING: No saved index found — run sbrain_core.py first")
        return matcher
