"""
sbrain_crossmatch.py — Semantic Cross-Standard Concept Matcher v3
Changes over v2:
  1. ConceptLink gains relationship_type field (equivalent/partial_equivalent/transformation/reference)
  2. SX1000iBooster fixed — bidirectional alias lookup so bridge confirmations actually fire
  3. SX1000i included as bridge node in _compute_all_links (was silently excluded before)
  4. build() passes SX1000i into standards list so links are generated
  5. get_best_match() priority order fixed:
       STRICT_MAP → AERO_ALIASES → _link_index → vector search
     (previously _link_index ran first, returning weak semantic hits before aliases)
  6. Domain penalty changed from -= 0.35 to *= 0.5 (stronger enforcement)
  7. Confidence floor raised: get_best_match() won't return score < 0.50 for
     non-rule matches, preventing garbage weak matches from reaching the translator
  8. find_best_concept_in_target() gains SX1000i multi-hop fallback
  9. Synthetic definitions enriched with domain classification
  10. normalized_cos = (cos + 1) / 2 applied in _compute_all_links for proper [0,1] range
  11. _classify_relationship() added as new sibling to _classify()
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
    relationship_type: str = "unknown"   # equivalent | partial_equivalent | transformation | reference


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _split_camel(name: str) -> str:
    """
    'partNumberValue' → 'part number value'
    'dmCode'          → 'dm code'
    'NSN'             → 'NSN'   (all-caps acronyms kept together)
    """
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
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
                parent_map = {child: parent for parent in root.iter() for child in parent}
                records.extend(
                    self._parse_xsd_file(root, parent_map, xsd_file.name, standard, seen)
                )
            except Exception as e:
                print(f"    Warning XSD {xsd_file.name}: {e}")
        return records

    def _parse_xsd_file(self, root, parent_map, fname, standard, seen) -> list:
        records = []

        for elem in root.findall(f".//{{{XS_NS}}}element"):
            name = elem.get("name")
            if not name or name in seen:
                continue
            seen.add(name)

            defn = self._extract_xsd_annotation(elem)
            xs_type = elem.get("type", "")
            parents = self._find_parents_via_map(elem, parent_map)

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

        for ct in root.findall(f".//{{{XS_NS}}}complexType"):
            name = ct.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            defn = self._extract_xsd_annotation(ct)
            records.append(ConceptRecord(
                tag_name=name,
                standard=standard,
                definition_text=defn if defn else self._synthetic_definition(name, "", []),
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
        """
        Enriched synthetic definition with domain classification.
        Much richer than the old generic template — gives embeddings real signal.
        """
        words = _split_camel(tag_name)

        # Domain classification
        tag_lower = tag_name.lower()
        if any(k in tag_lower for k in ["code", "ident", "number", "id", "ref"]):
            domain = "identifier"
        elif any(k in tag_lower for k in ["name", "nomenclature", "title", "description"]):
            domain = "nomenclature descriptor"
        elif any(k in tag_lower for k in ["quantity", "count", "qty", "amount"]):
            domain = "quantity value"
        elif any(k in tag_lower for k in ["price", "cost", "value", "rate"]):
            domain = "financial value"
        elif any(k in tag_lower for k in ["date", "time", "period", "year"]):
            domain = "temporal value"
        elif any(k in tag_lower for k in ["status", "state", "indicator", "flag"]):
            domain = "status indicator"
        elif any(k in tag_lower for k in ["address", "location", "site"]):
            domain = "location reference"
        else:
            domain = "data field"

        parent_ctx = f" within {parents[0]}" if parents else ""
        type_ctx   = f" typed as {xs_type}" if xs_type else ""
        return (
            f"{words} is a {domain}{parent_ctx} used in aerospace "
            f"technical publications and logistics data{type_ctx}."
        )

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

        local_model_path = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
        print(f"Loading embedding model from: {local_model_path}")
        self.model = SentenceTransformer(str(local_model_path))
        #self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
# SX1000i BOOSTER  (fixed — bidirectional)
# ─────────────────────────────────────────────

class SX1000iBooster:
    """
    Fixed bidirectional alias lookup.

    Old bug: confirmed[alias] = {sx_tag} — so boost(src, tgt) checked whether
    the target SX1000i tag was in confirmed[source], which almost never fired
    because source tags are S1000D/S2000M names, not SX1000i names.

    Fix: store bidirectional edges so any pair sharing a common SX1000i bridge
    node gets the confirmation boost.
    """
    def __init__(self, sx_records: list):
        self.confirmed: dict = {}
        for rec in sx_records:
            sx_key = _norm_key(rec.tag_name)
            for alias in rec.aliases:
                alias_key = _norm_key(alias)
                # Both directions: alias→sx and sx→alias
                self.confirmed.setdefault(alias_key, set()).add(sx_key)
                self.confirmed.setdefault(sx_key, set()).add(alias_key)

    def boost(self, source_tag: str, target_tag: str) -> float:
        sk = _norm_key(source_tag)
        tk = _norm_key(target_tag)
        source_connections = self.confirmed.get(sk, set())
        target_connections = self.confirmed.get(tk, set())
        # Strongest signal: source and target share a common SX1000i bridge node
        if source_connections & target_connections:
            return 0.20
        # Weaker signal: one is directly an alias of the other
        if tk in source_connections or sk in target_connections:
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
        "name_match":        0.55,
        "alias_match":       0.50,
        "unmapped":          0.0,
    }

    # Canonical STRICT overrides — always win, score = 1.0
    # These are known-correct aerospace mappings that must never be beaten by FAISS
    STRICT_MAP: Dict[str, Dict[str, str]] = {
        "S2000M": {
            "quantityPerAssembly":   "quantityPerNextHigherAssembly",
            "identName":             "itemNomenclature",
            "partNumberValue":       "partNumber",
            "fullNatoStockNumber":   "nsn",
            "natoStockNumber":       "nsn",
            "dmCode":                "dataModuleCode",
            "partNumber":            "partNumber",
            "nsn":                   "nsn",
            "figureNumber":          "figureReference",
            "itemSeqNumber":         "itemSequenceNumber",
        },
        "S1000D": {
            "itemNomenclature":            "identName",
            "qpa":                         "quantityPerAssembly",
            "ncageCode":                   "manufacturerCodeValue",
            "unitOfIssue":                 "unitOfMeasure",
            "provisioningSequenceNumber":  "item",
        },
        "S3000L": {
            "partNumberValue":   "partNumber",
            "identName":         "itemNomenclature",
            "lruPartNumber":     "partNumber",
            "hardwareItemId":    "partNumber",
        },
    }

    # Aerospace alias overrides — run before _link_index, score = 0.95
    # These are well-known tag equivalences across S-series standards
    AERO_ALIASES: Dict[str, Dict[str, str]] = {
        "S2000M": {
            "manufacturerCodeValue": "ncageCode",
            "identName":             "itemNomenclature",
            "quantityPerAssembly":   "qpa",
            "criticalityCode":       "criticalityIndicator",
            "sourceOfSupply":        "sourceOfSupplyCode",
            "unitPrice":             "unitPrice",
        },
        "S1000D": {
            "ncageCode":             "manufacturerCodeValue",
            "itemNomenclature":      "identName",
            "qpa":                   "quantityPerAssembly",
        },
        "S3000L": {
            "lsaTaskCode":           "maintenanceTaskCode",
            "failureMode":           "failureModeCode",
            "lruIdentifier":         "partNumber",
            "maintenanceLevel":      "maintenanceLevelCode",
        },
        "SX1000i": {
            "ipsElement":            "supportElement",
            "ipsPlan":               "supportPlan",
            "ipsRequirement":        "supportRequirement",
        },
    }

    def __init__(self, output_dir: str = "./output", model_name: str = "all-MiniLM-L6-v2"):
        self.output_dir = Path(output_dir)
        self.harvester = DefinitionHarvester()
        self.index = SemanticConceptIndex(model_name)
        self.structural = StructuralScorer()
        self.booster = None
        self.links: list = []
        self._link_index: dict = {}

        # Lazy import to prevent circular dependency
        from sbrain_learning_memory import SBrainLearningMemory
        self.learning_memory = SBrainLearningMemory()

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

        # FIX: include SX1000i in the link graph so bridge links are generated
        all_standards = list(standard_dirs.keys())
        if sx1000i_dir and Path(sx1000i_dir).exists():
            all_standards = all_standards + ["SX1000i"]

        self.links = self._compute_all_links(all_records, all_standards)

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
        # SX1000i is always included as bridge node even if not in original standards list
        bridge_standards = set(standards) | {"SX1000i"}
        links = []
        for rec in records:
            if rec.standard not in bridge_standards:
                continue
            neighbours = self.index.search(rec, top_k=8, exclude_standard=rec.standard)
            for neighbour, cos_score in neighbours:
                if neighbour.standard not in bridge_standards:
                    continue
                struct_bonus = self.structural.score(rec, neighbour)
                sx_boost     = self.booster.boost(rec.tag_name, neighbour.tag_name) if self.booster else 0.0
                name_bonus   = _name_similarity(rec.tag_name, neighbour.tag_name) * 0.12

                # Normalize cosine from [-1,1] to [0,1] — FAISS IndexFlatIP
                # returns inner product on L2-normalised vecs, range is [-1,1]
                normalized_cos = (cos_score + 1.0) / 2.0
                final_score    = min(normalized_cos + struct_bonus + sx_boost + name_bonus, 1.0)

                match_type        = self._classify(cos_score, struct_bonus, sx_boost, name_bonus)
                relationship_type = self._classify_relationship(rec, neighbour, cos_score, name_bonus)
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
                    relationship_type=relationship_type,
                    evidence=evidence,
                ))
        links.sort(key=lambda l: l.score, reverse=True)
        return links

    def _classify(self, cos: float, struct: float, sx: float, name: float) -> str:
        """Classify HOW a match was found (method/evidence type)."""
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

    def _classify_relationship(self, source, target, cos_score: float, name_bonus: float) -> str:
        """
        Classify WHAT KIND of semantic relationship exists between two concepts.
        Used by the translator to decide how to handle the mapping.

        equivalent        — same concept, safe to map value directly
        partial_equivalent — overlapping but not identical, map with caution
        transformation    — requires value restructuring or field renaming
        reference         — weak link, don't map blindly
        """
        TRANSFORM_PAIRS = {
            ("partNumberValue",        "partNumber"),
            ("identName",              "itemNomenclature"),
            ("fullNatoStockNumber",    "nsn"),
            ("quantityPerAssembly",    "quantityPerNextHigherAssembly"),
            ("dmCode",                 "dataModuleCode"),
            ("manufacturerCodeValue",  "ncageCode"),
            ("ncageCode",              "manufacturerCodeValue"),
            ("itemNomenclature",       "identName"),
            ("qpa",                    "quantityPerAssembly"),
            ("lruPartNumber",          "partNumber"),
            ("lruIdentifier",          "partNumber"),
        }

        # Exact name match after normalisation = equivalent
        if _norm_key(source.tag_name) == _norm_key(target.tag_name):
            return "equivalent"

        # Very high cosine + name overlap = effectively the same concept
        if cos_score >= 0.90 and name_bonus >= 0.08:
            return "equivalent"

        # Known aerospace transformation pairs
        if (source.tag_name, target.tag_name) in TRANSFORM_PAIRS:
            return "transformation"

        # High cosine but different name = same concept, different standard terminology
        if cos_score >= 0.78:
            return "partial_equivalent"

        # Weak cosine = reference link only — translator should not map blindly
        if cos_score < 0.60:
            return "reference"

        return "partial_equivalent"

    def get_ontology_graph(self) -> Dict:
        """Returns full bidirectional ontology graph for any-to-any translation."""
        graph = {}
        for link in self.links:
            key = (link.source_std, link.source_tag)
            if key not in graph:
                graph[key] = []
            graph[key].append({
                "target_std":       link.target_std,
                "target_tag":       link.target_tag,
                "score":            link.score,
                "match_type":       link.match_type,
                "relationship_type": link.relationship_type,
            })
        return graph

    def find_best_concept_in_target(self, source_tag: str, from_std: str, to_std: str) -> Optional[ConceptLink]:
        candidates = self.get_all_matches(source_tag, from_std, to_std, min_score=0.35)
        if candidates:
            return max(candidates, key=lambda x: x.score)

        # Fallback: live vector search with synthetic definition
        synth_def  = self.harvester._synthetic_definition(source_tag)
        neighbours = self.index.search_by_text(synth_def, top_k=8, exclude_standard=from_std)

        best_link  = None
        best_score = 0.0

        for rec, cos_score in neighbours:
            if rec.standard != to_std:
                continue
            name_bonus  = _name_similarity(source_tag, rec.tag_name) * 0.25
            final_score = cos_score + name_bonus

            if final_score > best_score and final_score >= 0.32:
                best_score = final_score
                best_link  = ConceptLink(
                    source_tag=source_tag,
                    source_std=from_std,
                    target_tag=rec.tag_name,
                    target_std=to_std,
                    score=round(final_score, 4),
                    match_type="semantic_fallback",
                    relationship_type="partial_equivalent",
                    evidence=f"cos={cos_score:.3f} + name={name_bonus:.3f}"
                )

        # Multi-hop: if still no match, try S source → SX1000i → target standard
        if best_link is None and from_std != "SX1000i" and to_std != "SX1000i":
            sx_match = self.get_best_match(source_tag, from_std, "SX1000i")
            if sx_match and sx_match.score >= 0.65:
                hop_match = self.get_best_match(sx_match.target_tag, "SX1000i", to_std)
                if hop_match and hop_match.score >= 0.65:
                    bridged_score = round(sx_match.score * hop_match.score, 4)
                    best_link = ConceptLink(
                        source_tag=source_tag,
                        source_std=from_std,
                        target_tag=hop_match.target_tag,
                        target_std=to_std,
                        score=bridged_score,
                        match_type="sx1000i_bridge",
                        relationship_type="partial_equivalent",
                        evidence=(
                            f"via SX1000i:{sx_match.target_tag} "
                            f"hop1={sx_match.score:.3f} hop2={hop_match.score:.3f}"
                        ),
                    )

        return best_link

    def _name_similarity(self, a: str, b: str) -> float:
        """Simple token overlap similarity for fallback."""
        ta = set(re.sub(r'([A-Z])', r' \1', a).lower().split())
        tb = set(re.sub(r'([A-Z])', r' \1', b).lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _build_link_index(self):
        """
        Build a fast dict for O(1) lookup during translation.
        Key: (normalised_tag, source_std, target_std)
        """
        self._link_index = {}
        for lnk in self.links:
            key = (_norm_key(lnk.source_tag), lnk.source_std, lnk.target_std)
            self._link_index.setdefault(key, []).append(lnk)
        for k in self._link_index:
            self._link_index[k].sort(key=lambda l: l.score, reverse=True)

    # ── Query API ──────────────────────────────────────────────────────────────

    def get_best_match(self, tag: str, from_std: str, to_std: str, context: str = ""):
        """
        Priority order (fixed from v2):
          1. STRICT_MAP  — hardcoded correct aerospace mappings, score = 1.0
          2. AERO_ALIASES — known tag aliases, score = 0.95, runs BEFORE link_index
          3. _link_index  — precomputed FAISS links
          4. vector search — live embedding lookup with domain filtering
        """
        validated = self.learning_memory.get_validated_mapping(from_std, tag, to_std)
        if validated:
            return ConceptLink(
                source_tag=tag,
                source_std=from_std,
                target_tag=validated["target_tag"],
                target_std=to_std,
                score=1.0,
                match_type="human_validated",
                relationship_type="equivalent",
                evidence=f"user_confirmed_{validated.get('times_confirmed', 1)}x"
            )
        norm = _norm_key(tag)

        # ── 1. STRICT overrides — always win ──────────────────────────────────
        strict_targets = self.STRICT_MAP.get(to_std, {})
        if tag in strict_targets:
            return ConceptLink(
                source_tag=tag,
                source_std=from_std,
                target_tag=strict_targets[tag],
                target_std=to_std,
                score=1.0,
                match_type="rule_override",
                relationship_type="equivalent",
                evidence="STRICT_MAP",
            )

        # ── 2. AERO_ALIASES — before _link_index so weak FAISS can't override ─
        alias_targets = self.AERO_ALIASES.get(to_std, {})
        if tag in alias_targets:
            target_tag = alias_targets[tag]
            # Verify the target tag actually exists in the index for this standard
            target_exists = any(
                r.tag_name == target_tag and r.standard == to_std
                for r in self.index.records
            )
            if target_exists:
                return ConceptLink(
                    source_tag=tag,
                    source_std=from_std,
                    target_tag=target_tag,
                    target_std=to_std,
                    score=0.95,
                    match_type="alias_match",
                    relationship_type="transformation",
                    evidence="AERO_ALIAS",
                )

        # ── 3. Precomputed link index (fast O(1) path) ─────────────────────────
        key = (norm, from_std, to_std)
        if key in self._link_index:
            best = self._link_index[key][0]
            # Don't return weak reference-type links from the index —
            # let them fall through to vector search which may find better
            if best.score >= 0.50 or best.relationship_type not in ("reference", "unknown"):
                return best

        # ── 4. Domain enforcement setup ────────────────────────────────────────
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

        # ── 5. Live vector search ──────────────────────────────────────────────
        # Enrich context for known identity-bearing tags
        IDENTITY_TAGS = {
            "manufacturerCodeValue", "partNumberValue", "identName",
            "ncageCode", "partNumber", "itemNomenclature",
        }
        if tag in IDENTITY_TAGS:
            context = f"{context} | part identification field in provisioning data module".strip(" |")

        rich_query = f"{tag} {context}" if context else tag
        query_vec  = self.index.model.encode([rich_query]).astype(np.float32)
        faiss.normalize_L2(query_vec)

        D, I = self.index.index.search(query_vec, 20)

        candidates = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.index.records):
                continue
            res = self.index.records[idx]
            if res.standard != to_std:
                continue

            final_score = float(score)
            target_norm = _norm_key(res.tag_name)

            # Domain penalty — multiplicative (stronger than old -= 0.35)
            if source_domain:
                domain_keys = DOMAIN_MAP[source_domain]
                target_def  = res.definition_text.lower()
                if not any(k in target_norm or k in target_def for k in domain_keys):
                    final_score *= 0.5   # halve score for domain mismatches

            # Name overlap bonus
            if norm in target_norm or target_norm in norm:
                final_score += 0.20

            candidates.append(ConceptLink(
                source_tag=tag,
                source_std=from_std,
                target_tag=res.tag_name,
                target_std=to_std,
                score=min(round(final_score, 4), 1.0),
                match_type="semantic_filtered",
                relationship_type="partial_equivalent",
                evidence=f"domain={source_domain} cos={score:.3f}",
            ))

        if candidates:
            candidates.sort(key=lambda x: x.score, reverse=True)
            best = candidates[0]
            # Confidence floor: don't return weak matches that aren't rule-backed
            if best.score < 0.50:
                return None
            return best

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
                    raw_links = json.load(f)
                # Handle old concept_links.json that lack relationship_type field
                matcher.links = [
                    ConceptLink(**{k: v for k, v in l.items() if k in ConceptLink.__dataclass_fields__})
                    for l in raw_links
                ]
            matcher._build_link_index()
            print(f"  Cross-matcher loaded: {len(matcher.links)} links, "
                  f"{len(matcher._link_index)} unique lookup keys")
        else:
            print("  WARNING: No saved index found — run sbrain_core.py first")
        return matcher
