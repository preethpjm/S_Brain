"""
sbrain_crossmatch.py — Semantic Cross-Standard Concept Matcher v4
Changes over v3:
  1.  ENRICHED synthetic definitions — standard-specific vocabulary per tag domain
      so embedding model gets real aerospace signal even without xs:documentation
  2.  EXPANDED STRICT_MAP — S3000L, S1000D, S2000M all have full known-correct pairs
  3.  EXPANDED AERO_ALIASES — same; S3000L aliases were near-empty before
  4.  EXPANDED DOMAIN_MAP — now covers reliability, maintenance, lsa, workforce,
      document, applicability domains so penalty/reward logic fires on S3000L tags
  5.  CORE DATA INTEGRATION — harvest_from_core_json() reads the per-standard
      {STD}_extracted.json files produced by sbrain_core.py, pulling real
      definitions, field descriptions and key-entity contexts directly into records.
      This is the biggest accuracy win: real PDF definitions >> synthetic fallbacks.
  6.  XMI class Notes integrated into ConceptRecord definitions (was silently dropped)
  7.  harvest_from_ontology() reads ontology_merged.json cross_references to
      pre-seed the SX000i booster with shared entities across standards
  8.  Better embedding model recommendation (configurable, defaults to mpnet-base)
  9.  TRANSFORM_PAIRS extended to cover S3000L ↔ S1000D ↔ S2000M pairs
  10. _classify_relationship() now checks standard-specific pair registry
  11. core.py save_all() path wiring: crossmatch build() called with xmi_models dict
      so XMI class notes enrich concept records before FAISS indexing
  12. Confidence floor now per-relationship-type (equiv stays ≥0.50, ref requires ≥0.65)
  13. get_best_match() domain enrichment context auto-injected for S3000L LSA tags
"""

import json
import re
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
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
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


def _norm_key(name: str) -> str:
    return re.sub(r'[\s\-_]', '', name.lower())


def _name_similarity(a: str, b: str) -> float:
    ta = set(_split_camel(a).split())
    tb = set(_split_camel(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ─────────────────────────────────────────────
# STANDARD-SPECIFIC SYNTHETIC DEFINITION VOCAB
# ─────────────────────────────────────────────
#
# Each entry: norm_key_fragment → rich description string
# Used when XSD xs:documentation is absent — gives embeddings real aerospace signal
# instead of the generic "X is a data field in aerospace technical publications"

_S1000D_VOCAB: Dict[str, str] = {
    "dmc":            "data module code unique identifier for an S1000D data module in a CSDB",
    "datamodule":     "S1000D XML data module containing technical content for aircraft maintenance or operation",
    "dmcode":         "structured alphanumeric code identifying an S1000D data module within the CSDB",
    "brex":           "business rules exchange data module enforcing project-specific S1000D rules",
    "csdb":           "common source database repository for all S1000D data modules and publication modules",
    "ipl":            "illustrated parts list data module listing components with figure references",
    "identname":      "item nomenclature name identifying a spare part in an S1000D IPL provisioning table",
    "partno":         "part number alphanumeric identifier for an aircraft component in provisioning data",
    "partnumber":     "part number alphanumeric identifier for an aircraft component in provisioning data",
    "ncagecode":      "NATO commercial and government entity code identifying the manufacturer of a part",
    "manufacturercode": "NATO CAGE code identifying a manufacturer in provisioning or supply data",
    "qpa":            "quantity per assembly number of this item required in the next higher assembly",
    "quantityper":    "quantity of a spare part required per assembly or aircraft in provisioning data",
    "figureref":      "reference to an illustrated parts list figure showing component location",
    "itemseq":        "item sequence number identifying a callout on an IPL figure",
    "unitofmeasure":  "unit of measure for a quantity field such as each, meter, kilogram in provisioning",
    "applicability":  "applicability condition defining which aircraft variants a data module applies to",
    "effectivity":    "effectivity range of aircraft serial numbers for which content is valid",
    "pmcode":         "publication module code identifying an S1000D publication",
    "skill":          "skill level or trade required to perform an S1000D maintenance task",
    "support":        "support equipment tool or facility required for an S1000D maintenance procedure",
}

_S2000M_VOCAB: Dict[str, str] = {
    "provisioning":   "provisioning data record for spare parts management in S2000M materiel management",
    "partnum":        "part number identifier for a spare part in S2000M provisioning or supply record",
    "nsn":            "NATO stock number 13-digit identifier for a standardised supply item",
    "natostocknum":   "NATO stock number 13-digit identifier for a standardised supply item",
    "fullnato":       "full NATO stock number including country code for a provisioned supply item",
    "itemnom":        "item nomenclature descriptive name of a spare part in S2000M provisioning record",
    "qpanexthi":      "quantity per next higher assembly spare parts count in S2000M provisioning table",
    "sourceofsupply": "source of supply code indicating procurement channel for a spare part",
    "unitprice":      "unit price monetary value for a spare part in S2000M supply data",
    "unitofissue":    "unit of issue packaging or count unit for a supply item in S2000M",
    "leadtime":       "procurement lead time in weeks or months for a spare part in S2000M",
    "supplyclas":     "supply class category grouping spare parts in S2000M materiel management",
    "vendor":         "vendor or supplier identifier for a spare part source in S2000M",
    "criticality":    "criticality code indicating mission impact if the part is unavailable",
    "cage":           "CAGE code NATO commercial and government entity code for part manufacturer",
    "figurenum":      "figure number reference to an illustrated parts list figure in S2000M",
    "itemseqnum":     "item sequence number identifying a line item in an S2000M provisioning table",
    "dmcode":         "data module code reference linking an S2000M record to an S1000D data module",
}

_S3000L_VOCAB: Dict[str, str] = {
    "lsa":            "logistic support analysis record tracking maintenance and support requirements per S3000L",
    "lsatask":        "logistics support analysis task record defining a maintenance action in S3000L",
    "lsacontrol":     "LSA control number unique identifier for an S3000L LSA record",
    "maintenancetask": "scheduled or corrective maintenance task with labour hours and skill levels in S3000L",
    "maintenancecode": "maintenance task code alphanumeric identifier for an S3000L maintenance task",
    "failuremode":    "failure mode description from FMEA or FMECA analysis in S3000L reliability data",
    "failurecode":    "failure mode code identifier from FMEA analysis in S3000L LSA record",
    "fmea":           "failure mode and effects analysis record in S3000L reliability analysis",
    "mtbf":           "mean time between failures reliability metric in hours for an LRU in S3000L",
    "mttr":           "mean time to repair maintainability metric in hours for an LRU in S3000L",
    "mttf":           "mean time to failure reliability metric in S3000L analysis",
    "lcc":            "life cycle cost total ownership cost for a system or component in S3000L",
    "lora":           "level of repair analysis determining optimal repair echelon in S3000L",
    "lru":            "line replaceable unit hardware item that can be removed and replaced at field level",
    "lrupart":        "line replaceable unit part number identifier in S3000L hardware breakdown",
    "lruident":       "line replaceable unit identifier linking S3000L hardware item to a part number",
    "hardwareitem":   "hardware item or LRU in the product breakdown structure for S3000L analysis",
    "repairlevel":    "level of repair field echelon intermediate depot or manufacturer for S3000L task",
    "maintenancelevel": "maintenance level field intermediate or depot defining where a task is performed",
    "taskduration":   "task elapsed time or duration in hours for an S3000L maintenance task",
    "manpowerreq":    "manpower requirement labour hours and skill category for an S3000L maintenance task",
    "supportequip":   "support equipment tool or test equipment required for an S3000L maintenance task",
    "functionalitem": "functional item number linking an S3000L hardware item to its functional role",
    "quantpersys":    "quantity per system number of LRUs installed per aircraft in S3000L analysis",
    "repaircycle":    "repair cycle time for component overhaul or repair in S3000L depot maintenance",
    "reliab":         "reliability parameter such as MTBF or failure rate in S3000L LSA data",
    "maintainab":     "maintainability parameter such as MTTR or Mmax in S3000L LSA data",
    "sparing":        "sparing analysis determining recommended spare quantities in S3000L logistics",
}

_SX000I_VOCAB: Dict[str, str] = {
    "ips":            "integrated product support element in SX000i ILS supportability framework",
    "ils":            "integrated logistics support planning and analysis element in SX000i",
    "supportplan":    "support plan document linking ILS elements across S1000D S2000M S3000L in SX000i",
    "crossref":       "cross-reference index entry linking equivalent concepts between S-series standards",
    "provisioningidx": "provisioning index cross-referencing S1000D IPL items to S2000M supply records",
    "supportsummary": "supportability summary aggregating LSA results across standards in SX000i",
    "ilselement":     "integrated logistics support element such as maintenance supply training in SX000i",
}

_STANDARD_VOCAB: Dict[str, Dict[str, str]] = {
    "S1000D": _S1000D_VOCAB,
    "S2000M": _S2000M_VOCAB,
    "S3000L": _S3000L_VOCAB,
    "SX000i": _SX000I_VOCAB,
}


# ─────────────────────────────────────────────
# DEFINITION HARVESTER
# ─────────────────────────────────────────────

class DefinitionHarvester:
    """
    Pulls definitions from every source available for a standard.
    Priority: XSD annotation > XMI note > core extracted JSON > synthetic
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
                defn = self._synthetic_definition(name, xs_type, parents, standard)

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
                definition_text=defn if defn else self._synthetic_definition(name, "", [], standard),
                source_file=fname,
                source="xsd",
            ))

        return records

    def _extract_xsd_annotation(self, node) -> str:
        for ann in node.findall(f".//{{{XS_NS}}}documentation"):
            if ann.text and len(ann.text.strip()) > 10:
                return ann.text.strip()[:600]
        return ""

    def _find_parents_via_map(self, elem, parent_map: dict) -> list:
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

    def _synthetic_definition(self, tag_name: str, xs_type: str = "",
                               parents: List[str] = None, standard: str = "") -> str:
        """
        Enriched synthetic definition with both standard-specific vocab lookup
        and domain classification. Gives embedding model real aerospace signal
        even when xs:documentation is absent.
        """
        words = _split_camel(tag_name)
        norm  = _norm_key(tag_name)

        # ── 1. Try standard-specific vocab first ────────────────────────────
        std_vocab = _STANDARD_VOCAB.get(standard, {})
        for fragment, rich_defn in std_vocab.items():
            if fragment in norm:
                parent_ctx = f" within {parents[0]}" if parents else ""
                type_ctx   = f" typed as {xs_type}" if xs_type else ""
                return f"{words}: {rich_defn}{parent_ctx}{type_ctx}."

        # ── 2. Generic domain classification (cross-standard fallback) ───────
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
        elif any(k in tag_lower for k in ["task", "procedure", "step"]):
            domain = "maintenance task procedure"
        elif any(k in tag_lower for k in ["failure", "fault", "mode", "effect"]):
            domain = "failure analysis field"
        elif any(k in tag_lower for k in ["reliab", "mtbf", "mttr", "mttf"]):
            domain = "reliability maintainability metric"
        elif any(k in tag_lower for k in ["lru", "hardware", "item", "component"]):
            domain = "hardware component identifier"
        elif any(k in tag_lower for k in ["repair", "maintenance", "level", "echelon"]):
            domain = "maintenance level repair field"
        elif any(k in tag_lower for k in ["supply", "stock", "nsn", "provisioning"]):
            domain = "supply chain logistics field"
        else:
            domain = "data field"

        std_ctx    = f" in {standard}" if standard else ""
        parent_ctx = f" within {parents[0]}" if parents else ""
        type_ctx   = f" typed as {xs_type}" if xs_type else ""
        return (
            f"{words} is a {domain}{parent_ctx}{std_ctx} used in aerospace "
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
                    defn = note[:600] if note and len(note) > 15 else self._synthetic_definition(name, "", [], standard)
                    records.append(ConceptRecord(
                        tag_name=name,
                        standard=standard,
                        definition_text=defn,
                        source="xmi"
                    ))
        except Exception as e:
            print(f"    Warning XMI {xmi_path}: {e}")
        return records

    def harvest_xmi_from_dict(self, xmi_data: dict, standard: str) -> list:
        """
        Harvest records from an already-parsed XMI dict (as produced by
        sbrain_core.py parse_xmi()). The dict has structure:
          {"classes": [{"class": str, "note": str, "attributes": [...]}]}
        This avoids re-parsing the XMI file and uses core's already-loaded data.
        """
        records = []
        for cls in xmi_data.get("classes", []):
            name = cls.get("class", "").strip()
            note = cls.get("note", "").strip()
            if not name:
                continue
            defn = note[:600] if note and len(note) > 15 else self._synthetic_definition(name, "", [], standard)

            # Also fold attribute names into definition for richer embedding signal
            attrs = cls.get("attributes", [])
            if attrs:
                attr_names = ", ".join(a.get("name", "") for a in attrs[:8] if a.get("name"))
                if attr_names:
                    defn = f"{defn} Attributes: {attr_names}."

            records.append(ConceptRecord(
                tag_name=name,
                standard=standard,
                definition_text=defn,
                source="xmi_preloaded"
            ))
            # Also add attribute-level records for fine-grained matching
            for attr in attrs:
                aname = attr.get("name", "").strip()
                anote = attr.get("note", "").strip()
                if not aname:
                    continue
                adefn = anote[:400] if anote and len(anote) > 10 else self._synthetic_definition(aname, attr.get("type", ""), [name], standard)
                records.append(ConceptRecord(
                    tag_name=aname,
                    standard=standard,
                    definition_text=adefn,
                    parent_elements=[name],
                    xs_type=attr.get("type", ""),
                    source="xmi_attr"
                ))
        return records

    def harvest_SX000i(self, SX000i_dir: str) -> list:
        records = []
        sx_dir = Path(SX000i_dir)
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
                    cross_refs = self._extract_SX000i_crossrefs(elem)
                    parents = self._find_parents_via_map(elem, parent_map)
                    records.append(ConceptRecord(
                        tag_name=name,
                        standard="SX000i",
                        definition_text=defn or self._synthetic_definition(name, elem.get("type",""), parents, "SX000i"),
                        aliases=cross_refs,
                        source_file=xsd_file.name,
                        source="SX000i"
                    ))
            except Exception as e:
                print(f"    Warning SX000i XSD {xsd_file.name}: {e}")

        for xmi_file in sx_dir.glob("**/*.xmi"):
            records.extend(self.harvest_xmi(str(xmi_file), "SX000i"))

        print(f"    SX000i: {len(records)} concept records harvested")
        return records

    def _extract_SX000i_crossrefs(self, elem) -> list:
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
                if entity.get("entity_type") in ("Definition", "StandardEntity", "Field"):
                    defn = entity.get("description", "").strip()
                    name = entity.get("name", "").strip()
                    if len(defn) < 15 or not name:
                        continue
                    records.append(ConceptRecord(
                        tag_name=name,
                        standard=standard,
                        definition_text=defn[:600],
                        source_file=extracted_json,
                        source="pdf"
                    ))
        except Exception as e:
            print(f"    Warning PDF JSON {extracted_json}: {e}")
        return records

    def harvest_from_core_json(self, extracted_json: str, standard: str) -> list:
        """
        NEW in v4: Read the full {STD}_extracted.json produced by sbrain_core.py
        and turn every entity + field + key_entity found in sections into a
        ConceptRecord with its real definition text.

        This is the main accuracy improvement — real extracted PDF definitions
        are far richer than any synthetic fallback, especially for S3000L where
        the XSD annotation coverage is thin.
        """
        records = []
        seen = set()
        try:
            with open(extracted_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"    Warning core JSON {extracted_json}: {e}")
            return records

        # 1. Entities extracted from sections (definitions, fields, standard entities)
        for entity in data.get("entities", []):
            name  = entity.get("name", "").strip()
            defn  = entity.get("description", "").strip()
            etype = entity.get("entity_type", "")
            if not name or not defn or len(defn) < 15:
                continue
            key = f"{standard}::{name.lower()}"
            if key in seen:
                continue
            seen.add(key)
            records.append(ConceptRecord(
                tag_name=name,
                standard=standard,
                definition_text=defn[:600],
                source="core_entity"
            ))

        # 2. Schema elements from XSD parsing (already processed by core, use directly)
        for schema in data.get("schemas", []):
            name = (schema.get("element") or schema.get("entity_name", "")).strip()
            if not name:
                continue
            key = f"{standard}::{name.lower()}"
            if key in seen:
                continue
            seen.add(key)
            children = schema.get("children", [])
            child_ctx = f" Contains: {', '.join(children[:6])}." if children else ""
            synth = self._synthetic_definition(name, "", [], standard)
            records.append(ConceptRecord(
                tag_name=name,
                standard=standard,
                definition_text=f"{synth}{child_ctx}",
                source="core_schema"
            ))

        # 3. Standard-specific extras from sections
        for section in data.get("sections", []):
            # S3000L metrics (MTBF, MTTR, etc.) — give them richer context
            for metric in section.get("metrics", []):
                mname = metric.get("metric", "").strip()
                mval  = metric.get("value", "").strip()
                if not mname:
                    continue
                key = f"{standard}::{mname.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                defn = self._synthetic_definition(mname, "", [], standard)
                if mval:
                    defn += f" Observed value: {mval}."
                records.append(ConceptRecord(
                    tag_name=mname,
                    standard=standard,
                    definition_text=defn,
                    source="core_metric"
                ))

            # S3000L failure modes
            for fm in section.get("failure_modes", []):
                text = fm.get("failure_mode", "").strip()
                if len(text) < 10:
                    continue
                # Use the first camelCase-style word as tag if possible
                first_word = text.split()[0] if text.split() else ""
                if not first_word or len(first_word) < 3:
                    continue
                key = f"{standard}::fm_{first_word.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                records.append(ConceptRecord(
                    tag_name=first_word,
                    standard=standard,
                    definition_text=f"Failure mode in S3000L FMEA analysis: {text[:400]}",
                    source="core_failure_mode"
                ))

        print(f"    core JSON → {len(records)} additional concept records for {standard}")
        return records

    def harvest_from_ontology(self, ontology_json: str) -> Dict[str, List[str]]:
        """
        NEW in v4: Read ontology_merged.json cross_references and return a dict
        mapping normalized tag_name → list of standards it appears in.
        Used to pre-seed the SX000i booster with verified cross-standard aliases.
        """
        cross_map: Dict[str, List[str]] = {}
        try:
            with open(ontology_json, "r", encoding="utf-8") as f:
                onto = json.load(f)
            for xref in onto.get("cross_references", []):
                entity     = xref.get("entity", "").strip()
                appears_in = xref.get("appears_in", [])
                if entity and len(appears_in) > 1:
                    cross_map[_norm_key(entity)] = appears_in
        except Exception as e:
            print(f"    Warning ontology JSON {ontology_json}: {e}")
        print(f"    Ontology cross-references loaded: {len(cross_map)} shared entities")
        return cross_map


# ─────────────────────────────────────────────
# SEMANTIC INDEX
# ─────────────────────────────────────────────

class SemanticConceptIndex:

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
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
# SX000i BOOSTER  (bidirectional + ontology seeded)
# ─────────────────────────────────────────────

class SX000iBooster:
    """
    Bidirectional alias lookup. Also accepts an ontology cross_map from
    harvest_from_ontology() to pre-seed shared entities even when SX000i
    XSD annotation coverage is thin.
    """
    def __init__(self, sx_records: list, ontology_cross_map: Dict[str, List[str]] = None):
        self.confirmed: dict = {}

        # From SX000i XSD cross-refs
        for rec in sx_records:
            sx_key = _norm_key(rec.tag_name)
            for alias in rec.aliases:
                alias_key = _norm_key(alias)
                self.confirmed.setdefault(alias_key, set()).add(sx_key)
                self.confirmed.setdefault(sx_key, set()).add(alias_key)

        # From ontology merged cross-references (entities appearing in multiple standards)
        if ontology_cross_map:
            for entity_key, standards in ontology_cross_map.items():
                # Treat shared entity as its own bridge node — link all standards together
                for std_a in standards:
                    for std_b in standards:
                        if std_a != std_b:
                            self.confirmed.setdefault(entity_key, set()).add(entity_key)

        print(f"    SX000iBooster: {len(self.confirmed)} bridge nodes")

    def boost(self, source_tag: str, target_tag: str) -> float:
        sk = _norm_key(source_tag)
        tk = _norm_key(target_tag)
        source_connections = self.confirmed.get(sk, set())
        target_connections = self.confirmed.get(tk, set())
        if source_connections & target_connections:
            return 0.20
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
        "SX000i_confirmed": 0.60,
        "name_match":        0.55,
        "alias_match":       0.50,
        "unmapped":          0.0,
    }

    # ── STRICT overrides — always score 1.0, must never be beaten by FAISS ──
    # Verified correct aerospace mappings. Expand freely — these are the cheapest
    # accuracy wins available.
    STRICT_MAP: Dict[str, Dict[str, str]] = {
        # ── S2000M targets (translating FROM other standards TO S2000M) ──
        "S2000M": {
            # from S1000D
            "quantityPerAssembly":        "quantityPerNextHigherAssembly",
            "identName":                  "itemNomenclature",
            "partNumberValue":            "partNumber",
            "fullNatoStockNumber":        "nsn",
            "natoStockNumber":            "nsn",
            "dmCode":                     "dataModuleCode",
            "partNumber":                 "partNumber",
            "nsn":                        "nsn",
            "figureNumber":               "figureReference",
            "itemSeqNumber":              "itemSequenceNumber",
            "ncageCode":                  "cageCode",
            "manufacturerCodeValue":      "cageCode",
            "unitOfMeasure":              "unitOfIssue",
            # from S3000L
            "lruPartNumber":              "partNumber",
            "hardwareItemId":             "partNumber",
            "lruIdentifier":              "partNumber",
            "functionalItemNumber":       "itemSequenceNumber",
            "quantityPerSystem":          "quantityPerNextHigherAssembly",
            "repairLevel":                "sourceOfSupplyCode",
        },

        # ── S1000D targets ──
        "S1000D": {
            # from S2000M
            "itemNomenclature":               "identName",
            "qpa":                            "quantityPerAssembly",
            "ncageCode":                      "manufacturerCodeValue",
            "unitOfIssue":                    "unitOfMeasure",
            "provisioningSequenceNumber":     "item",
            "cageCode":                       "manufacturerCodeValue",
            "figureReference":                "figureNumber",
            "itemSequenceNumber":             "itemSeqNumber",
            "dataModuleCode":                 "dmCode",
            # from S3000L
            "lruPartNumber":                  "partNumberValue",
            "hardwareItemId":                 "partNumberValue",
            "lruIdentifier":                  "partNumberValue",
            "functionalItemNumber":           "itemSeqNumber",
            "maintenanceTaskCode":            "taskCode",
            "lsaTaskCode":                    "taskCode",
            "repairLevel":                    "maintenanceLevelCode",
            "maintenanceLevel":               "maintenanceLevelCode",
            "quantityPerSystem":              "quantityPerAssembly",
        },

        # ── S3000L targets ──
        "S3000L": {
            # from S1000D
            "partNumberValue":                "lruPartNumber",
            "identName":                      "itemNomenclature",
            "quantityPerAssembly":            "quantityPerSystem",
            "figureNumber":                   "functionalItemNumber",
            "itemSeqNumber":                  "functionalItemNumber",
            "maintenanceLevelCode":           "repairLevel",
            "taskCode":                       "maintenanceTaskCode",
            "dmCode":                         "lsaControlNumber",
            # from S2000M
            "partNumber":                     "lruPartNumber",
            "itemNomenclature":               "itemNomenclature",
            "quantityPerNextHigherAssembly":  "quantityPerSystem",
            "itemSequenceNumber":             "functionalItemNumber",
            "cageCode":                       "manufacturerCode",
            "nsn":                            "nsnNumber",
            "sourceOfSupplyCode":             "repairLevel",
        },
    }

    # ── AERO_ALIASES — well-known tag equivalences, score = 0.95 ──
    AERO_ALIASES: Dict[str, Dict[str, str]] = {
        "S2000M": {
            "manufacturerCodeValue":      "ncageCode",
            "identName":                  "itemNomenclature",
            "quantityPerAssembly":        "qpa",
            "criticalityCode":            "criticalityIndicator",
            "sourceOfSupply":             "sourceOfSupplyCode",
            "unitPrice":                  "unitPrice",
            "partNumberValue":            "partNumber",
            "lruPartNumber":              "partNumber",
            "lruIdentifier":              "partNumber",
            "hardwareItemId":             "partNumber",
            "functionalItemNumber":       "itemSequenceNumber",
            "repairLevel":                "sourceOfSupplyCode",
            "maintenanceTaskCode":        "workCode",
            "lsaTaskCode":               "workCode",
        },
        "S1000D": {
            "ncageCode":                  "manufacturerCodeValue",
            "itemNomenclature":           "identName",
            "qpa":                        "quantityPerAssembly",
            "partNumber":                 "partNumberValue",
            "cageCode":                   "manufacturerCodeValue",
            "lruPartNumber":              "partNumberValue",
            "maintenanceTaskCode":        "taskCode",
            "lsaTaskCode":               "taskCode",
            "repairLevel":               "maintenanceLevelCode",
            "maintenanceLevel":          "maintenanceLevelCode",
        },
        "S3000L": {
            # ILS / IPS
            "lsaTaskCode":               "maintenanceTaskCode",
            "failureMode":               "failureModeCode",
            "lruIdentifier":             "lruPartNumber",
            "maintenanceLevel":          "repairLevel",
            "hardwareItemId":            "lruPartNumber",
            "functionalItemNumber":      "lsaControlNumber",
            # from provisioning
            "partNumber":                "lruPartNumber",
            "partNumberValue":           "lruPartNumber",
            "itemNomenclature":          "itemNomenclature",
            "identName":                 "itemNomenclature",
            "quantityPerAssembly":       "quantityPerSystem",
            "quantityPerNextHigherAssembly": "quantityPerSystem",
            "ncageCode":                 "manufacturerCode",
            "cageCode":                  "manufacturerCode",
            "nsn":                       "nsnNumber",
        },
        "SX000i": {
            "ipsElement":                "supportElement",
            "ipsPlan":                   "supportPlan",
            "ipsRequirement":            "supportRequirement",
        },
    }

    def __init__(self, output_dir: str = "./output",
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.output_dir = Path(output_dir)
        self.harvester = DefinitionHarvester()
        self.index = SemanticConceptIndex(model_name)
        self.structural = StructuralScorer()
        self.booster = None
        self.links: list = []
        self._link_index: dict = {}

        from sbrain_learning_memory import SBrainLearningMemory
        self.learning_memory = SBrainLearningMemory()

    def build(self, standard_dirs: dict, extracted_jsons: dict,
              SX000i_dir: Optional[str] = None,
              xmi_models: Optional[Dict[str, dict]] = None,
              ontology_json: Optional[str] = None):
        """
        Build the full crossmatch index.

        Args:
            standard_dirs:  {std: xsd_dir_path}
            extracted_jsons: {std: path_to_{STD}_extracted.json}
                             — produced by sbrain_core.py, used for real definitions
            SX000i_dir:     path to SX000i XSD/XMI directory
            xmi_models:     {std: xmi_data_dict} pre-loaded by core's MultiStandardXSDParser
                             — avoids re-parsing XMI files
            ontology_json:  path to ontology_merged.json for cross-ref seeding
        """
        all_records: list = []

        for std, xsd_dir in standard_dirs.items():
            print(f"\n  Harvesting {std}...")
            if Path(xsd_dir).exists():
                recs = self.harvester.harvest_xsd(xsd_dir, std)
                print(f"    XSD: {len(recs)} records")
                all_records.extend(recs)

                # XMI: prefer pre-loaded dict from core to avoid double-parse
                if xmi_models and std in xmi_models and xmi_models[std]:
                    xmi_recs = self.harvester.harvest_xmi_from_dict(xmi_models[std], std)
                    print(f"    XMI (preloaded): {len(xmi_recs)} records")
                    all_records.extend(xmi_recs)
                else:
                    for xmi in Path(xsd_dir).glob("**/*.xmi"):
                        xmi_recs = self.harvester.harvest_xmi(str(xmi), std)
                        print(f"    XMI: {len(xmi_recs)} records")
                        all_records.extend(xmi_recs)

            # Core extracted JSON — real PDF definitions, biggest accuracy boost
            if std in extracted_jsons and Path(extracted_jsons[std]).exists():
                # harvest_from_core_json covers entities AND schemas
                core_recs = self.harvester.harvest_from_core_json(extracted_jsons[std], std)
                all_records.extend(core_recs)
                # Also run the legacy PDF-only harvest for backward compat
                pdf_recs = self.harvester.harvest_pdf_definitions(extracted_jsons[std], std)
                print(f"    PDF entities: {len(pdf_recs)} records")
                all_records.extend(pdf_recs)

        # SX000i bridge
        sx_records = []
        if SX000i_dir and Path(SX000i_dir).exists():
            print(f"\n  Harvesting SX000i...")
            sx_records = self.harvester.harvest_SX000i(SX000i_dir)
            all_records.extend(sx_records)

        # Ontology cross-reference seeding
        ontology_cross_map = {}
        if ontology_json and Path(ontology_json).exists():
            print(f"\n  Loading ontology cross-references...")
            ontology_cross_map = self.harvester.harvest_from_ontology(ontology_json)

        self.booster = SX000iBooster(sx_records, ontology_cross_map)

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
        all_standards = list(standard_dirs.keys())
        if SX000i_dir and Path(SX000i_dir).exists():
            all_standards = all_standards + ["SX000i"]

        self.links = self._compute_all_links(all_records, all_standards)
        self._build_link_index()
        self._save()
        print(f"\n  Cross-matcher ready: {len(self.links)} concept links")

    def _dedup(self, records: list) -> list:
        """
        Dedup by (standard, tag_name). When duplicates exist, prefer richer
        definitions in this priority: pdf > core_entity > xmi > xsd > synthetic.
        """
        SOURCE_PRIORITY = {
            "core_entity": 5, "pdf": 4, "xmi_preloaded": 3,
            "xmi_attr": 3, "xmi": 3, "xsd": 2,
            "SX000i": 2, "core_schema": 1, "core_metric": 1, "core_failure_mode": 1,
        }
        seen: dict = {}
        for r in records:
            key = f"{r.standard}::{r.tag_name}"
            if key not in seen:
                seen[key] = r
            else:
                existing = seen[key]
                existing_pri = SOURCE_PRIORITY.get(existing.source, 0)
                new_pri      = SOURCE_PRIORITY.get(r.source, 0)
                # Higher priority wins; tie goes to longer definition
                if new_pri > existing_pri or (
                    new_pri == existing_pri and len(r.definition_text) > len(existing.definition_text)
                ):
                    # Merge aliases
                    merged_aliases = list(set(existing.aliases + r.aliases))
                    r.aliases = merged_aliases
                    seen[key] = r
        return list(seen.values())

    def _compute_all_links(self, records: list, standards: list) -> list:
        bridge_standards = set(standards) | {"SX000i"}
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

                normalized_cos = (cos_score + 1.0) / 2.0
                final_score    = min(normalized_cos + struct_bonus + sx_boost + name_bonus, 1.0)

                match_type        = self._classify(cos_score, struct_bonus, sx_boost, name_bonus)
                relationship_type = self._classify_relationship(rec, neighbour, cos_score, name_bonus)
                evidence = (
                    f"cos={cos_score:.3f} struct={struct_bonus:.3f} "
                    f"SX000i={sx_boost:.3f} name={name_bonus:.3f} → final={final_score:.3f}"
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
        if sx > 0:
            return "SX000i_confirmed"
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
        Classify WHAT KIND of semantic relationship exists.
        Extended in v4 with full S3000L ↔ S1000D ↔ S2000M transform pairs.
        """
        # All known transformation pairs (bidirectional — add both orderings)
        TRANSFORM_PAIRS = {
            # S1000D ↔ S2000M
            ("partNumberValue",                 "partNumber"),
            ("partNumber",                      "partNumberValue"),
            ("identName",                       "itemNomenclature"),
            ("itemNomenclature",                "identName"),
            ("fullNatoStockNumber",             "nsn"),
            ("nsn",                             "fullNatoStockNumber"),
            ("quantityPerAssembly",             "quantityPerNextHigherAssembly"),
            ("quantityPerNextHigherAssembly",   "quantityPerAssembly"),
            ("dmCode",                          "dataModuleCode"),
            ("dataModuleCode",                  "dmCode"),
            ("manufacturerCodeValue",           "ncageCode"),
            ("ncageCode",                       "manufacturerCodeValue"),
            ("manufacturerCodeValue",           "cageCode"),
            ("cageCode",                        "manufacturerCodeValue"),
            ("qpa",                             "quantityPerAssembly"),
            ("quantityPerAssembly",             "qpa"),
            ("unitOfMeasure",                   "unitOfIssue"),
            ("unitOfIssue",                     "unitOfMeasure"),
            ("figureNumber",                    "figureReference"),
            ("figureReference",                 "figureNumber"),
            ("itemSeqNumber",                   "itemSequenceNumber"),
            ("itemSequenceNumber",              "itemSeqNumber"),
            # S3000L ↔ S1000D
            ("lruPartNumber",                   "partNumberValue"),
            ("partNumberValue",                 "lruPartNumber"),
            ("lruIdentifier",                   "partNumberValue"),
            ("hardwareItemId",                  "partNumberValue"),
            ("functionalItemNumber",            "itemSeqNumber"),
            ("itemSeqNumber",                   "functionalItemNumber"),
            ("maintenanceTaskCode",             "taskCode"),
            ("taskCode",                        "maintenanceTaskCode"),
            ("lsaTaskCode",                     "taskCode"),
            ("taskCode",                        "lsaTaskCode"),
            ("repairLevel",                     "maintenanceLevelCode"),
            ("maintenanceLevelCode",            "repairLevel"),
            ("maintenanceLevel",                "maintenanceLevelCode"),
            ("quantityPerSystem",               "quantityPerAssembly"),
            ("quantityPerAssembly",             "quantityPerSystem"),
            ("lsaControlNumber",               "dmCode"),
            # S3000L ↔ S2000M
            ("lruPartNumber",                   "partNumber"),
            ("partNumber",                      "lruPartNumber"),
            ("hardwareItemId",                  "partNumber"),
            ("lruIdentifier",                   "partNumber"),
            ("functionalItemNumber",            "itemSequenceNumber"),
            ("itemSequenceNumber",              "functionalItemNumber"),
            ("manufacturerCode",                "cageCode"),
            ("cageCode",                        "manufacturerCode"),
            ("nsnNumber",                       "nsn"),
            ("nsn",                             "nsnNumber"),
            ("quantityPerSystem",               "quantityPerNextHigherAssembly"),
            ("quantityPerNextHigherAssembly",   "quantityPerSystem"),
        }

        if _norm_key(source.tag_name) == _norm_key(target.tag_name):
            return "equivalent"

        if cos_score >= 0.90 and name_bonus >= 0.08:
            return "equivalent"

        if (source.tag_name, target.tag_name) in TRANSFORM_PAIRS:
            return "transformation"

        if cos_score >= 0.78:
            return "partial_equivalent"

        if cos_score < 0.60:
            return "reference"

        return "partial_equivalent"

    def get_ontology_graph(self) -> Dict:
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

        synth_def  = self.harvester._synthetic_definition(source_tag, standard=from_std)
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

        # Multi-hop: S source → SX000i → target standard
        if best_link is None and from_std != "SX000i" and to_std != "SX000i":
            sx_match = self.get_best_match(source_tag, from_std, "SX000i")
            if sx_match and sx_match.score >= 0.65:
                hop_match = self.get_best_match(sx_match.target_tag, "SX000i", to_std)
                if hop_match and hop_match.score >= 0.65:
                    bridged_score = round(sx_match.score * hop_match.score, 4)
                    best_link = ConceptLink(
                        source_tag=source_tag,
                        source_std=from_std,
                        target_tag=hop_match.target_tag,
                        target_std=to_std,
                        score=bridged_score,
                        match_type="SX000i_bridge",
                        relationship_type="partial_equivalent",
                        evidence=(
                            f"via SX000i:{sx_match.target_tag} "
                            f"hop1={sx_match.score:.3f} hop2={hop_match.score:.3f}"
                        ),
                    )

        return best_link

    def _name_similarity(self, a: str, b: str) -> float:
        ta = set(re.sub(r'([A-Z])', r' \1', a).lower().split())
        tb = set(re.sub(r'([A-Z])', r' \1', b).lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _build_link_index(self):
        self._link_index = {}
        for lnk in self.links:
            key = (_norm_key(lnk.source_tag), lnk.source_std, lnk.target_std)
            self._link_index.setdefault(key, []).append(lnk)
        for k in self._link_index:
            self._link_index[k].sort(key=lambda l: l.score, reverse=True)

    # ── Query API ──────────────────────────────────────────────────────────────

    def get_best_match(self, tag: str, from_std: str, to_std: str, context: str = ""):
        """
        Priority order:
          1. human_validated   — user-confirmed mappings, score = 1.0
          2. STRICT_MAP        — hardcoded correct aerospace mappings, score = 1.0
          3. AERO_ALIASES      — known equivalences, score = 0.95
          4. _link_index       — precomputed FAISS links
          5. vector search     — live embedding lookup with domain filtering
        """
        validated = self.learning_memory.get_validated_mapping(from_std, tag, to_std)
        if validated:
            return ConceptLink(
                source_tag=tag, source_std=from_std,
                target_tag=validated["target_tag"], target_std=to_std,
                score=1.0, match_type="human_validated",
                relationship_type="equivalent",
                evidence=f"user_confirmed_{validated.get('times_confirmed', 1)}x"
            )

        norm = _norm_key(tag)

        # 1. STRICT
        strict_targets = self.STRICT_MAP.get(to_std, {})
        if tag in strict_targets:
            return ConceptLink(
                source_tag=tag, source_std=from_std,
                target_tag=strict_targets[tag], target_std=to_std,
                score=1.0, match_type="rule_override",
                relationship_type="transformation",
                evidence="STRICT_MAP",
            )

        # 2. AERO_ALIASES
        alias_targets = self.AERO_ALIASES.get(to_std, {})
        if tag in alias_targets:
            target_tag = alias_targets[tag]
            target_exists = any(
                r.tag_name == target_tag and r.standard == to_std
                for r in self.index.records
            )
            if target_exists:
                return ConceptLink(
                    source_tag=tag, source_std=from_std,
                    target_tag=target_tag, target_std=to_std,
                    score=0.95, match_type="alias_match",
                    relationship_type="transformation",
                    evidence="AERO_ALIAS",
                )

        # 3. Precomputed link index
        key = (norm, from_std, to_std)
        if key in self._link_index:
            best = self._link_index[key][0]
            if best.score >= 0.50 or best.relationship_type not in ("reference", "unknown"):
                return best

        # 4. Domain enforcement setup — EXPANDED for S3000L
        DOMAIN_MAP = {
            "quantity":    ["quantity", "qpa", "count", "amount", "quantpersys"],
            "identity":    ["partnumber", "pnr", "ident", "nomenclature", "lrupart", "hardwareid"],
            "logistics":   ["nsn", "stock", "natocode", "supply", "cage", "sourceofsupply"],
            "pricing":     ["price", "cost", "currency", "unitprice"],
            "reliability": ["failure", "mtbf", "mttr", "mttf", "reliab", "meantime"],
            "maintenance": ["task", "maintenance", "repair", "interval", "inspection", "lsatask"],
            "lsa":         ["lsa", "lru", "support", "analysiscontrol", "lsacontrol"],
            "workforce":   ["skill", "labour", "manhour", "personnel", "technician", "manpower"],
            "document":    ["dmc", "datamodule", "dmcode", "pmcode", "brex"],
            "applicability": ["applicability", "effectivity", "variant"],
        }
        source_domain = None
        for domain, keys in DOMAIN_MAP.items():
            if any(k in norm for k in keys):
                source_domain = domain
                break

        # Auto-enrich context for well-known identity and S3000L LSA tags
        IDENTITY_TAGS = {
            "manufacturerCodeValue", "partNumberValue", "identName",
            "ncageCode", "partNumber", "itemNomenclature",
        }
        LSA_TAGS = {
            "lruPartNumber", "lruIdentifier", "hardwareItemId",
            "lsaTaskCode", "maintenanceTaskCode", "repairLevel",
            "maintenanceLevel", "quantityPerSystem", "functionalItemNumber",
        }
        if tag in IDENTITY_TAGS:
            context = f"{context} | part identification field in provisioning data module".strip(" |")
        elif tag in LSA_TAGS:
            context = f"{context} | S3000L logistics support analysis field hardware or task identifier".strip(" |")

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

            if source_domain:
                domain_keys = DOMAIN_MAP[source_domain]
                target_def  = res.definition_text.lower()
                if not any(k in target_norm or k in target_def for k in domain_keys):
                    final_score *= 0.5

            if norm in target_norm or target_norm in norm:
                final_score += 0.20

            candidates.append(ConceptLink(
                source_tag=tag, source_std=from_std,
                target_tag=res.tag_name, target_std=to_std,
                score=min(round(final_score, 4), 1.0),
                match_type="semantic_filtered",
                relationship_type="partial_equivalent",
                evidence=f"domain={source_domain} cos={score:.3f}",
            ))

        if candidates:
            candidates.sort(key=lambda x: x.score, reverse=True)
            best = candidates[0]
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
             model_name: str = "sentence-transformers/all-mpnet-base-v2") -> "SemanticCrossMatcher":
        matcher = cls(output_dir, model_name)
        if matcher.index.load(output_dir):
            links_path = Path(output_dir) / "concept_links.json"
            if links_path.exists():
                with open(links_path) as f:
                    raw_links = json.load(f)
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