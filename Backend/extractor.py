"""
Aerospace Standards PDF Extraction Pipeline — v3 (Quality Fixes)

FIXES over v2:
  1. XML elements filtered at LINE LEVEL — CPF changelog context excluded per-match
  2. Unified _should_skip_section() gate — preamble/legal/address pages excluded
     from ALL extraction (fields, definitions, key_entities, rules, schemas)
  3. Schema extraction replaced — prose bullet matching removed; XSD parser added
     (falls back to relationship inference from changelog text if no XSD provided)
  4. Table extraction — pdfplumber added as fallback when fitz finds nothing
  5. Element hierarchy — parent/child relationships inferred from context strings

NEW PARAMETER: --xsd-dir  path to folder containing S1000D .xsd files (optional)
"""

import fitz          # PyMuPDF  (pymupdf>=1.24.0)
import pdfplumber    # pdfplumber>=0.10.0
import json
import re
import os
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm

try:
    import xml.etree.ElementTree as ET
    _HAS_ET = True
except ImportError:
    _HAS_ET = False


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

STANDARD_PROFILES = {
    "S1000D": {
        "description": "Technical Publications Standard",
        "key_entities": [
            "Data Module", "IPL", "Illustrated Parts List",
            "Data Module Code", "DMC", "BREX", "CSDB",
            "Publication Module", "Information Set",
            "Applicability", "Effectivity", "Common Source Database",
            "Business Rules", "Data Module Requirement List", "DMRL",
            "Schema", "XML", "SGML",
        ],
        "key_sections": [
            "scope", "definitions", "data module", "ipl",
            "procedure", "description", "maintenance", "repair",
            "illustrated parts", "applicability", "brex",
            "publication", "catalog", "supply", "figure",
        ],
        "schema_entities": ["DataModule", "IPLItem", "Task", "Part", "Tool", "Figure"],
        "skip_title_patterns": [
            r"^\[PREAMBLE\]",
            r"\.{5,}",
            r"^\s*\d+\s*$",
        ],
        # FIX 2: expanded skip list — legal/license/address sections
        "skip_rules_in": [
            "[preamble]", "table of contents", "copyright",
            "user agreement", "license to use", "license",
            "intellectual property", "no modifications",
        ],
    },
    "S2000M": {
        "description": "Materiel Management Standard (Logistics)",
        "key_entities": [
            "Provisioning", "Spare Part", "Supply", "Procurement",
            "Vendor", "Supplier", "Stock", "Inventory",
            "Lead Time", "Unit of Issue", "NSN", "Part Number",
            "Source of Supply", "Procurement Data", "RSPL",
            "Recommended Spare Parts List", "PTD", "CAGE Code", "MOQ",
        ],
        "key_sections": [
            "provisioning", "supply", "logistics", "materiel",
            "procurement", "inventory", "stock", "vendor",
            "spare parts", "support", "replenishment", "rspl",
        ],
        "schema_entities": ["Part", "Supplier", "InventoryItem", "ProcurementRecord", "SparePartsList"],
        "skip_title_patterns": [r"\.{5,}", r"^\[PREAMBLE\]", r"^\s*\d+\s*$"],
        "skip_rules_in": ["[preamble]", "copyright", "user agreement", "license"],
    },
    "S3000L": {
        "description": "Logistics Support Analysis Standard",
        "key_entities": [
            "LSA", "Logistics Support Analysis", "FMEA", "FMECA",
            "Failure Mode", "Maintenance Task", "LCC", "Life Cycle Cost",
            "MTBF", "MTTR", "Reliability", "Maintainability",
            "Task Analysis", "Level of Repair", "Support Equipment",
            "LORA", "Level of Repair Analysis", "LSA Record", "LSAR",
            "Corrective Maintenance", "Preventive Maintenance",
        ],
        "key_sections": [
            "lsa", "maintenance", "analysis", "failure mode",
            "reliability", "maintainability", "task", "support",
            "level of repair", "fmea", "fmeca", "life cycle",
        ],
        "schema_entities": ["MaintenanceTask", "FailureMode", "SupportEquipment", "LSARecord", "RepairLevel"],
        "skip_title_patterns": [r"\.{5,}", r"^\[PREAMBLE\]", r"^\s*\d+\s*$"],
        "skip_rules_in": ["[preamble]", "copyright", "user agreement", "license"],
    },
}

FUTURE_STANDARDS = {
    "S4000P": {"description": "Predictive Maintenance"},
    "S5000F": {"description": "Feedback of In-service Experience"},
    "S6000T": {"description": "Training"},
    "S7000L": {"description": "Production Data"},
}


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Section:
    title: str
    content: str
    level: int
    page_start: int
    page_end: int
    section_type: str
    tags: list = field(default_factory=list)
    table_data: list = field(default_factory=list)
    is_toc: bool = False
    is_changelog: bool = False


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str
    source_section: str
    page: int
    standard: str
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)


@dataclass
class ExtractedRule:
    rule_text: str
    rule_type: str
    applies_to: str
    source_section: str
    page: int
    standard: str


@dataclass
class ExtractedSchema:
    entity_name: str
    fields: list
    relationships: list
    source_section: str
    standard: str
    source: str = "xsd"             # "xsd" | "inferred" | "prose" (prose = old noisy method)


@dataclass
class DocumentResult:
    standard: str
    pdf_path: str
    extracted_at: str
    total_pages: int
    sections: list
    entities: list
    rules: list
    schemas: list
    tables: list
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# PDF PARSER LAYER
# ─────────────────────────────────────────────

class PDFParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        try:
            self._plumber = pdfplumber.open(pdf_path)
        except Exception:
            self._plumber = None

    def get_page_text(self, page_num: int) -> dict:
        page = self.doc[page_num]
        return {
            "text": page.get_text("text"),
            "blocks": page.get_text("blocks"),
            "page_num": page_num,
        }

    def get_font_info(self, page_num: int) -> list:
        page = self.doc[page_num]
        font_data = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            font_data.append({
                                "text": text,
                                "size": round(span["size"], 1),
                                "bold": bool(span.get("flags", 0) & 16),
                                "font": span.get("font", ""),
                            })
        return font_data

    def get_dominant_font_sizes(self, sample_pages: int = 20) -> dict:
        from collections import Counter
        size_counts = Counter()
        step = max(1, self.total_pages // sample_pages)
        for i in range(0, min(self.total_pages, sample_pages * step), step):
            for span in self.get_font_info(i):
                if span["text"] and len(span["text"]) > 3:
                    size_counts[span["size"]] += 1
        if not size_counts:
            return {"body": 10.0, "h3": 12.0, "h2": 14.0, "h1": 16.0}
        body = size_counts.most_common(1)[0][0]
        larger = sorted([s for s in size_counts if s > body + 1], reverse=True)
        return {
            "body": body,
            "h3": larger[2] if len(larger) > 2 else body + 2,
            "h2": larger[1] if len(larger) > 1 else body + 4,
            "h1": larger[0] if len(larger) > 0 else body + 6,
        }

    def extract_tables_fitz(self, page_num: int) -> list:
        """fitz built-in table finder."""
        page = self.doc[page_num]
        tables = []
        try:
            for tbl in page.find_tables():
                df = tbl.extract()
                if df and len(df) > 1:
                    headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(df[0])]
                    rows = []
                    for row in df[1:]:
                        rows.append({
                            headers[j] if j < len(headers) else f"col_{j}": str(v).strip() if v else ""
                            for j, v in enumerate(row)
                        })
                    tables.append({
                        "page": page_num,
                        "headers": headers,
                        "rows": rows[:100],
                        "row_count": len(df) - 1,
                        "source": "fitz",
                    })
        except Exception:
            pass
        return tables

    # FIX 4: pdfplumber fallback for text-based tables fitz misses
    def extract_tables_plumber(self, page_num: int) -> list:
        if self._plumber is None:
            return []
        tables = []
        try:
            page = self._plumber.pages[page_num]
            # Use explicit table settings — S1000D PDFs have tight spacing
            settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
                "join_tolerance": 3,
            }
            raw = page.extract_tables(settings) or []
            for tbl in raw:
                if not tbl or len(tbl) < 2:
                    continue
                headers = [str(h or "").strip() or f"col_{i}" for i, h in enumerate(tbl[0])]
                rows = [
                    {headers[j] if j < len(headers) else f"col_{j}": str(v or "").strip()
                     for j, v in enumerate(row)}
                    for row in tbl[1:]
                ]
                if any(any(v for v in r.values()) for r in rows):  # skip empty tables
                    tables.append({"page": page_num, "headers": headers,
                                   "rows": rows[:100], "row_count": len(tbl)-1,
                                   "source": "pdfplumber"})
        except Exception:
            pass
        return tables
    
    def close(self):
        self.doc.close()
        if self._plumber:
            self._plumber.close()


# ─────────────────────────────────────────────
# SECTION SEGMENTER
# ─────────────────────────────────────────────

class SectionSegmenter:
    CHAPTER_NUM  = re.compile(r"^Chapter\s+([\d]+(?:\.[\d]+)*)\b", re.I)
    NUMBERED_SEC = re.compile(r"^(\d+\.\d+)(\.\d+){0,4}\s+[A-Z]")
    TOC_DOTS     = re.compile(r"\.{5,}")
    TOC_PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$")
    CPF_LINE     = re.compile(r"CPF\s+\d{4}-\d+", re.I)
    NOTE_WARN    = re.compile(r"^\s*(NOTE|WARNING|CAUTION|IMPORTANT|ATTENTION)\s*[:\-]?", re.I)

    def __init__(self, font_sizes: dict, standard: str = "S1000D"):
        self.font_sizes = font_sizes
        self.profile = STANDARD_PROFILES.get(standard, {})
        self._skip_patterns = [
            re.compile(p) for p in self.profile.get("skip_title_patterns", [])
        ]

    def _is_toc_title(self, title: str) -> bool:
        return bool(self.TOC_DOTS.search(title)) or bool(self.TOC_PAGE_NUM.match(title))

    def _heading_level(self, line: str, font_size: float = None, bold: bool = False) -> Optional[int]:
        line = line.strip()
        if not line or len(line) < 3:
            return None
        m = self.CHAPTER_NUM.match(line)
        if m:
            return min(m.group(1).count(".") + 1, 6)
        m2 = self.NUMBERED_SEC.match(line)
        if m2:
            return min(line.split()[0].count(".") + 1, 6)
        if font_size:
            if font_size >= self.font_sizes.get("h1", 16) - 0.5:
                return 1
            if font_size >= self.font_sizes.get("h2", 14) - 0.5:
                return 2
            if bold and font_size >= self.font_sizes.get("h3", 12) - 0.5:
                return 3
        return None

    def segment(self, pages_data: list) -> list:
        sections = []
        current = Section(
            title="[PREAMBLE]", content="", level=0,
            page_start=0, page_end=0, section_type="body", tags=[],
        )

        for page_data in pages_data:
            page_num = page_data["page_num"]
            for line in page_data["text"].split("\n"):
                line_s = line.strip()
                if not line_s:
                    continue
                if self.TOC_DOTS.search(line_s):
                    continue
                if self.NOTE_WARN.match(line_s):
                    current.content += f"\n[{line_s.split()[0].upper()}] {line_s} "
                    current.tags.append("note_warning")
                    continue

                level = self._heading_level(line_s)
                if level is not None:
                    if current.content.strip():
                        current.page_end = page_num
                        sections.append(current)
                    is_toc = self._is_toc_title(line_s)
                    is_changelog = bool(self.CPF_LINE.search(line_s))
                    current = Section(
                        title=line_s[:200], content="", level=level,
                        page_start=page_num, page_end=page_num,
                        section_type="heading", tags=[],
                        is_toc=is_toc, is_changelog=is_changelog,
                    )
                else:
                    if self.CPF_LINE.search(line_s):
                        current.is_changelog = True
                    current.content += " " + line_s

        if current.content.strip():
            current.page_end = pages_data[-1]["page_num"] if pages_data else 0
            sections.append(current)

        return sections


# ─────────────────────────────────────────────
# BASE EXTRACTOR — v3
# ─────────────────────────────────────────────

class BaseExtractor:
    RULE_PATTERNS = {
        "SHALL":    re.compile(r"[^.!?]*\bshall\b[^.!?]*[.!?]", re.I),
        "MUST":     re.compile(r"[^.!?]*\bmust\b[^.!?]*[.!?]", re.I),
        "SHOULD":   re.compile(r"[^.!?]*\bshould\b[^.!?]*[.!?]", re.I),
        "MAY":      re.compile(r"[^.!?]*\bmay\b[^.!?]*[.!?]", re.I),
        "REQUIRED": re.compile(r"[^.!?]*\bis\s+required\b[^.!?]*[.!?]", re.I),
    }

    FIELD_PATTERN = re.compile(
        r'(?:attribute|element|field|parameter|column|property)\s+["\']?'
        r'(<?)([a-zA-Z][a-zA-Z0-9_]{2,59})(>?)["\']?',
        re.I,
    )
    DEFINITION_PATTERN = re.compile(
        r"^([A-Z][A-Za-z\s\-\/]{3,60})\s*[:\-]\s+(.{15,400})", re.M
    )

    NOISE_WORDS = {
        "of", "in", "is", "and", "or", "the", "a", "an", "to", "for",
        "with", "that", "this", "are", "was", "be", "by", "at", "as",
        "on", "it", "its", "if", "not", "but", "from", "all", "has",
        "have", "which", "when", "where", "will", "can", "may", "set",
        "get", "new", "old", "any", "one", "two", "use", "used", "using",
        "flag", "flags", "within", "values", "affected", "content",
        "true", "false", "null", "none", "type", "id", "ref",
        # FIX 2: extra noise from legal text
        "rights", "right", "license", "user", "agreement", "terms",
        "shall", "must", "may", "provided", "permitted",
    }

    # FIX 2: patterns that identify non-content sections
    _ADDRESS_PATTERN = re.compile(r"^\d{4,5}\s+[A-Z][a-z]+")  # "1050 Brussels"

    def __init__(self, standard: str):
        self.standard = standard
        self.profile = STANDARD_PROFILES.get(standard, {})
        self._skip_lower = [s.lower() for s in self.profile.get("skip_rules_in", [])]

    # ── FIX 2: unified skip gate ──────────────────────────────────────────────
    def _should_skip_section(self, section: Section) -> bool:
        """
        Single gate used by ALL extraction methods.
        Returns True if this section should produce zero output.
        """
        if section.is_changelog or section.is_toc:
            return True

        title_lower = section.title.lower()

        # Named skip list (preamble, copyright, license …)
        if any(skip in title_lower for skip in self._skip_lower):
            return True

        # Address / location artifacts:  "1050 Brussels", "Suite 400"
        if self._ADDRESS_PATTERN.match(section.title.strip()):
            return True

        # Very short titles that are not numbered sections — probably artifacts
        if (len(section.title) < 15
                and not re.search(r"chapter|\d+\.\d+|scope|general|intro", title_lower)):
            return True

        # Preamble block
        if section.title.strip() == "[PREAMBLE]":
            return True

        return False

    # ── helpers ───────────────────────────────────────────────────────────────
    def _is_noise_field(self, name: str) -> bool:
        if len(name) < 3 or len(name) > 60:
            return True
        if name.lower() in self.NOISE_WORDS:
            return True
        if name.islower() and len(name) < 4:
            return True
        return False

    # ── extraction methods ────────────────────────────────────────────────────
    def extract_rules(self, section: Section) -> list:
        if self._should_skip_section(section):        # FIX 2
            return []
        rules, seen = [], set()
        for rule_type, pattern in self.RULE_PATTERNS.items():
            for match in pattern.finditer(section.content):
                rule_text = match.group(0).strip()
                if len(rule_text) < 20:
                    continue
                h = hashlib.md5(rule_text[:80].encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                rules.append(ExtractedRule(
                    rule_text=rule_text[:600],
                    rule_type=rule_type,
                    applies_to=section.title[:100],
                    source_section=section.title[:100],
                    page=section.page_start,
                    standard=self.standard,
                ))
        return rules

    def extract_definitions(self, section: Section) -> list:
        if self._should_skip_section(section):        # FIX 2
            return []
        entities, seen = [], set()
        for match in self.DEFINITION_PATTERN.finditer(section.content):
            name = match.group(1).strip()
            desc = match.group(2).strip()
            if self._is_noise_field(name) or name.lower() in seen:
                continue
            seen.add(name.lower())
            entities.append(ExtractedEntity(
                name=name, entity_type="Definition", description=desc[:400],
                source_section=section.title[:100], page=section.page_start,
                standard=self.standard,
            ))
        return entities

    def extract_fields(self, section: Section) -> list:
        if self._should_skip_section(section):        # FIX 2
            return []
        entities, seen = [], set()
        for match in self.FIELD_PATTERN.finditer(section.content):
            field_name = match.group(2).strip()
            if self._is_noise_field(field_name) or field_name.lower() in seen:
                continue
            seen.add(field_name.lower())
            ctx_start = max(0, match.start() - 80)
            ctx_end = min(len(section.content), match.end() + 200)
            entities.append(ExtractedEntity(
                name=field_name, entity_type="Field",
                description=section.content[ctx_start:ctx_end].strip()[:300],
                source_section=section.title[:100], page=section.page_start,
                standard=self.standard, confidence=0.9,
            ))
        return entities

    def extract_key_entities(self, section: Section) -> list:
        if self._should_skip_section(section):        # FIX 2
            return []
        entities, seen = [], set()
        text_lower = section.content.lower()
        for key_entity in self.profile.get("key_entities", []):
            if key_entity.lower() in text_lower and key_entity.lower() not in seen:
                seen.add(key_entity.lower())
                idx = text_lower.find(key_entity.lower())
                ctx = section.content[max(0, idx - 30):min(len(section.content), idx + 300)].strip()
                entities.append(ExtractedEntity(
                    name=key_entity, entity_type="StandardEntity",
                    description=ctx[:400], source_section=section.title[:100],
                    page=section.page_start, standard=self.standard, confidence=0.85,
                ))
        return entities

    # FIX 3: schema extraction from prose REMOVED — replaced by XSD parser
    # (kept as a stub returning None so subclasses don't break)
    def extract_schema_hints(self, section: Section) -> Optional[ExtractedSchema]:
        return None

    def extract_all(self, section: Section) -> dict:
        return {
            "rules": self.extract_rules(section),
            "definitions": self.extract_definitions(section),
            "fields": self.extract_fields(section),
            "key_entities": self.extract_key_entities(section),
            "schema_hints": None,   # populated by XSD layer, not per-section
        }


# ─────────────────────────────────────────────
# XSD SCHEMA PARSER  (FIX 3 + FIX 5)
# ─────────────────────────────────────────────

XS_NS = "http://www.w3.org/2001/XMLSchema"

def parse_s1000d_xsd_dir(xsd_dir: str) -> tuple[list, list]:
    xsd_dir = Path(xsd_dir)
    xsd_files = list(xsd_dir.glob("**/*.xsd"))
    print(f"  Parsing {len(xsd_files)} XSD files from {xsd_dir} …")

    XS = XS_NS  # "http://www.w3.org/2001/XMLSchema"

    # Pass 1 — collect ALL named complexTypes and groups globally
    # (S1000D defines types in one file and references them from another)
    global_types: dict[str, ET.Element] = {}   # name -> xs:complexType element
    global_groups: dict[str, ET.Element] = {}  # name -> xs:group element
    global_attr_groups: dict[str, list] = {}   # name -> list of attribute names

    all_roots: list[tuple[str, ET.Element]] = []
    for xsd_file in xsd_files:
        try:
            tree = ET.parse(str(xsd_file))
            root = tree.getroot()
            # includes = root.findall(f"{{{XS_NS}}}include")
            # if includes:
            #     print(f"  {xsd_file.name} includes:")
            #     for inc in includes:
            #         print(f"    {inc.get('schemaLocation')}")
            all_roots.append((xsd_file.name, root))
        except ET.ParseError:
            continue

        for ct in root.findall(f".//{{{XS}}}complexType"):
            name = ct.get("name")
            if name:
                global_types[name] = ct

        for grp in root.findall(f".//{{{XS}}}group"):
            name = grp.get("name")
            if name:
                global_groups[name] = grp

        for ag in root.findall(f".//{{{XS}}}attributeGroup"):
            name = ag.get("name")
            if name:
                attrs = [
                    {"name": a.get("name"), "type": a.get("type","string"),
                     "required": a.get("use","optional") == "required"}
                    for a in ag.findall(f".//{{{XS}}}attribute")
                    if a.get("name")
                ]
                global_attr_groups[name] = attrs
    
    # Resolve nested attributeGroup references (groups that ref other groups)
    MAX_PASSES = 4
    for _ in range(MAX_PASSES):
        changed = False
        for root in all_roots:
            for ag in root[1].findall(f"{{{XS}}}attributeGroup"):
                name = ag.get("name")
                if not name:
                    continue
                for ref_el in ag.findall(f".//{{{XS}}}attributeGroup[@ref]"):
                    ref = ref_el.get("ref", "").split(":")[-1]
                    if ref in global_attr_groups:
                        extra = global_attr_groups[ref]
                        existing_names = {a["name"] for a in global_attr_groups.get(name, [])}
                        new_attrs = [a for a in extra if a["name"] not in existing_names]
                        if new_attrs:
                            global_attr_groups.setdefault(name, []).extend(new_attrs)
                            changed = True
        if not changed:
            break

    def resolve_children(node: ET.Element) -> list[str]:
        """Walk sequence/choice/all/group refs to collect child element refs."""
        children = []
        for child in node:
            tag = child.tag.replace(f"{{{XS}}}", "")
            if tag == "element":
                ref = child.get("ref") or child.get("name")
                if ref and ":" not in ref:          # skip namespace-prefixed
                    children.append(ref)
            elif tag in ("sequence", "choice", "all"):
                children.extend(resolve_children(child))
            elif tag == "group":
                ref = child.get("ref")
                if ref and ref in global_groups:
                    children.extend(resolve_children(global_groups[ref]))
            elif tag == "complexContent":
                children.extend(resolve_children(child))
            elif tag in ("extension", "restriction"):
                children.extend(resolve_children(child))
        return children

    def resolve_attributes(node: ET.Element) -> list[dict]:
        """
        Walk ONLY the direct structure of this complexType/extension/restriction
        to collect attribute names. Uses iter() but skips nested element scopes.
        """
        attrs = []
        if node is None:
            return attrs

        # Elements that introduce a new element scope — stop recursing into them
        STOP_TAGS = {f"{{{XS}}}element"}

        def _walk(el):
            for child in el:
                if child.tag in STOP_TAGS:
                    continue   # don't descend into child element definitions
                tag = child.tag.replace(f"{{{XS}}}", "")
                if tag == "attribute" and child.get("name"):
                    attrs.append({
                        "name":     child.get("name"),
                        "type":     child.get("type", "string"),
                        "required": child.get("use", "optional") == "required",
                    })
                elif tag == "attributeGroup":
                    ref = child.get("ref")
                    if ref:
                        # strip namespace prefix if present e.g. "s1000d:dmRefAttrs"
                        ref_local = ref.split(":")[-1]
                        for ag_name, ag_attrs in global_attr_groups.items():
                            if ag_name == ref_local or ag_name.endswith(ref_local):
                                attrs.extend(ag_attrs)
                else:
                    _walk(child)

        _walk(node)
        return attrs

    # Pass 2 — build schemas per xs:element
    schemas, hierarchy = [], []
    seen_elements: set[str] = set()

    for fname, root in all_roots:
        for elem in root.findall(f"{{{XS}}}element"):
            name = elem.get("name")
            if not name or name in seen_elements:
                continue
            seen_elements.add(name)

            # Find the complexType — inline or referenced
            inline_ct = elem.find(f"{{{XS}}}complexType")
            type_ref   = elem.get("type")

            ct = inline_ct
            if ct is None and type_ref and type_ref in global_types:
                ct = global_types[type_ref]

            children = resolve_children(ct) if ct is not None else []
            # remove self-references and duplicates
            children = list(dict.fromkeys(c for c in children if c != name))

            attributes = resolve_attributes(ct) if ct is not None else []
            # deduplicate attributes
            seen_a, attrs_dedup = set(), []
            for a in attributes:
                if a["name"] not in seen_a:
                    seen_a.add(a["name"])
                    attrs_dedup.append(a)

            required = elem.get("minOccurs", "0") not in ("0", None)

            schemas.append({
                "element":    name,
                "children":   children[:30],
                "attributes": attrs_dedup[:20],
                "required":   required,
                "source_file": fname,
                "source":     "xsd",
            })

            for child_name in children:
                hierarchy.append({
                    "parent":      name,
                    "child":       child_name,
                    "source_file": fname,
                    "source":      "xsd",
                })

    print(f"  XSD: {len(schemas)} elements, {len(hierarchy)} parent-child relationships")
    return schemas, hierarchy


def infer_hierarchy_from_changelog(xml_elements: list) -> list:
    """
    FIX 5: When no XSD is available, recover parent-child relationships
    from changelog context strings like:
      "moved child element <X> from the element <Y>"
      "add child elements <A>, <B> to element <C>"
    """
    CHILD_OF = re.compile(
        r"<([a-zA-Z]\w+)>\s+(?:from|to)\s+the\s+element\s+<([a-zA-Z]\w+)>",
        re.I,
    )
    ADD_TO = re.compile(
        r"<([a-zA-Z]\w+)>(?:[,\s]+<([a-zA-Z]\w+)>)*\s+to\s+(?:the\s+)?(?:element\s+)?<([a-zA-Z]\w+)>",
        re.I,
    )
    CHILD_ELEM = re.compile(r"child\s+element\s+<([a-zA-Z]\w+)>", re.I)
    PARENT_ELEM = re.compile(r"(?:from|to)\s+the\s+element\s+<([a-zA-Z]\w+)>", re.I)

    relationships = []
    seen = set()

    for el in xml_elements:
        ctx = el.get("context", "")
        children_found = CHILD_ELEM.findall(ctx)
        parents_found  = PARENT_ELEM.findall(ctx)

        for child in children_found:
            for parent in parents_found:
                key = f"{parent}→{child}"
                if key not in seen:
                    seen.add(key)
                    relationships.append({
                        "parent": parent,
                        "child": child,
                        "source": "changelog_inference",
                        "context": ctx[:150],
                    })

    print(f"  Hierarchy inference (changelog): {len(relationships)} relationships")
    return relationships


# ─────────────────────────────────────────────
# STANDARD-SPECIFIC EXTRACTORS
# ─────────────────────────────────────────────

class S1000DExtractor(BaseExtractor):
    DMC_PATTERN = re.compile(
        r"DMC[-\s]([A-Z0-9]{2,12}-[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9A-Z]{3,10})",
        re.I,
    )
    XML_ELEMENT = re.compile(r"<([a-zA-Z][a-zA-Z0-9]{2,50})[\s>\/]")
    HTML_TAGS   = {
        "div", "span", "table", "tr", "td", "th", "br", "hr",
        "p", "a", "ul", "ol", "li", "img", "b", "i", "em", "strong",
    }
    # FIX 1: pattern to detect CPF context immediately before an XML match
    CPF_CONTEXT = re.compile(r"CPF\s+\d{4}-\d+", re.I)

    def extract_dmc_references(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        refs, seen = [], set()
        for m in self.DMC_PATTERN.finditer(section.content):
            dmc = m.group(1)
            if dmc not in seen:
                seen.add(dmc)
                refs.append({"dmc": dmc, "page": section.page_start, "section": section.title[:80]})
        return refs

    def extract_xml_elements(self, section: Section) -> list:
        """
        FIX 1: Check a window of text BEFORE each match for a CPF marker.
                If the match sits inside a changelog sentence, skip it.
        FIX 2: Also skip sections that are preamble/license/address.
        """
        if self._should_skip_section(section):
            return []

        elements, seen = [], set()
        content = section.content

        for m in self.XML_ELEMENT.finditer(content):
            # FIX 1: look back up to 120 chars before this match
            look_back_start = max(0, m.start() - 120)
            local_ctx       = content[look_back_start: m.start()]
            if self.CPF_CONTEXT.search(local_ctx):
                continue   # ← skip: this XML tag is inside a changelog sentence

            el = m.group(1)
            if el.lower() in seen or self._is_noise_field(el) or el.lower() in self.HTML_TAGS:
                continue
            seen.add(el.lower())

            ctx_start = max(0, m.start() - 20)
            ctx_end   = min(len(content), m.end() + 150)
            elements.append({
                "element": el,
                "context": content[ctx_start:ctx_end].strip()[:200],
                "page":    section.page_start,
                "section": section.title[:80],
            })

        return elements

    def extract_all(self, section: Section) -> dict:
        base = super().extract_all(section)
        base["dmc_references"] = self.extract_dmc_references(section)
        base["xml_elements"]   = self.extract_xml_elements(section)
        return base


class S2000MExtractor(BaseExtractor):
    PART_NUMBER = re.compile(r"\b([A-Z][A-Z0-9]{2,12}-[A-Z0-9]{2,12}(?:-[A-Z0-9]{2,10})?)\b")
    NSN_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{3}-\d{4})\b")

    def extract_part_numbers(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        parts, seen = [], set()
        for m in self.PART_NUMBER.finditer(section.content):
            pn = m.group(1)
            if pn not in seen:
                seen.add(pn)
                parts.append({"part_number": pn, "page": section.page_start, "section": section.title[:80]})
        return parts

    def extract_nsn(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        nsns, seen = [], set()
        for m in self.NSN_PATTERN.finditer(section.content):
            n = m.group(1)
            if n not in seen:
                seen.add(n)
                nsns.append({"nsn": n, "page": section.page_start, "section": section.title[:80]})
        return nsns

    def extract_all(self, section: Section) -> dict:
        base = super().extract_all(section)
        base["part_numbers"]   = self.extract_part_numbers(section)
        base["nsn_references"] = self.extract_nsn(section)
        return base


class S3000LExtractor(BaseExtractor):
    TASK_PATTERN = re.compile(
        r"(?:task|procedure|step)\s*(?:number|no\.?|id|code)?\s*[:\-]?\s*([A-Z][A-Z0-9\-]{2,25})\b",
        re.I,
    )
    FAILURE_MODE = re.compile(r"failure\s+mode\s*[:\-]\s*(.{10,250}?)(?:\.|;|\n)", re.I)
    METRIC       = re.compile(
        r"\b(MTBF|MTTR|MTTF|LCC|MTBUR|MDT|MFHBF|MCBF)\s*[:\=]?\s*([\d\.,]+\s*(?:hours?|hrs?|years?|cycles?|FH)?)\b",
        re.I,
    )

    def extract_tasks(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        tasks, seen = [], set()
        for m in self.TASK_PATTERN.finditer(section.content):
            tid = m.group(1)
            if tid.lower() not in seen and not self._is_noise_field(tid):
                seen.add(tid.lower())
                tasks.append({"task_id": tid, "page": section.page_start, "section": section.title[:80]})
        return tasks

    def extract_failure_modes(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        return [
            {"failure_mode": m.group(1).strip()[:300], "page": section.page_start, "section": section.title[:80]}
            for m in self.FAILURE_MODE.finditer(section.content)
        ]

    def extract_metrics(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        metrics, seen = [], set()
        for m in self.METRIC.finditer(section.content):
            key = f"{m.group(1).upper()}_{m.group(2)}"
            if key not in seen:
                seen.add(key)
                metrics.append({
                    "metric": m.group(1).upper(),
                    "value": m.group(2).strip(),
                    "page": section.page_start,
                    "section": section.title[:80],
                })
        return metrics

    def extract_all(self, section: Section) -> dict:
        base = super().extract_all(section)
        base["tasks"]         = self.extract_tasks(section)
        base["failure_modes"] = self.extract_failure_modes(section)
        base["metrics"]       = self.extract_metrics(section)
        return base


EXTRACTOR_MAP = {
    "S1000D": S1000DExtractor,
    "S2000M": S2000MExtractor,
    "S3000L": S3000LExtractor,
}


# ─────────────────────────────────────────────
# NORMALIZER LAYER
# ─────────────────────────────────────────────

class Normalizer:
    CANONICAL_FIELD_MAP = {
        "pn": "part_number", "partnumber": "part_number", "partno": "part_number",
        "p/n": "part_number", "itemnumber": "item_number", "itemno": "item_number",
        "desc": "nomenclature", "description": "nomenclature", "partname": "nomenclature",
        "qty": "quantity", "qte": "quantity",
        "uom": "unit_of_measure", "unitofissue": "unit_of_measure",
        "rev": "revision",
        "leadtime": "lead_time_days", "leadtimedays": "lead_time_days",
        "stock": "stock_level", "stocklevel": "stock_level",
        "supplier": "supplier_name", "vendor": "supplier_name",
        "nsn": "nato_stock_number",
        "dm": "data_module_code", "dmc": "data_module_code",
        "issuenumber": "issue_number",
        "inwork": "in_work_flag",
        "mtbf": "mean_time_between_failure",
        "mttr": "mean_time_to_repair",
        "mttf": "mean_time_to_failure",
        "lcc": "life_cycle_cost",
        "lora": "level_of_repair_analysis",
    }

    def normalize_field_name(self, raw: str) -> str:
        key = re.sub(r"[\s\-_/]", "", raw.lower())
        return self.CANONICAL_FIELD_MAP.get(key, re.sub(r"[\s\-/]", "_", raw.lower()))

    def deduplicate_rules(self, rules: list) -> list:
        seen, out = set(), []
        for r in rules:
            h = hashlib.md5(r["rule_text"][:100].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                out.append(r)
        return out

    def deduplicate_entities(self, entities: list) -> list:
        seen, out = set(), []
        for e in entities:
            key = f"{e['name'].lower()}_{e['entity_type']}"
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class AerospaceExtractorPipeline:
    def __init__(
        self,
        output_dir: str = "./output",
        max_pages: int = None,
        max_sections: int = None,
        xsd_dir: str = None,            # NEW — path to S1000D XSD folder
    ):
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pages    = max_pages
        self.max_sections = max_sections
        self.xsd_dir      = xsd_dir
        self.normalizer   = Normalizer()
        self.all_results  = {}

        # Parse XSD once, shared across all PDFs processed in this run
        self._xsd_schemas:   list = []
        self._xsd_hierarchy: list = []
        if xsd_dir:
            self._xsd_schemas, self._xsd_hierarchy = parse_s1000d_xsd_dir(xsd_dir)

    def detect_standard(self, pdf_path: str) -> str:
        name = Path(pdf_path).name.upper()
        for std in STANDARD_PROFILES:
            if std in name:
                return std
        try:
            doc = fitz.open(pdf_path)
            first = doc[0].get_text("text").upper()
            doc.close()
            for std in STANDARD_PROFILES:
                if std in first:
                    return std
        except Exception:
            pass
        return "S1000D"

    def process(self, pdf_path: str, standard: str = None) -> DocumentResult:
        if not standard:
            standard = self.detect_standard(pdf_path)

        print(f"\n{'='*60}")
        print(f"  Processing: {Path(pdf_path).name}")
        print(f"  Standard:   {standard} — {STANDARD_PROFILES[standard]['description']}")
        print(f"{'='*60}")

        parser     = PDFParser(pdf_path)
        font_sizes = parser.get_dominant_font_sizes(sample_pages=20)
        print(f"  Pages: {parser.total_pages} | Font sizes: {font_sizes}")

        page_limit = self.max_pages or parser.total_pages
        print(f"  Parsing {page_limit} pages …")
        pages_data = [
            parser.get_page_text(i)
            for i in tqdm(range(min(page_limit, parser.total_pages)), desc="  Pages", ncols=70)
        ]

        segmenter = SectionSegmenter(font_sizes, standard=standard)
        sections  = segmenter.segment(pages_data)

        sections_content = [s for s in sections if not s.is_toc and len(s.content.strip()) > 30]
        sections_toc     = [s for s in sections if s.is_toc]
        sections_log     = [s for s in sections if s.is_changelog]

        print(
            f"  Sections: {len(sections)} total | {len(sections_content)} content | "
            f"{len(sections_toc)} TOC filtered | {len(sections_log)} changelog"
        )

        if self.max_sections:
            sections_content = sections_content[:self.max_sections]

        ExtractorClass = EXTRACTOR_MAP.get(standard, BaseExtractor)
        extractor      = ExtractorClass(standard)

        all_rules, all_entities, all_xml_elements = [], [], []
        sections_out = []

        for sec in tqdm(sections_content, desc="  Extracting", ncols=70):
            result = extractor.extract_all(sec)
            all_rules.extend([asdict(r) for r in result.get("rules", [])])
            all_entities.extend([asdict(e) for e in result.get("definitions", [])])
            all_entities.extend([asdict(e) for e in result.get("fields", [])])
            all_entities.extend([asdict(e) for e in result.get("key_entities", [])])
            if result.get("xml_elements"):
                all_xml_elements.extend(result["xml_elements"])

            sec_dict = {
                "title":           sec.title,
                "level":           sec.level,
                "page_start":      sec.page_start,
                "page_end":        sec.page_end,
                "content_preview": sec.content[:400].strip(),
                "tags":            sec.tags,
                "is_changelog":    sec.is_changelog,
            }
            for k in ("dmc_references", "part_numbers", "nsn_references",
                      "tasks", "failure_modes", "metrics"):
                if k in result and result[k]:
                    sec_dict[k] = result[k]
            sections_out.append(sec_dict)

        # ── Tables: fitz first, pdfplumber fallback (FIX 4) ──────────────────
        print("  Extracting tables …")
        tables = []
        for i in tqdm(range(min(page_limit, parser.total_pages)),
                      desc="  Tables", ncols=70, leave=False):
            fitz_tbls = parser.extract_tables_fitz(i)
            if fitz_tbls:
                for t in fitz_tbls:
                    t["standard"] = standard
                tables.extend(fitz_tbls)
            else:
                # FIX 4: only call pdfplumber when fitz found nothing on this page
                plumber_tbls = parser.extract_tables_plumber(i)
                for t in plumber_tbls:
                    t["standard"] = standard
                tables.extend(plumber_tbls)

        print(f"  Tables found: {len(tables)}")

        # ── Deduplicate ────────────────────────────────────────────────────────
        all_rules    = self.normalizer.deduplicate_rules(all_rules)
        all_entities = self.normalizer.deduplicate_entities(all_entities)

        seen_el, xml_elements_dedup = set(), []
        for el in all_xml_elements:
            if el["element"].lower() not in seen_el:
                seen_el.add(el["element"].lower())
                xml_elements_dedup.append(el)

        # ── Schema & hierarchy ────────────────────────────────────────────────
        # FIX 3: use XSD if available, otherwise infer from changelog
        if standard == "S1000D":
            if self._xsd_schemas:
                all_schemas = self._xsd_schemas
                hierarchy   = self._xsd_hierarchy
                schema_source = "xsd"
            else:
                all_schemas   = []
                hierarchy     = infer_hierarchy_from_changelog(xml_elements_dedup)
                schema_source = "changelog_inference"
        else:
            all_schemas   = []
            hierarchy     = []
            schema_source = "none"

        # ── Assemble result ───────────────────────────────────────────────────
        extra_meta: dict = {}
        if standard == "S1000D":
            extra_meta["xml_elements"]      = xml_elements_dedup
            extra_meta["xml_element_count"] = len(xml_elements_dedup)
            extra_meta["hierarchy"]         = hierarchy
            extra_meta["hierarchy_count"]   = len(hierarchy)
            extra_meta["schema_source"]     = schema_source

        doc_result = DocumentResult(
            standard=standard,
            pdf_path=str(pdf_path),
            extracted_at=datetime.utcnow().isoformat(),
            total_pages=parser.total_pages,
            sections=sections_out,
            entities=all_entities,
            rules=all_rules,
            schemas=all_schemas,
            tables=tables,
            metadata={
                "profile":                 STANDARD_PROFILES[standard],
                "section_count":           len(sections_content),
                "toc_sections_filtered":   len(sections_toc),
                "changelog_sections":      len(sections_log),
                "entity_count":            len(all_entities),
                "rule_count":              len(all_rules),
                "schema_count":            len(all_schemas),
                "table_count":             len(tables),
                "font_sizes":              font_sizes,
                **extra_meta,
            },
        )
        self.all_results[standard] = doc_result
        parser.close()

        print(f"\n  ✅ Done:")
        print(f"     Sections (content): {len(sections_content)}")
        print(f"     Entities:           {len(all_entities)}")
        print(f"     Rules:              {len(all_rules)}")
        print(f"     Tables:             {len(tables)}")
        if standard == "S1000D":
            print(f"     XML elements (clean): {len(xml_elements_dedup)}")
            print(f"     Schemas:              {len(all_schemas)} ({schema_source})")
            print(f"     Hierarchy edges:      {len(hierarchy)}")
        return doc_result

    def save_all(self):
        for standard, result in self.all_results.items():
            out = asdict(result)
            # Promote metadata sub-keys to top-level for convenience
            for k in ("xml_elements", "hierarchy"):
                if k in result.metadata:
                    out[k] = result.metadata[k]
            path = self.output_dir / f"{standard}_extracted.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"  💾 Saved: {path}")
        self._save_merged_ontology()

    def _save_merged_ontology(self):
        ontology = {
            "generated_at":  datetime.utcnow().isoformat(),
            "standards":     list(self.all_results.keys()),
            "entities":      {},
            "rules":         {},
            "schemas":       {},
            "xml_elements":  {},
            "hierarchy":     [],
            "cross_references": [],
        }

        for standard, result in self.all_results.items():
            # Entities
            for e in result.entities:
                key = self.normalizer.normalize_field_name(e["name"])
                if key not in ontology["entities"]:
                    ontology["entities"][key] = {"canonical_name": key, "appearances": []}
                ontology["entities"][key]["appearances"].append({
                    "standard":    standard,
                    "entity_type": e["entity_type"],
                    "description": e["description"][:200],
                    "page":        e["page"],
                })

            # Rules
            for r in result.rules:
                rt = r["rule_type"]
                if rt not in ontology["rules"]:
                    ontology["rules"][rt] = []
                ontology["rules"][rt].append({
                    "standard":   standard,
                    "rule":       r["rule_text"][:300],
                    "applies_to": r["applies_to"],
                })

            # Schemas
            for s in result.schemas:
                key = f"{standard}_{s.get('element', s.get('entity_name', ''))[:50]}"
                ontology["schemas"][key] = s

            # XML elements + hierarchy (S1000D)
            if "xml_elements" in result.metadata:
                for el in result.metadata["xml_elements"]:
                    ename = el["element"]
                    if ename not in ontology["xml_elements"]:
                        ontology["xml_elements"][ename] = {
                            "element":  ename,
                            "standard": standard,
                            "pages":    [],
                            "contexts": [],
                        }
                    ontology["xml_elements"][ename]["pages"].append(el["page"])
                    if len(ontology["xml_elements"][ename]["contexts"]) < 3:
                        ontology["xml_elements"][ename]["contexts"].append(el["context"])

            if "hierarchy" in result.metadata:
                ontology["hierarchy"].extend(result.metadata["hierarchy"])

        # Cross-references: entities shared across multiple standards
        for key, entity_data in ontology["entities"].items():
            standards_seen = list({a["standard"] for a in entity_data["appearances"]})
            if len(standards_seen) > 1:
                ontology["cross_references"].append({
                    "entity":          key,
                    "appears_in":      standards_seen,
                    "connection_type": "shared_entity",
                })

        path = self.output_dir / "ontology_merged.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)

        print(f"\n  🧠 Merged ontology: {path}")
        print(f"     Entities:         {len(ontology['entities'])}")
        print(f"     XML elements:     {len(ontology['xml_elements'])}")
        print(f"     Hierarchy edges:  {len(ontology['hierarchy'])}")
        print(f"     Cross-references: {len(ontology['cross_references'])}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Aerospace Standards PDF Extractor v3 — S1000D/S2000M/S3000L"
    )
    parser.add_argument("pdfs", nargs="+", help="PDF files to process")
    parser.add_argument(
        "--standard", nargs="+", choices=list(STANDARD_PROFILES.keys()),
        help="Override standard detection (one per PDF)",
    )
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--max-pages",    type=int, default=None)
    parser.add_argument("--max-sections", type=int, default=None)
    parser.add_argument(
        "--xsd-dir", default=None,
        help="Path to folder containing S1000D XSD schema files (.xsd). "
             "Enables ground-truth schema extraction. Download from s1000d.org "
             "or your standards body subscription.",
    )
    args = parser.parse_args()

    pipeline  = AerospaceExtractorPipeline(
        output_dir=args.output,
        max_pages=args.max_pages,
        max_sections=args.max_sections,
        xsd_dir=args.xsd_dir,
    )
    standards = args.standard or [None] * len(args.pdfs)
    for pdf_path, std in zip(args.pdfs, standards):
        if not os.path.exists(pdf_path):
            print(f"  ❌ Not found: {pdf_path}")
            continue
        pipeline.process(pdf_path, standard=std)
    pipeline.save_all()
    print("\n✅ Extraction complete.")


if __name__ == "__main__":
    main()