"""
S-Brain Core v4.4
- S1000D, S2000M, S3000L, SX1000i all supported
- Semantic cross-matcher (sbrain_crossmatch.py) built after all PDFs processed
- RAG index built once across all standards
- No hardcoded field maps — everything flows through crossmatch
"""

import fitz
import pdfplumber
import json
import re
import os
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from sbrain_crossmatch import SemanticCrossMatcher

XS_NS = "http://www.w3.org/2001/XMLSchema"

# ─────────────────────────────────────────────
# STANDARD PROFILES
# ─────────────────────────────────────────────

STANDARD_PROFILES = {
    "S1000D": {
        "description": "Technical Publications Standard",
        "key_entities": [
            "Data Module", "IPL", "Illustrated Parts List", "Data Module Code", "DMC",
            "BREX", "CSDB", "Publication Module", "Information Set", "Applicability",
            "Effectivity", "Common Source Database", "Business Rules",
            "Data Module Requirement List", "DMRL", "Schema", "XML", "SGML",
        ],
        "key_sections": [
            "scope", "definitions", "data module", "ipl", "procedure", "description",
            "maintenance", "repair", "illustrated parts", "applicability", "brex",
            "publication", "catalog", "supply", "figure",
        ],
        "schema_entities": ["DataModule", "IPLItem", "Task", "Part", "Tool", "Figure"],
        "skip_title_patterns": [r"^\[PREAMBLE\]", r"\.{5,}", r"^\s*\d+\s*$"],
        "skip_rules_in": [
            "[preamble]", "table of contents", "copyright", "user agreement",
            "license to use", "license", "intellectual property", "no modifications",
        ],
    },
    "S2000M": {
        "description": "Materiel Management Standard (Logistics)",
        "key_entities": [
            "Provisioning", "Spare Part", "Supply", "Procurement", "Vendor", "Supplier",
            "Stock", "Inventory", "Lead Time", "Unit of Issue", "NSN", "Part Number",
            "Source of Supply", "RSPL",
        ],
        "key_sections": [
            "provisioning", "supply", "logistics", "materiel", "procurement",
            "inventory", "stock", "vendor",
        ],
        "schema_entities": ["Part", "Supplier", "InventoryItem", "ProcurementRecord", "SparePartsList"],
        "skip_title_patterns": [r"\.{5,}", r"^\[PREAMBLE\]", r"^\s*\d+\s*$"],
        "skip_rules_in": ["[preamble]", "copyright", "user agreement", "license"],
    },
    "S3000L": {
        "description": "Logistics Support Analysis Standard",
        "key_entities": [
            "LSA", "Logistics Support Analysis", "FMEA", "FMECA", "Failure Mode",
            "Maintenance Task", "LCC", "Life Cycle Cost", "MTBF", "MTTR",
            "Reliability", "Maintainability", "LORA", "LSAR",
        ],
        "key_sections": [
            "lsa", "maintenance", "analysis", "failure mode", "reliability",
            "maintainability", "task",
        ],
        "schema_entities": ["MaintenanceTask", "FailureMode", "SupportEquipment", "LSARecord", "RepairLevel"],
        "skip_title_patterns": [r"\.{5,}", r"^\[PREAMBLE\]", r"^\s*\d+\s*$"],
        "skip_rules_in": ["[preamble]", "copyright", "user agreement", "license"],
    },
    "SX1000i": {
        "description": "Integrated Logistics Support Index Standard",
        "key_entities": [
            "ILS", "Integrated Logistics Support", "Supportability", "LORA",
            "Part Number Index", "Cross-reference", "Provisioning Index",
            "Supply Support", "Technical Manual", "Maintenance Planning",
            "Support Equipment", "Training", "Manpower", "Facilities",
        ],
        "key_sections": [
            "ils", "integrated logistics", "supportability", "cross-reference",
            "index", "supply", "provisioning", "training", "maintenance planning",
        ],
        "schema_entities": [
            "ILSRecord", "SupportabilityAnalysis", "CrossReference",
            "ProvisioningIndex", "SupplySupport",
        ],
        "skip_title_patterns": [r"\.{5,}", r"^\[PREAMBLE\]", r"^\s*\d+\s*$"],
        "skip_rules_in": ["[preamble]", "copyright", "user agreement", "license"],
    },
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
    source: str = "xsd"

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
# PDF PARSER
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

    def extract_tables_fitz(self, page_num: int) -> list:
        page = self.doc[page_num]
        tables = []
        try:
            for tbl in page.find_tables():
                df = tbl.extract()
                if df and len(df) > 1:
                    headers = [
                        str(h).strip() if h else f"col_{i}"
                        for i, h in enumerate(df[0])
                    ]
                    rows = [
                        {
                            headers[j] if j < len(headers) else f"col_{j}": str(v).strip() if v else ""
                            for j, v in enumerate(row)
                        }
                        for row in df[1:]
                    ]
                    tables.append({
                        "page": page_num, "headers": headers,
                        "rows": rows[:100], "row_count": len(df) - 1, "source": "fitz",
                    })
        except Exception:
            pass
        return tables

    def extract_tables_plumber(self, page_num: int) -> list:
        if self._plumber is None:
            return []
        tables = []
        try:
            page = self._plumber.pages[page_num]
            settings = {
                "vertical_strategy": "lines", "horizontal_strategy": "lines",
                "snap_tolerance": 3, "join_tolerance": 3,
            }
            raw = page.extract_tables(settings) or []
            for tbl in raw:
                if not tbl or len(tbl) < 2:
                    continue
                headers = [str(h or "").strip() or f"col_{i}" for i, h in enumerate(tbl[0])]
                rows = [
                    {
                        headers[j] if j < len(headers) else f"col_{j}": str(v or "").strip()
                        for j, v in enumerate(row)
                    }
                    for row in tbl[1:]
                ]
                if any(any(v for v in r.values()) for r in rows):
                    tables.append({
                        "page": page_num, "headers": headers,
                        "rows": rows[:100], "row_count": len(tbl) - 1, "source": "pdfplumber",
                    })
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
            page_start=0, page_end=0, section_type="body",
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
                        section_type="heading", is_toc=is_toc, is_changelog=is_changelog,
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
# BASE EXTRACTOR
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
        r'(<?)([a-zA-Z][a-zA-Z0-9_]{2,59})(>?)["\']?', re.I,
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
        "rights", "right", "license", "user", "agreement", "terms",
        "shall", "must", "provided", "permitted",
    }
    _ADDRESS_PATTERN = re.compile(r"^\d{4,5}\s+[A-Z][a-z]+")

    def __init__(self, standard: str):
        self.standard = standard
        self.profile = STANDARD_PROFILES.get(standard, {})
        self._skip_lower = [s.lower() for s in self.profile.get("skip_rules_in", [])]

    def _should_skip_section(self, section: Section) -> bool:
        if section.is_changelog or section.is_toc:
            return True
        title_lower = section.title.lower()
        if any(skip in title_lower for skip in self._skip_lower):
            return True
        if self._ADDRESS_PATTERN.match(section.title.strip()):
            return True
        if (len(section.title) < 15
                and not re.search(r"chapter|\d+\.\d+|scope|general|intro", title_lower)):
            return True
        if section.title.strip() == "[PREAMBLE]":
            return True
        return False

    def _is_noise_field(self, name: str) -> bool:
        if len(name) < 3 or len(name) > 60:
            return True
        if name.lower() in self.NOISE_WORDS:
            return True
        if name.islower() and len(name) < 4:
            return True
        return False

    def extract_rules(self, section: Section) -> list:
        if self._should_skip_section(section):
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
                    rule_text=rule_text[:600], rule_type=rule_type,
                    applies_to=section.title[:100], source_section=section.title[:100],
                    page=section.page_start, standard=self.standard,
                ))
        return rules

    def extract_definitions(self, section: Section) -> list:
        if self._should_skip_section(section):
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
        if self._should_skip_section(section):
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
        if self._should_skip_section(section):
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

    def extract_schema_hints(self, section: Section) -> Optional[ExtractedSchema]:
        return None

    def extract_all(self, section: Section) -> dict:
        return {
            "rules":       self.extract_rules(section),
            "definitions": self.extract_definitions(section),
            "fields":      self.extract_fields(section),
            "key_entities": self.extract_key_entities(section),
            "schema_hints": None,
        }


# ─────────────────────────────────────────────
# STANDARD-SPECIFIC EXTRACTORS
# ─────────────────────────────────────────────

class S1000DExtractor(BaseExtractor):
    DMC_PATTERN = re.compile(
        r"DMC[-\s]([A-Z0-9]{2,12}-[A-Z0-9]{4}-[A-Z0-9]{3}-[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9A-Z]{3,10})",
        re.I,
    )
    XML_ELEMENT = re.compile(r"<([a-zA-Z][a-zA-Z0-9]{2,50})[\s>\/]")
    HTML_TAGS = {
        "div", "span", "table", "tr", "td", "th", "br", "hr",
        "p", "a", "ul", "ol", "li", "img", "b", "i", "em", "strong",
    }
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
        if self._should_skip_section(section):
            return []
        elements, seen = [], set()
        content = section.content
        for m in self.XML_ELEMENT.finditer(content):
            look_back_start = max(0, m.start() - 120)
            if self.CPF_CONTEXT.search(content[look_back_start: m.start()]):
                continue
            el = m.group(1)
            if el.lower() in seen or self._is_noise_field(el) or el.lower() in self.HTML_TAGS:
                continue
            seen.add(el.lower())
            ctx_start = max(0, m.start() - 20)
            ctx_end = min(len(content), m.end() + 150)
            elements.append({
                "element": el,
                "context": content[ctx_start:ctx_end].strip()[:200],
                "page": section.page_start,
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
    METRIC = re.compile(
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
                    "metric": m.group(1).upper(), "value": m.group(2).strip(),
                    "page": section.page_start, "section": section.title[:80],
                })
        return metrics

    def extract_all(self, section: Section) -> dict:
        base = super().extract_all(section)
        base["tasks"]         = self.extract_tasks(section)
        base["failure_modes"] = self.extract_failure_modes(section)
        base["metrics"]       = self.extract_metrics(section)
        return base


class SX1000iExtractor(BaseExtractor):
    """Extractor for the ILS index standard — focuses on cross-references and index entries."""

    CROSSREF_PATTERN = re.compile(
        r"(?:see|refer to|cross.reference|equivalent)\s+([A-Z][A-Za-z0-9\s\-]{3,80})", re.I
    )
    INDEX_ENTRY = re.compile(
        r"^([A-Z][A-Za-z\s\-\/]{3,60})\s*[,;]\s+([A-Z0-9\-]{3,40})", re.M
    )

    def extract_crossrefs(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        refs, seen = [], set()
        for m in self.CROSSREF_PATTERN.finditer(section.content):
            ref = m.group(1).strip()
            if ref.lower() not in seen:
                seen.add(ref.lower())
                refs.append({"reference": ref, "page": section.page_start, "section": section.title[:80]})
        return refs

    def extract_index_entries(self, section: Section) -> list:
        if self._should_skip_section(section):
            return []
        entries, seen = [], set()
        for m in self.INDEX_ENTRY.finditer(section.content):
            term = m.group(1).strip()
            code = m.group(2).strip()
            key = f"{term}::{code}".lower()
            if key not in seen:
                seen.add(key)
                entries.append({
                    "term": term, "code": code,
                    "page": section.page_start, "section": section.title[:80],
                })
        return entries

    def extract_all(self, section: Section) -> dict:
        base = super().extract_all(section)
        base["crossrefs"]     = self.extract_crossrefs(section)
        base["index_entries"] = self.extract_index_entries(section)
        return base


EXTRACTOR_MAP = {
    "S1000D": S1000DExtractor,
    "S2000M": S2000MExtractor,
    "S3000L": S3000LExtractor,
    "SX1000i": SX1000iExtractor,
}


# ─────────────────────────────────────────────
# XSD PARSER
# ─────────────────────────────────────────────

def _extract_children_from_ct(ct, ns):
    children = []

    def walk(node):
        for child in node:
            tag = child.tag.replace(f"{{{ns}}}", "")
            if tag == "element":
                name = child.get("name") or child.get("ref")
                if name and ":" not in name:
                    children.append(name)
            elif tag in ("sequence", "choice", "all"):
                walk(child)

    if ct is not None:
        walk(ct)
    return list(dict.fromkeys(children))


def parse_xsd_dir(xsd_dir: str, standard: str) -> tuple:
    xsd_dir = Path(xsd_dir)
    xsd_files = list(xsd_dir.glob("**/*.xsd"))
    print(f"  Found {len(xsd_files)} XSD files for {standard}")

    global_types = {}
    for xsd_file in xsd_files:
        try:
            tree = ET.parse(str(xsd_file))
            root = tree.getroot()
            for ct in root.findall(f".//{{{XS_NS}}}complexType"):
                if name := ct.get("name"):
                    global_types[name] = ct
        except Exception as e:
            print(f"    Warning: {xsd_file.name} - {e}")

    schemas, hierarchy, seen = [], [], set()

    for xsd_file in xsd_files:
        try:
            tree = ET.parse(str(xsd_file))
            root = tree.getroot()
            fname = xsd_file.name

            for elem in root.findall(f".//{{{XS_NS}}}element"):
                name = elem.get("name")
                if not name or name in seen:
                    continue
                seen.add(name)

                ct = elem.find(f"./{{{XS_NS}}}complexType")
                if ct is None:
                    type_ref = elem.get("type")
                    if type_ref:
                        ct = global_types.get(type_ref.split(":")[-1])

                children = _extract_children_from_ct(ct, XS_NS)
                schemas.append({
                    "element": name, "standard": standard,
                    "children": children, "source_file": fname, "source": "xsd",
                })
                for child in children:
                    hierarchy.append({"parent": name, "child": child, "standard": standard})
        except Exception as e:
            print(f"    Warning: {xsd_file.name} - {e}")

    return schemas, hierarchy


def parse_xmi(xmi_path: Path, standard: str) -> dict:
    try:
        tree = ET.parse(str(xmi_path))
        root = tree.getroot()
        classes, class_map = [], {}

        for table in root.findall(".//Table"):
            if table.get("name") == "t_object":
                for row in table.findall("Row"):
                    cols = {col.get("name"): col.get("value") for col in row.findall("Column")}
                    if cols.get("Object_Type") == "Class" and (name := cols.get("Name")):
                        class_map[cols.get("Object_ID")] = name
                        classes.append({
                            "class": name, "standard": standard,
                            "note": cols.get("Note", "")[:500],
                            "attributes": [], "associations": [], "source": "xmi",
                        })

        for table in root.findall(".//Table"):
            if table.get("name") == "t_attribute":
                for row in table.findall("Row"):
                    cols = {col.get("name"): col.get("value") for col in row.findall("Column")}
                    obj_id = cols.get("Object_ID")
                    attr_name = cols.get("Name")
                    if obj_id in class_map and attr_name:
                        for c in classes:
                            if c["class"] == class_map[obj_id]:
                                c["attributes"].append({
                                    "name": attr_name,
                                    "type": cols.get("Type", "string"),
                                    "note": cols.get("Note", ""),
                                })
                                break

        total_classes = len(classes)
        total_attrs = sum(len(c["attributes"]) for c in classes)
        print(f"    XMI: {total_classes} classes + {total_attrs} attributes from {xmi_path.name}")
        return {"classes": classes, "source_file": xmi_path.name,
                "total_classes": total_classes, "total_attributes": total_attrs}
    except Exception as e:
        print(f"    Warning: XMI parse failed for {xmi_path.name} - {e}")
        return {"classes": [], "total_classes": 0, "total_attributes": 0}


# ─────────────────────────────────────────────
# MULTI-STANDARD XSD PARSER
# ─────────────────────────────────────────────

class MultiStandardXSDParser:
    def __init__(self):
        self.schemas: Dict[str, list] = {}
        self.hierarchy: Dict[str, list] = {}
        self.xmi_models: Dict[str, dict] = {}

    def load_standard(self, standard: str, xsd_dir: str):
        print(f"  Loading XSDs + XMI for {standard} from {xsd_dir}...")
        schemas, hierarchy = parse_xsd_dir(xsd_dir, standard)

        xmi_file = next(Path(xsd_dir).glob("**/*data_model*.xmi"), None)
        xmi_count = 0
        if xmi_file:
            xmi_data = parse_xmi(xmi_file, standard)
            self.xmi_models[standard] = xmi_data
            xmi_count = xmi_data.get("total_classes", 0)
        else:
            print(f"    No XMI file found in {xsd_dir}")

        print(f"    → {len(schemas)} XSD elements | {len(hierarchy)} hierarchy edges | "
              f"{xmi_count} XMI classes loaded")
        self.schemas[standard] = schemas
        self.hierarchy[standard] = hierarchy


# ─────────────────────────────────────────────
# NORMALIZER
# ─────────────────────────────────────────────

class Normalizer:
    # Kept minimal — semantic matching handles cross-std alignment now
    CANONICAL_FIELD_MAP = {
        "pn": "part_number", "partnumber": "part_number", "partno": "part_number",
        "desc": "nomenclature", "description": "nomenclature",
        "qty": "quantity", "uom": "unit_of_measure", "unitofissue": "unit_of_measure",
        "nsn": "nato_stock_number", "dm": "data_module_code", "dmc": "data_module_code",
        "mtbf": "mean_time_between_failure", "mttr": "mean_time_to_repair",
        "lcc": "life_cycle_cost", "lora": "level_of_repair_analysis",
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
# RAG
# ─────────────────────────────────────────────

class SBrainRAG:
    def __init__(self, output_dir: str = "./output"):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

        base_dir = Path(__file__).resolve().parent.parent
        local_model_path = base_dir / "models" / "all-MiniLM-L6-v2"
        print(f"[RAG] Using LOCAL model at: {local_model_path}")
        #self.model = SentenceTransformer(str(local_model_path))
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.index = None
        self.documents = []
        self.metadata = []
        self.index_path = os.path.join(output_dir, "rag_index.faiss")
        self.docs_path  = os.path.join(output_dir, "rag_docs.npy")
        self.meta_path  = os.path.join(output_dir, "rag_meta.npy")

    def build_index(self, ontology: dict, extracted_results: dict):
        docs = []
        for key, data in ontology.get("entities", {}).items():
            for app in data.get("appearances", []):
                text = (
                    f"Entity: {key}\nStandard: {app['standard']}\n"
                    f"Type: {app.get('entity_type')}\nDescription: {app.get('description','')}"
                )
                docs.append((text, {"type": "entity", "standard": app["standard"], "name": key}))

        for std, schemas in extracted_results.items():
            for sch in schemas.get("schemas", []):
                text = (
                    f"Schema Element: {sch.get('element')}\n"
                    f"Standard: {std}\nChildren: {sch.get('children', [])}"
                )
                docs.append((text, {"type": "schema", "standard": std}))

        for std, data in extracted_results.items():
            for rule in data.get("rules", []):
                text = f"Rule ({std}): {rule.get('rule_text')}"
                docs.append((text, {"type": "rule", "standard": std}))

        if not docs:
            print("  RAG: No documents to index — skipping")
            return

        texts, metas = zip(*docs)
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        self.documents = list(texts)
        self.metadata = list(metas)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        np.save(self.docs_path, np.array(self.documents, dtype=object))
        np.save(self.meta_path, np.array(self.metadata, dtype=object))
        print(f"  RAG built & saved ({len(docs)} docs)")

    def load_index(self) -> bool:
        if all(os.path.exists(p) for p in [self.index_path, self.docs_path, self.meta_path]):
            self.index     = faiss.read_index(self.index_path)
            self.documents = np.load(self.docs_path, allow_pickle=True).tolist()
            self.metadata  = np.load(self.meta_path,  allow_pickle=True).tolist()
            print(f"  RAG loaded ({len(self.documents)} docs)")
            return True
        return False

    def query(self, question: str, top_k: int = 5) -> list:
        if not self.index or not self.documents:
            return []
        q_emb = self.model.encode([question]).astype(np.float32)
        distances, indices = self.index.search(q_emb, min(top_k, len(self.documents)))
        return [
            {"text": self.documents[idx], "meta": self.metadata[idx], "score": float(distances[0][i])}
            for i, idx in enumerate(indices[0]) if idx < len(self.documents)
        ]


# ─────────────────────────────────────────────
# STUB LLM (replace with Ollama or API model)
# ─────────────────────────────────────────────

class StubLLM:
    def __call__(self, prompt: str) -> list:
        if "Context:" in prompt and "Question:" in prompt:
            context  = prompt.split("Context:")[-1].split("Question:")[0].strip()
            answer = (
                f"Based on the standards documentation:\n\n{context[:800]}\n\n"
                f"[POC mode — replace StubLLM with Ollama or an API model for real answers]"
            )
        else:
            answer = prompt[:600]
        return [{"generated_text": f"Answer: {answer}"}]


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class AerospaceExtractorPipeline:

    def __init__(self, output_dir: str = "./output", xsd_dirs: dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = Normalizer()
        self.all_results: Dict[str, DocumentResult] = {}
        self.xsd_parser = MultiStandardXSDParser()
        self.llm = StubLLM()
        self.max_pages = None
        self.max_sections = None
        self.rag = SBrainRAG(output_dir=str(self.output_dir))

        default_xsd = {
            "S1000D": "Inputs/S1000D",
            "S2000M": "Inputs/S2000M",
            "S3000L": "Inputs/S3000L",
            # SX1000i XSD loaded if directory exists
        }
        if xsd_dirs:
            if isinstance(xsd_dirs, str):
                try:
                    xsd_dirs = json.loads(xsd_dirs)
                except Exception:
                    xsd_dirs = {}
            default_xsd.update(xsd_dirs)

        # Auto-detect SX1000i inputs directory
        sx_path = "Inputs/SX1000i"
        if os.path.exists(sx_path) and "SX1000i" not in default_xsd:
            default_xsd["SX1000i"] = sx_path

        for std, path in default_xsd.items():
            if os.path.exists(path):
                self.xsd_parser.load_standard(std, path)
            else:
                print(f"  Warning: XSD path not found for {std}: {path}")

    def detect_standard(self, pdf_path: str) -> str:
        name = Path(pdf_path).name.upper()
        # SX1000i check first (contains "SX1000")
        for std in ["SX1000i", "S3000L", "S2000M", "S1000D"]:
            std_key = std.upper().replace("SX1000I", "SX1000")
            if std_key in name or std.upper() in name:
                return std
        try:
            doc = fitz.open(pdf_path)
            first = doc[0].get_text("text").upper()
            doc.close()
            for std in ["SX1000i", "S3000L", "S2000M", "S1000D"]:
                if std.upper() in first:
                    return std
        except Exception:
            pass
        return "S1000D"

    def process(self, pdf_path: str, standard: str = None) -> DocumentResult:
        if not standard:
            standard = self.detect_standard(pdf_path)

        # If standard not in profiles, default to S1000D processing behaviour
        profile = STANDARD_PROFILES.get(standard, STANDARD_PROFILES["S1000D"])

        print(f"\n{'='*75}")
        print(f"  Processing : {Path(pdf_path).name}")
        print(f"  Standard   : {standard} — {profile['description']}")
        print(f"{'='*75}")

        parser = PDFParser(pdf_path)
        font_sizes = parser.get_dominant_font_sizes(sample_pages=20)
        print(f"  Pages: {parser.total_pages} | Fonts: {font_sizes}")

        page_limit = self.max_pages or parser.total_pages
        pages_data = [
            parser.get_page_text(i)
            for i in tqdm(range(min(page_limit, parser.total_pages)), desc="  Pages", ncols=80)
        ]

        segmenter = SectionSegmenter(font_sizes, standard=standard)
        sections = segmenter.segment(pages_data)

        sections_content = [s for s in sections if not s.is_toc and len(s.content.strip()) > 30]
        sections_toc = [s for s in sections if s.is_toc]
        sections_log = [s for s in sections if s.is_changelog]

        print(f"  Sections: {len(sections)} total | {len(sections_content)} content | "
              f"{len(sections_toc)} TOC | {len(sections_log)} changelog")

        if self.max_sections:
            sections_content = sections_content[:self.max_sections]

        ExtractorClass = EXTRACTOR_MAP.get(standard, BaseExtractor)
        extractor = ExtractorClass(standard)

        all_rules, all_entities, all_xml_elements = [], [], []
        sections_out = []

        for sec in tqdm(sections_content, desc="  Extracting", ncols=80):
            result = extractor.extract_all(sec)
            all_rules.extend([asdict(r) for r in result.get("rules", [])])
            all_entities.extend([
                asdict(e)
                for e in result.get("definitions", [])
                + result.get("fields", [])
                + result.get("key_entities", [])
            ])
            if result.get("xml_elements"):
                all_xml_elements.extend(result["xml_elements"])

            sec_dict = {
                "title": sec.title[:200], "level": sec.level,
                "page_start": sec.page_start, "page_end": sec.page_end,
                "content_preview": sec.content[:600].strip(),
                "tags": sec.tags, "is_changelog": sec.is_changelog,
            }
            for k in (
                "dmc_references", "part_numbers", "nsn_references",
                "tasks", "failure_modes", "metrics", "crossrefs", "index_entries",
            ):
                if k in result and result[k]:
                    sec_dict[k] = result[k]
            sections_out.append(sec_dict)

        print("  Extracting tables...")
        tables = []
        for i in tqdm(range(min(page_limit, parser.total_pages)), desc="  Tables", ncols=80, leave=False):
            for t in parser.extract_tables_fitz(i) + parser.extract_tables_plumber(i):
                t["standard"] = standard
                tables.append(t)

        all_rules    = self.normalizer.deduplicate_rules(all_rules)
        all_entities = self.normalizer.deduplicate_entities(all_entities)

        all_schemas = self.xsd_parser.schemas.get(standard, [])
        hierarchy   = self.xsd_parser.hierarchy.get(standard, [])
        xmi_count   = len(self.xsd_parser.xmi_models.get(standard, {}).get("classes", []))

        print(f"  Schemas: {len(all_schemas)} elements | {len(hierarchy)} edges | {xmi_count} XMI classes")
        print(f"  Entities: {len(all_entities)} | Rules: {len(all_rules)} | Tables: {len(tables)}")

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
                "profile": profile,
                "section_count": len(sections_content),
                "toc_filtered": len(sections_toc),
                "changelog_count": len(sections_log),
                "entity_count": len(all_entities),
                "rule_count": len(all_rules),
                "table_count": len(tables),
                "font_sizes": font_sizes,
                "xml_element_count": len(set(el.get("element", "").lower() for el in all_xml_elements)),
                "hierarchy_count": len(hierarchy),
                "xmi_classes": xmi_count,
                "hierarchy": hierarchy[:500],
            },
        )

        # If the same standard appears more than once (e.g. SX1000i pdf detected as S1000D),
        # merge rather than overwrite.
        if standard in self.all_results:
            existing = self.all_results[standard]
            existing.entities.extend(all_entities)
            existing.rules.extend(all_rules)
            existing.sections.extend(sections_out)
            existing.tables.extend(tables)
        else:
            self.all_results[standard] = doc_result

        print(f"  Done — {standard} complete")
        parser.close()
        return doc_result

    # ── Save + post-processing ──────────────────────────────────────────────

    def _build_temp_ontology(self) -> dict:
        onto = {"entities": {}, "schemas": {}}
        for std, res in self.all_results.items():
            for e in getattr(res, "entities", []):
                key = self.normalizer.normalize_field_name(e.get("name", ""))
                if key:
                    onto["entities"][key] = {"appearances": [e]}
            for s in getattr(res, "schemas", []):
                onto["schemas"][s.get("element", "")] = s
        return onto

    def save_all(self):
        # 1. Save per-standard JSONs
        for standard, result in self.all_results.items():
            out = asdict(result)
            path = self.output_dir / f"{standard}_extracted.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {path}")

        # 2. Merged ontology
        self._save_merged_ontology()

        # 3. RAG index
        print("\n  Building unified RAG index...")
        full_ontology = self._build_temp_ontology()
        all_results_for_rag = {
            std: {"entities": list(res.entities), "schemas": list(res.schemas), "rules": list(res.rules)}
            for std, res in self.all_results.items()
        }
        self.rag.build_index(full_ontology, all_results_for_rag)

        # 4. Semantic cross-matcher
        print("\n  Building Semantic Cross-Matcher...")
        crossmatch_output = str(self.output_dir / "crossmatch")

        # Determine which standards actually have XSD input dirs
        standard_dirs = {}
        candidate_dirs = {
            "S1000D": "Inputs/S1000D",
            "S2000M": "Inputs/S2000M",
            "S3000L": "Inputs/S3000L",
            "SX1000i": "Inputs/SX1000i",
        }
        for std, path in candidate_dirs.items():
            if os.path.exists(path):
                standard_dirs[std] = path

        extracted_jsons = {}
        for std in self.all_results:
            json_path = self.output_dir / f"{std}_extracted.json"
            if json_path.exists():
                extracted_jsons[std] = str(json_path)

        # SX1000i treated as boost source, not a translation endpoint
        sx_dir = standard_dirs.pop("SX1000i", None)

        try:
            matcher = SemanticCrossMatcher(output_dir=crossmatch_output)
            matcher.build(
                standard_dirs=standard_dirs,
                extracted_jsons=extracted_jsons,
                sx1000i_dir=sx_dir,
            )
            print(f"  Cross-matcher saved to: {crossmatch_output}")
        except Exception as e:
            print(f"  Cross-matcher build failed (continuing): {e}")
            import traceback
            traceback.print_exc()

        print(f"\n  RAG ready — {len(self.rag.documents)} docs across {list(self.all_results.keys())}")

    def _save_merged_ontology(self):
        ontology = {
            "generated_at": datetime.utcnow().isoformat(),
            "standards": list(self.all_results.keys()),
            "entities": {}, "rules": {}, "schemas": {},
            "xml_elements": {}, "hierarchy": [], "cross_references": [],
        }

        for standard, result in self.all_results.items():
            for e in result.entities:
                key = self.normalizer.normalize_field_name(e["name"])
                if key not in ontology["entities"]:
                    ontology["entities"][key] = {"canonical_name": key, "appearances": []}
                ontology["entities"][key]["appearances"].append({
                    "standard": standard, "entity_type": e["entity_type"],
                    "description": e["description"][:200], "page": e["page"],
                })

            for r in result.rules:
                rt = r["rule_type"]
                ontology["rules"].setdefault(rt, []).append({
                    "standard": standard, "rule": r["rule_text"][:300],
                    "applies_to": r["applies_to"],
                })

            for s in result.schemas:
                key = f"{standard}_{s.get('element', s.get('entity_name', ''))[:50]}"
                ontology["schemas"][key] = s

            if "hierarchy" in result.metadata:
                ontology["hierarchy"].extend(result.metadata["hierarchy"])

        for key, entity_data in ontology["entities"].items():
            standards_seen = list({a["standard"] for a in entity_data["appearances"]})
            if len(standards_seen) > 1:
                ontology["cross_references"].append({
                    "entity": key, "appears_in": standards_seen,
                    "connection_type": "shared_entity",
                })

        path = self.output_dir / "ontology_merged.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)

        print(f"\n  Merged ontology saved: {path}")
        print(f"    Entities        : {len(ontology['entities'])}")
        print(f"    Hierarchy edges : {len(ontology['hierarchy'])}")
        print(f"    Cross-references: {len(ontology['cross_references'])}")

    # ── Query ───────────────────────────────────────────────────────────────

    def generate_answer(self, question: str, results: list) -> str:
        context = "\n\n".join([r["text"] for r in results[:5]])
        prompt = f"""You are an aerospace standards expert.
RULES: Answer ONLY using the context. If unsure say "Not found in standards".

Context:
{context}

Question:
{question}

Answer:"""
        output = self.llm(prompt)[0]["generated_text"]
        return output.split("Answer:")[-1].strip()


# ─────────────────────────────────────────────
# INTERACTIVE QUERY
# ─────────────────────────────────────────────

def smart_filter(results: list, question: str) -> list:
    q = question.lower()
    if "part number" in q or "part no" in q:
        return [r for r in results if "part" in r["text"].lower()] or results
    if "schema" in q or "element" in q:
        return [r for r in results if r["meta"]["type"] == "schema"] or results
    if "rule" in q or "shall" in q or "must" in q:
        return [r for r in results if r["meta"]["type"] == "rule"] or results
    return results


def interactive_query(pipeline: AerospaceExtractorPipeline):
    print("\nS-Brain Query Mode (type 'exit' to quit)\n")
    while True:
        q = input("Ask: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        results = pipeline.rag.query(q, top_k=8)
        results = smart_filter(results, q)
        if not results:
            print("No relevant information found in standards.")
            continue
        top = sorted(results, key=lambda x: x["score"])[:5]
        answer = pipeline.generate_answer(q, top)
        print(f"\nAnswer:\n{answer}")
        print("\nTop chunks:")
        for i, r in enumerate(top):
            print(f"[{i+1}] ({r['meta']['type']} | {r['meta']['standard']})")
            print(r["text"][:300])
            print("-" * 50)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="S-Brain v4.4")
    parser.add_argument("pdfs", nargs="+", help="PDF files to process")
    parser.add_argument("--standard", nargs="+", default=None,
                        help="Standard per PDF (S1000D, S2000M, S3000L, SX1000i)")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--xsd-dirs", type=str, default=None,
                        help='JSON dict: {"S1000D":"Inputs/S1000D", ...}')
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit pages per PDF (for testing)")
    args = parser.parse_args()

    pipeline = AerospaceExtractorPipeline(output_dir=args.output, xsd_dirs=args.xsd_dirs)
    if args.max_pages:
        pipeline.max_pages = args.max_pages

    standards = args.standard or [None] * len(args.pdfs)
    if len(standards) < len(args.pdfs):
        standards += [None] * (len(args.pdfs) - len(standards))

    for pdf_path, std in zip(args.pdfs, standards):
        if not os.path.exists(pdf_path):
            print(f"  Not found: {pdf_path}")
            continue
        pipeline.process(pdf_path, standard=std)

    pipeline.save_all()
    print("\nS-Brain v4.4 complete.")
    interactive_query(pipeline)


if __name__ == "__main__":
    main()
