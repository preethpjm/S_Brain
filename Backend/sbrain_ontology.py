"""
sbrain_ontology.py — Agentic Ontology Layer v6.1
Changes over v6.0:

  FIX 1 — "item" added to S1000D _ITEM_CONTAINERS
  FIX 2 — Semantic item detection (_is_ipd_item / _find_item_groups fallback)
  FIX 3 — Item validation filter (_is_valid_item)
  FIX 4 — IPL hard-rule layer (IPL_RULES + injection in _understand_item)
  FIX 5 — IPD mode detection (_detect_mode)
"""

import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL CROSS-STANDARD TAG MAP
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_MAP: Dict[str, str] = {
    "identName":                     "itemNomenclature",
    "nomenclature":                  "itemNomenclature",
    "partNumberValue":               "partNumber",
    "fullNatoStockNumber":           "nsn",
    "natoStockNumber":               "nsn",
    "quantityPerAssembly":           "quantityPerNextHigherAssembly",
    "criticalityCode":               "criticalityIndicator",
    "sourceOfSupply":                "sourceOfSupplyCode",
    "unitPrice":                     "unitPrice",
    "manufacturerCodeValue":         "ncageCode",
    "partNumber":                    "partNumber",
    "nsn":                           "nsn",
    "figureNumber":                  "figureReference",
    "itemSeqNumber":                 "itemSequenceNumber",
    "itemNomenclature":              "identName",
    "qpa":                           "quantityPerAssembly",
    "ncageCode":                     "manufacturerCodeValue",
    "unitOfIssue":                   "unitOfMeasure",
    "provisioningSequenceNumber":    "item",
    "lsaTaskCode":                   "maintenanceTaskCode",
    "failureMode":                   "failureModeCode",
    "lruIdentifier":                 "partNumber",
    "lruPartNumber":                 "partNumber",
    "hardwareItemId":                "partNumber",
    "maintenanceLevel":              "maintenanceLevelCode",
    "meanTimeBetweenFailure":        "mtbf",
    "meanTimeToRepair":              "mttr",
    "taskCode":                      "maintenanceTaskCode",
    "failureModeEffect":             "consequenceOfFailure",
    "ipsElement":                    "supportElement",
    "ipsPlan":                       "supportPlan",
    "ipsRequirement":                "supportRequirement",
    "ipsControlElement":             "dataModule",
    "supportTaskRef":                "taskRef",
}

CANONICAL_PRIORITY: Set[str] = {
    "fullNatoStockNumber",
    "partNumberValue",
    "identName",
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — IPL HARD-RULE LAYER
# ─────────────────────────────────────────────────────────────────────────────

IPL_RULES: Dict[str, str] = {
    "item.itemSeqNumber":              "itemSequenceNumber",
    "item.indentureLevel":             "indentureLevel",
    "item.partNumber":                 "partNumber",
    "item.identName":                  "itemNomenclature",
    "item.quantityPerAssembly":        "quantityPerNextHigherAssembly",
    "item.nextHigherAssembly":         "nextHigherAssembly",
    "item.fullNatoStockNumber":        "nsn",
    "catalogSeqNumber.item":           "itemSequenceNumber",
    "catalogSeqNumber.figureNumber":   "figureReference",
    "indenture":                       "indentureLevel",
}

# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTE PROMOTION TABLE
# ─────────────────────────────────────────────────────────────────────────────

ATTR_PROMOTE: Dict[str, Dict[str, tuple]] = {
    "quantityPerAssembly": {
        "quantityUnitOfMeasure": ("unitOfIssue", 0.90),
    },
    "natoStockNumber": {
        "natoSupplyClass":        ("natoSupplyClass",    0.95),
        "natoCodificationBureau": ("codificationBureau", 0.90),
    },
    "catalogSeqNumber": {
        "item":         ("itemSequenceNumber", 1.00),
        "figureNumber": ("figureReference",    1.00),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER TAGS
# ─────────────────────────────────────────────────────────────────────────────

_WRAPPER_TAGS: Set[str] = {
    "natoStockNumber", "nomenclature", "itemIdentData", "partSegment",
    "partsList", "itemList", "content", "illustratedPartsCatalog",
    "identAndStatusSection", "dmAddress", "dmIdent", "dmAddressItems",
    "dmTitle", "dmStatus", "qualityAssurance", "applicability",
}

# ─────────────────────────────────────────────────────────────────────────────
# SKIP TAGS
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_TAGS: Set[str] = {
    "techName", "infoName", "dmCode", "language", "issueInfo",
    "security", "responsiblePartnerCompany", "originator",
    "applicability", "wholeModel", "qualityAssurance", "unverified",
    "dmTitle", "dmAddressItems", "figure", "title", "graphic",
    "languageIsoCode", "countryIsoCode", "issueNumber", "inWork",
    "securityClassification", "enterpriseCode", "issueType", "issueDate",
    "modelIdentCode", "systemDiffCode", "systemCode", "subSystemCode",
    "infoEntityIdent", "icnContents", "icnObject", "id",
}

# ─────────────────────────────────────────────────────────────────────────────
# ITEM CONTAINERS PER STANDARD
# FIX 1: "item" added to S1000D list
# ─────────────────────────────────────────────────────────────────────────────

_ITEM_CONTAINERS: Dict[str, List[str]] = {
    "S1000D": [
        "catalogSeqNumber", "itemSeqNumber", "sparePartsEntry",
        "supplyItem", "partEntry", "functionalItemRef",
        "item",  # FIX 1 — direct <item> children of <figure> in IPDs
    ],
    "S2000M": [
        "provisioningItem", "sparePartRecord", "supplyRecord",
        "itemRecord", "part", "provisioningPartData",
    ],
    "S3000L": [
        "lsaTask", "hardwareItem", "spareItem",
        "maintenanceTask", "failureModeRecord", "lruRecord",
    ],
    "SX000i": [
        "ipsElement", "supportItem", "requirementRecord",
        "ipsTask", "planRecord",
    ],
}

_FIELD_ORDER = [
    "figureReference", "itemSequenceNumber",
    "itemNomenclature", "partNumber", "nsn",
    "natoSupplyClass", "codificationBureau",
    "quantityPerNextHigherAssembly", "unitOfIssue",
    "unitPrice", "ncageCode", "manufacturerCode",
    "criticalityIndicator", "sourceOfSupplyCode",
    "nextHigherAssembly",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConceptNode:
    original_tag:      str
    value:             str
    target_tag:        str
    confidence:        float
    context:           str
    attributes:        Dict[str, str] = field(default_factory=dict)
    is_canonical:      bool = False
    priority:          bool = False
    match_type:        str = "canonical"
    relationship_type: str = "equivalent"


# ─────────────────────────────────────────────────────────────────────────────
# TAG SANITISER
# ─────────────────────────────────────────────────────────────────────────────

_CLEAN_MAP: Dict[str, str] = {
    "PartInProvisioningProject":                   "partNumber",
    "HardwarePartDefinitionCommerceData":          "unitPrice",
    "HardwarePartDefinitionCustomerFurnishedData": "ncageCode",
    "natoItemName":                                "itemNomenclature",
    "identName":                                   "itemNomenclature",
    "partNumberValue":                             "partNumber",
    "fullNatoStockNumber":                         "nsn",
    "quantityPerAssembly":                         "quantityPerNextHigherAssembly",
    "criticalityCode":                             "criticalityIndicator",
    "sourceOfSupply":                              "sourceOfSupplyCode",
    "LsaTaskDefinition":                           "lsaTask",
    "HardwareItemDefinition":                      "hardwareItem",
    "FailureModeDefinition":                       "failureMode",
    "IpsElementDefinition":                        "ipsElement",
    "SupportPlanRecord":                           "supportPlan",
}

_KEYWORD_TOKEN_MAP: Dict[str, str] = {
    "nato":         "nsn",
    "stock":        "nsn",
    "price":        "unitPrice",
    "cost":         "unitPrice",
    "nomenclature": "itemNomenclature",
    "manufacturer": "ncageCode",
    "supply":       "sourceOfSupplyCode",
    "critical":     "criticalityIndicator",
    "failure":      "failureMode",
    "maintenance":  "maintenanceTask",
    "support":      "supportElement",
    "quantity":     "quantityPerNextHigherAssembly",
}


def _camel_tokens(name: str) -> List[str]:
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return [t.lower() for t in s.replace('_', ' ').replace('-', ' ').split() if t]


def _sanitise_tag(raw: str) -> str:
    raw_clean = raw.strip()
    if raw_clean in _CLEAN_MAP:
        return _CLEAN_MAP[raw_clean]
    tokens = _camel_tokens(raw_clean)
    for k, v in _KEYWORD_TOKEN_MAP.items():
        if k in tokens:
            return v
    if len(raw_clean) <= 30 and re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', raw_clean):
        return raw_clean[0].lower() + raw_clean[1:]
    if len(tokens) >= 2:
        tail = tokens[-2].capitalize() + tokens[-1].capitalize()
        return tail[0].lower() + tail[1:]
    return re.sub(r'\s+', '', raw_clean.lower())


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — SEMANTIC ITEM DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_IPD_SEMANTIC_FIELDS: Set[str] = {
    "partNumberValue", "partNumber", "identName", "nomenclature",
}


def _is_ipd_item(elem: ET.Element) -> bool:
    tags = {re.sub(r'\{.*?\}', '', c.tag) for c in elem.iter()}
    return bool(tags & _IPD_SEMANTIC_FIELDS)


# ─────────────────────────────────────────────────────────────────────────────
# ITEM GROUP FINDER
# ─────────────────────────────────────────────────────────────────────────────

def _find_item_groups(root: ET.Element, from_std: str) -> List[ET.Element]:
    # 1. Known container tags (Fix 1 adds "item" to S1000D list)
    containers = _ITEM_CONTAINERS.get(from_std, [])
    for container in containers:
        found = root.findall(f".//{container}")
        if found:
            return found

    # 2. FIX 2 — semantic fallback
    semantic_items = [e for e in root.iter() if _is_ipd_item(e)]
    if semantic_items:
        return semantic_items

    # 3. Frequency heuristic
    tag_counts: Dict[str, int] = {}
    for elem in root.iter():
        t = re.sub(r'\{.*?\}', '', elem.tag)
        tag_counts[t] = tag_counts.get(t, 0) + 1

    root_tag = re.sub(r'\{.*?\}', '', root.tag)
    candidates = {
        t: c for t, c in tag_counts.items()
        if t != root_tag and c > 1 and len(t) > 3
        and t not in _SKIP_TAGS and t not in _WRAPPER_TAGS
    }
    if not candidates:
        return [root]

    best_tag = max(candidates, key=lambda t: candidates[t])
    return root.findall(f".//{best_tag}") or [root]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _f(tag, value, parent_context, attrs=None, priority=False):
    return {
        "tag":            tag,
        "value":          value,
        "parent_context": parent_context,
        "attr":           attrs or {},
        "priority":       priority,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONTOLOGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SBrainOntology:

    def __init__(self, matcher):
        self.matcher = matcher

    # FIX 5 — IPD MODE DETECTION
    def _detect_mode(self, root: ET.Element) -> str:
        if root.find(".//illustratedPartsCatalog") is not None:
            return "IPD"
        if root.find(".//procedure") is not None:
            return "PROCEDURAL"
        return "GENERIC"

    # FIX 3 — ITEM VALIDATION FILTER
    def _is_valid_item(self, concepts: List[ConceptNode]) -> bool:
        has_part = any(c.target_tag == "partNumber" for c in concepts)
        has_name = any(c.target_tag == "itemNomenclature" for c in concepts)
        return has_part or has_name

    def understand_and_translate(
        self,
        xml_root: ET.Element,
        from_std: str,
        to_std: str,
    ) -> ET.Element:
        # FIX 5 — detect mode first
        mode = self._detect_mode(xml_root)
        print(f"  Mode detected: {mode}")

        item_elements = _find_item_groups(xml_root, from_std)
        if not item_elements:
            item_elements = [xml_root]

        first_tag = re.sub(r'\{.*?\}', '', item_elements[0].tag)
        print(f"  Ontology: {len(item_elements)} item group(s) "
              f"(container='{first_tag}') | {from_std} -> {to_std}")

        # Build figure context map
        figure_context: Dict[ET.Element, str] = {}
        if from_std == "S1000D":
            fig_id_by_number: Dict[str, str] = {}
            all_figures = xml_root.findall(".//figure")
            for i, fig in enumerate(all_figures, start=1):
                fid = fig.get("id", "")
                fig_id_by_number[str(i)] = fid

            for parts_list in xml_root.findall(".//partsList"):
                fig_num = parts_list.get("figureNumber", "")
                fig_ref = fig_id_by_number.get(fig_num, fig_num)
                for item_elem in parts_list.findall(".//catalogSeqNumber"):
                    figure_context[item_elem] = fig_ref

            # FIX 5 — map <item> elements to parent figure in IPD mode
            if mode == "IPD":
                for fig in all_figures:
                    fig_id = fig.get("id", "")
                    for item_elem in fig.findall(".//item"):
                        if item_elem not in figure_context:
                            figure_context[item_elem] = fig_id

        hierarchy_tracker: Dict[int, str] = {}
        understood_items: List[List[ConceptNode]] = []

        for elem in item_elements:
            group     = self._flatten_element(elem, from_std)
            item_data = self._understand_item(group, from_std, to_std, mode=mode)

            # FIX 3 — skip invalid items in IPD mode
            if mode == "IPD" and not self._is_valid_item(item_data):
                continue

            # Inject figureReference from context map
            fig_ref = figure_context.get(elem, "")
            if fig_ref and not any(c.target_tag == "figureReference" for c in item_data):
                item_data.insert(0, ConceptNode(
                    original_tag="figure.id",
                    value=fig_ref,
                    target_tag="figureReference",
                    confidence=1.0,
                    context="figure_context",
                    is_canonical=True,
                    priority=True,
                    match_type="rule_override",
                    relationship_type="equivalent",
                ))

            try:
                current_indenture = int(
                    next(
                        (c.value for c in item_data
                         if c.original_tag in ("indenture", "indentureLevel")),
                        1,
                    )
                )
            except (ValueError, TypeError):
                current_indenture = 1

            current_pn = next(
                (c.value for c in item_data
                 if c.target_tag == "partNumber" and c.is_canonical),
                None,
            )
            if current_pn:
                hierarchy_tracker[current_indenture] = current_pn

            for lvl in list(hierarchy_tracker.keys()):
                if lvl > current_indenture:
                    del hierarchy_tracker[lvl]

            if current_indenture > 1:
                parent_pn = hierarchy_tracker.get(current_indenture - 1)
                if parent_pn:
                    item_data.append(ConceptNode(
                        original_tag="nha_link",
                        value=parent_pn,
                        target_tag="nextHigherAssembly",
                        confidence=1.0,
                        context="synthetic_hierarchy",
                        is_canonical=True,
                        match_type="rule_override",
                        relationship_type="equivalent",
                    ))

            understood_items.append(item_data)

        return self._generate_target(understood_items, to_std)

    # ── Extraction ────────────────────────────────────────────────────────────

    def _flatten_element(self, elem: ET.Element, from_std: str) -> List[dict]:
        group: List[dict] = []
        indenture = elem.get("indenture") or elem.get("indentureLevel")
        if indenture:
            group.append(_f("indenture", indenture, re.sub(r'\{.*?\}', '', elem.tag)))
        container_tag = re.sub(r'\{.*?\}', '', elem.tag)
        self._emit_promoted_attrs(container_tag, dict(elem.attrib), [container_tag], group)
        self._walk(elem, group, from_std, [container_tag])
        return group

    def _walk(self, node, group, from_std, parent_path):
        for child in node:
            tag      = re.sub(r'\{.*?\}', '', child.tag)
            new_path = parent_path + [tag]
            if tag in _SKIP_TAGS:
                continue
            ctx = " > ".join(parent_path[-2:])
            if child.text and child.text.strip():
                is_prio = tag in CANONICAL_PRIORITY
                group.append(_f(tag, child.text.strip(), ctx,
                                attrs=dict(child.attrib), priority=is_prio))
                self._emit_promoted_attrs(tag, child.attrib, parent_path, group)
            elif tag in _WRAPPER_TAGS:
                self._emit_promoted_attrs(tag, child.attrib, parent_path, group)
            elif child.attrib:
                promote_keys = set(ATTR_PROMOTE.get(tag, {}).keys())
                for attr_name, attr_val in child.attrib.items():
                    if attr_val.strip() and attr_name not in promote_keys:
                        group.append(_f(f"{tag}.{attr_name}", attr_val.strip(), ctx))
                self._emit_promoted_attrs(tag, child.attrib, parent_path, group)
            self._walk(child, group, from_std, new_path)

    def _emit_promoted_attrs(self, tag, attrib, parent_path, group):
        promote = ATTR_PROMOTE.get(tag, {})
        ctx = " > ".join(parent_path[-2:])
        for attr_name, (target_elem, conf) in promote.items():
            val = attrib.get(attr_name, "").strip()
            if val:
                group.append({
                    "tag":            f"__promoted_{target_elem}",
                    "value":          val,
                    "parent_context": ctx,
                    "attr":           {},
                    "promoted_as":    target_elem,
                    "promoted_conf":  conf,
                    "priority":       False,
                })

    # ── Semantic Resolution ───────────────────────────────────────────────────

    def _understand_item(
        self,
        group: List[dict],
        from_std: str,
        to_std: str,
        mode: str = "GENERIC",
    ) -> List[ConceptNode]:
        understood: List[ConceptNode] = []

        for c in group:
            # Pre-resolved promoted attributes
            if c["tag"].startswith("__promoted_"):
                understood.append(ConceptNode(
                    original_tag=c["tag"],
                    value=c["value"],
                    target_tag=c["promoted_as"],
                    confidence=c["promoted_conf"],
                    context=c["parent_context"],
                    is_canonical=True,
                    priority=False,
                    match_type="attr_promotion",
                    relationship_type="equivalent",
                ))
                continue

            is_prio = c.get("priority", False)

            # FIX 4 — IPL hard rules (fires before semantic/canonical)
            if mode == "IPD":
                last_ctx = c["parent_context"].split(" > ")[-1] if c["parent_context"] else ""
                ipl_target = (
                    IPL_RULES.get(f"{last_ctx}.{c['tag']}") or
                    IPL_RULES.get(c["tag"])
                )
                if ipl_target:
                    understood.append(ConceptNode(
                        original_tag=c["tag"],
                        value=c["value"],
                        target_tag=ipl_target,
                        confidence=1.0,
                        context=c["parent_context"],
                        attributes=c.get("attr", {}),
                        is_canonical=True,
                        priority=True,
                        match_type="rule_override",
                        relationship_type="equivalent",
                    ))
                    continue

            # STEP 1 — semantic matcher
            sem_match = self.matcher.get_best_match(
                c["tag"], from_std, to_std,
                context=c["parent_context"],
            )
            sem_target       = sem_match.target_tag   if sem_match else None
            sem_score        = float(sem_match.score) if sem_match else 0.0
            sem_type         = sem_match.match_type   if sem_match else "none"
            sem_relationship = getattr(sem_match, "relationship_type", "unknown") if sem_match else "unknown"

            # STEP 2 — reference-type passthrough guard
            if sem_match and sem_relationship == "reference" and sem_score < 0.70:
                understood.append(ConceptNode(
                    original_tag=c["tag"],
                    value=c["value"],
                    target_tag=c["tag"],
                    confidence=round(sem_score, 3),
                    context=c["parent_context"],
                    attributes=c.get("attr", {}),
                    is_canonical=False,
                    priority=False,
                    match_type="reference_passthrough",
                    relationship_type="reference",
                ))
                continue

            # STEP 3 — canonical map
            if c["tag"] in CANONICAL_MAP:
                canonical_target = CANONICAL_MAP[c["tag"]]
                if sem_score > 0:
                    agreement_bonus = 0.15 if (
                        sem_target and sem_target.lower() == canonical_target.lower()
                    ) else 0.10
                    conf = round(min(sem_score + agreement_bonus, 0.98), 3)
                else:
                    conf = 0.82
                if sem_type in ("rule_override", "alias_match") and sem_score >= 0.90:
                    conf = sem_score
                understood.append(ConceptNode(
                    original_tag=c["tag"],
                    value=c["value"],
                    target_tag=canonical_target,
                    confidence=conf,
                    context=c["parent_context"],
                    attributes=c.get("attr", {}),
                    is_canonical=True,
                    priority=is_prio or (c["tag"] in CANONICAL_PRIORITY),
                    match_type=sem_type if sem_type != "none" else "canonical_map",
                    relationship_type=sem_relationship if sem_relationship != "unknown" else "transformation",
                ))
                continue

            # STEP 4 — pure semantic match
            if sem_match and sem_score >= 0.50:
                understood.append(ConceptNode(
                    original_tag=c["tag"],
                    value=c["value"],
                    target_tag=sem_target,
                    confidence=round(min(max(sem_score, 0.36), 0.94), 3),
                    context=c["parent_context"],
                    attributes=c.get("attr", {}),
                    is_canonical=False,
                    priority=is_prio,
                    match_type=sem_type,
                    relationship_type=sem_relationship,
                ))
                continue

            # STEP 5 — passthrough
            understood.append(ConceptNode(
                original_tag=c["tag"],
                value=c["value"],
                target_tag=c["tag"],
                confidence=round(max(sem_score, 0.15), 3),
                context=c["parent_context"],
                attributes=c.get("attr", {}),
                is_canonical=False,
                priority=False,
                match_type="passthrough",
                relationship_type="unknown",
            ))

        return understood

    # ── XML Generation ────────────────────────────────────────────────────────

    def _generate_target(self, understood_items, to_std) -> ET.Element:
        root = ET.Element(f"{to_std.lower()}Message")
        meta = ET.SubElement(root, "messageHeader")
        meta.set("engine",   "S-Brain-v6.1")
        meta.set("dateTime", datetime.now().isoformat())
        meta.set("standard", to_std)

        item_list = ET.SubElement(root, "itemList")
        item_list.set("count", str(len(understood_items)))

        for item_concepts in understood_items:
            item_el = ET.SubElement(item_list, "item")
            slot: Dict[str, ConceptNode] = {}

            for node in item_concepts:
                if node.original_tag in ("indenture", "indentureLevel"):
                    item_el.set("indentureLevel", str(node.value))
                    continue

                tag = _sanitise_tag(node.target_tag)

                if tag == "partNumber" and not node.is_canonical:
                    if node.original_tag not in (
                        "partNumberValue", "partNumber", "lruIdentifier",
                        "lruPartNumber", "hardwareItemId", "item"
                    ):
                        continue

                if tag not in slot:
                    slot[tag] = node
                elif node.priority and not slot[tag].priority:
                    slot[tag] = node
                elif node.confidence > slot[tag].confidence and not slot[tag].priority:
                    slot[tag] = node

            emitted: Set[str] = set()
            for preferred in _FIELD_ORDER:
                if preferred in slot:
                    self._emit_node(item_el, preferred, slot[preferred])
                    emitted.add(preferred)
            for tag, node in slot.items():
                if tag not in emitted:
                    self._emit_node(item_el, tag, node)

        return root

    def _emit_node(self, parent, tag, node):
        child = ET.SubElement(parent, tag)
        child.text = str(node.value)
        child.set("conf",         f"{node.confidence:.2f}")
        child.set("src",          node.original_tag)
        child.set("matchType",    node.match_type)
        child.set("relationship", node.relationship_type)

        _DROP_ATTRS = {
            "quantityUnitOfMeasure", "natoSupplyClass", "natoCodificationBureau",
        }
        for k, v in node.attributes.items():
            if k in ("conf", "src", "matchType", "relationship") or k in _DROP_ATTRS:
                continue
            child.set(k, v)

        if node.original_tag == "unitPrice" and "currency" in node.attributes:
            child.set("currency", node.attributes["currency"])