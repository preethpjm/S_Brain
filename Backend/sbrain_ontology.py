"""
sbrain_ontology.py — Agentic Ontology Layer v5.3
Fixes:
  - Extracts ANY repeating element group, not just catalogSeqNumber
  - Smarter tag sanitisation (camelCase-aware, not first-word split)
  - Handles S1000D → S2000M and S2000M → S1000D symmetrically
"""

import xml.etree.ElementTree as ET
import re
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ConceptNode:
    original_tag: str
    value: str
    target_tag: str
    confidence: float
    context: str
    attributes: Dict[str, str] = field(default_factory=dict)

CANONICAL_MAP = {
    "identName": "itemNomenclature",
    "nomenclature": "itemNomenclature",
    "partNumberValue": "partNumber",
    "partNumber": "partNumber",
    "fullNatoStockNumber": "nsn",
    "natoStockNumber": "nsn",
    "quantityPerAssembly": "quantityPerNextHigherAssembly",
    "quantity": "quantityPerNextHigherAssembly",
    "criticalityCode": "criticalityIndicator",
    "sourceOfSupply": "sourceOfSupplyCode",
    "unitPrice": "unitPrice",
    "manufacturerCodeValue": "manufacturerCode",
}

# Wraps a long PascalCase XMI class name into a short usable tag.
# e.g. "HardwarePartDefinitionCommerceData" → "commerceData"
def _sanitise_tag(raw: str) -> str:
    raw_clean = raw.strip()

    # --------------------------------------------------
    # 1. STRICT / CANONICAL MAP (highest priority)
    # --------------------------------------------------
    CLEAN_MAP = {
        "PartInProvisioningProject": "partNumber",
        "HardwarePartDefinitionCommerceData": "unitPrice",
        "HardwarePartDefinitionCustomerFurnishedData": "manufacturerCode",
        "natoItemName": "itemNomenclature",
        "identName": "itemNomenclature",
        "partNumberValue": "partNumber",
        "fullNatoStockNumber": "nsn",
        "quantityPerAssembly": "quantityPerNextHigherAssembly",
        "criticalityCode": "criticalityIndicator",
        "sourceOfSupply": "sourceOfSupplyCode",
    }

    if raw_clean in CLEAN_MAP:
        return CLEAN_MAP[raw_clean]

    raw_lower = raw_clean.lower()

    # --------------------------------------------------
    # 2. KEYWORD-BASED SEMANTIC MAP (secondary)
    # --------------------------------------------------
    KEYWORD_MAP = {
        "nato": "nsn",
        "stock": "nsn",
        "price": "unitPrice",
        "cost": "unitPrice",
        "quantity": "quantityPerNextHigherAssembly",
        "ident": "itemNomenclature",
        "name": "itemNomenclature",
        "manufacturer": "manufacturerCode",
        "supply": "sourceOfSupplyCode",
        "critical": "criticalityIndicator",
    }

    for k, v in KEYWORD_MAP.items():
        if k in raw_lower:
            return v

    # --------------------------------------------------
    # 3. CLEAN SHORT TAGS (safe passthrough)
    # --------------------------------------------------
    if len(raw_clean) <= 25 and re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', raw_clean):
        return raw_clean[0].lower() + raw_clean[1:]

    # --------------------------------------------------
    # 4. CAMELCASE REDUCTION (last resort)
    # --------------------------------------------------
    words = re.sub(r'([A-Z])', r' \1', raw_clean).split()
    if len(words) >= 2:
        tail = words[-2] + words[-1]
        return tail[0].lower() + tail[1:]

    # --------------------------------------------------
    # 5. FINAL FALLBACK
    # --------------------------------------------------
    return re.sub(r'\s+', '', raw_lower)

# S1000D item container tags (in priority order)
_S1000D_ITEM_CONTAINERS = [
    "catalogSeqNumber",
    "itemSeqNumber",
    "sparePartsEntry",
    "supplyItem",
    "partEntry",
]

# S2000M item container tags
_S2000M_ITEM_CONTAINERS = [
    "provisioningItem",
    "sparePartRecord",
    "supplyRecord",
    "itemRecord",
    "part",
]


def _find_item_groups(root, from_std: str) -> List[ET.Element]:
    """
    Find all repeating item-level elements regardless of nesting.
    Tries known container tags first; falls back to the most frequently
    repeated child tag of the document root.
    """
    containers = _S1000D_ITEM_CONTAINERS if from_std == "S1000D" else _S2000M_ITEM_CONTAINERS

    for container in containers:
        found = root.findall(f".//{container}")
        if found:
            return found

    # Generic fallback: find the most common repeated child tag anywhere
    tag_counts: Dict[str, int] = {}
    for elem in root.iter():
        t = re.sub(r'\{.*\}', '', elem.tag)
        tag_counts[t] = tag_counts.get(t, 0) + 1

    if not tag_counts:
        return []

    # Filter out the root tag itself and very short tags (likely wrappers)
    root_tag = re.sub(r'\{.*\}', '', root.tag)
    candidates = {
        t: c for t, c in tag_counts.items()
        if t != root_tag and c > 1 and len(t) > 3
    }
    if not candidates:
        return [root]   # last resort: treat whole doc as one item

    best_tag = max(candidates, key=lambda t: candidates[t])
    return root.findall(f".//{best_tag}")


class SBrainOntology:
    def __init__(self, matcher):
        self.matcher = matcher

    def understand_and_translate(
        self, xml_root, from_std: str, to_std: str
    ) -> ET.Element:
        item_elements = _find_item_groups(xml_root, from_std)

        if not item_elements:
            # Nothing recognisable — wrap the whole root as one item
            item_elements = [xml_root]

        print(f"  Ontology: found {len(item_elements)} item groups "
              f"(tag={re.sub(r'{.*}','',item_elements[0].tag) if item_elements else '?'})")
        
        for lvl in list(hierarchy_tracker.keys()):
            if lvl > current_indenture:
                del hierarchy_tracker[lvl]

        hierarchy_tracker: Dict[int, str] = {}
        understood_items = []

        for elem in item_elements:
            group = self._flatten_element(elem, from_std)
            item_data = self._understand_item(group, from_std, to_std)

            # Track parent–child hierarchy via indenture level
            current_pn = next(
                (c.value for c in item_data
                 if "partnumber" in c.target_tag.lower() or
                    "itemnumber" in c.target_tag.lower()),
                None
            )
            try:
                current_indenture = int(
                    next((c.value for c in item_data
                          if c.original_tag in ("indenture", "indentureLevel")), 1)
                )
            except (ValueError, TypeError):
                current_indenture = 1

            if current_pn:
                hierarchy_tracker[current_indenture] = current_pn

            if current_indenture > 1:
                parent_pn = hierarchy_tracker.get(current_indenture - 1)
                if parent_pn:
                    item_data.append(ConceptNode(
                        original_tag="nha_link",
                        value=parent_pn,
                        target_tag="nextHigherAssembly",
                        confidence=1.0,
                        context="synthetic_hierarchy",
                    ))

            understood_items.append(item_data)

        return self._generate_target(understood_items, to_std)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _flatten_element(self, elem: ET.Element, from_std: str) -> List[dict]:
        """
        Walk an item element and collect all leaf text nodes as a flat list.
        Captures indenture from attributes when present.
        """
        group = []

        # Pull indenture from attributes (S1000D uses indenture="2" on the element)
        indenture = elem.get("indenture") or elem.get("indentureLevel")
        if indenture:
            group.append({
                "tag": "indenture",
                "value": indenture,
                "parent_context": re.sub(r'\{.*\}', '', elem.tag),
                "attr": {},
            })

        self._walk(elem, group, from_std,
                   parent_path=[re.sub(r'\{.*\}', '', elem.tag)])
        return group

    def _walk(self, node: ET.Element, group: list,
              from_std: str, parent_path: List[str]):
        for child in node:
            tag = re.sub(r'\{.*\}', '', child.tag)
            new_path = parent_path + [tag]

            if child.text and child.text.strip():
                group.append({
                    "tag":            tag,
                    "value":          child.text.strip(),
                    "parent_context": " > ".join(parent_path[-2:]),
                    "attr":           dict(child.attrib),
                })

            # Also emit non-text nodes that carry purely attribute data
            elif child.attrib:
                for attr_name, attr_val in child.attrib.items():
                    if attr_val.strip():
                        group.append({
                            "tag":            f"{tag}.{attr_name}",
                            "value":          attr_val.strip(),
                            "parent_context": " > ".join(parent_path[-2:]),
                            "attr":           {},
                        })

            self._walk(child, group, from_std, new_path)

    # ------------------------------------------------------------------
    # Semantic resolution
    # ------------------------------------------------------------------

    def _understand_item(
        self, group: List[dict], from_std: str, to_std: str
    ) -> List[ConceptNode]:
        understood = []

        for c in group:
            best = self.matcher.get_best_match(
                c["tag"], from_std, to_std,
                context=c["parent_context"]
            )

            # STEP 1: Canonical override (VERY IMPORTANT)
            if c["tag"] in CANONICAL_MAP:
                target_tag = CANONICAL_MAP[c["tag"]]
                conf = 0.90
            elif best and best.score >= 0.35:
                target_tag = best.target_tag
                conf = max(0.5, best.score)
            else:
                target_tag = c["tag"]
                conf = 0.30

            understood.append(ConceptNode(
                original_tag=c["tag"],
                value=c["value"],
                target_tag=target_tag,
                confidence=conf,
                context=c["parent_context"],
                attributes=c["attr"],
            ))

        return understood
    # ------------------------------------------------------------------
    # XML generation
    # ------------------------------------------------------------------

    def _generate_target(
        self, understood_items: List[List[ConceptNode]], to_std: str
    ) -> ET.Element:
        root = ET.Element(f"{to_std.lower()}Message")

        meta = ET.SubElement(root, "messageHeader")
        meta.set("engine",   "S-Brain-v5.3")
        meta.set("dateTime", datetime.now().isoformat())
        meta.set("standard", to_std)

        item_list = ET.SubElement(root, "itemList")

        for item_concepts in understood_items:
            item_el = ET.SubElement(item_list, "item")
            seen_tags = set()

            for node in item_concepts:

                if node.original_tag in ("indenture", "indentureLevel"):
                    item_el.set("indentureLevel", str(node.value))
                    continue

                tag = _sanitise_tag(node.target_tag)

                # 🚨 Prevent duplicates
                if tag in seen_tags:
                    continue
                seen_tags.add(tag)

                # 🚨 Prevent semantic corruption
                if tag == "partNumber" and node.original_tag != "partNumberValue":
                    continue

                child = ET.SubElement(item_el, tag)
                child.text = str(node.value)

                child.set("conf", f"{node.confidence:.2f}")
                child.set("src", node.original_tag)

                for k, v in node.attributes.items():
                    if k not in ("conf", "src"):
                        child.set(k, v)

        return root
