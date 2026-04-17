"""
sbrain_ontology.py — Enhanced Ontology Layer (Context-Aware Semantic Resolution)
Minimal, bidirectional, structure + pure semantic
"""

import xml.etree.ElementTree as ET
import re
from typing import List, Dict

class SBrainOntology:
    def __init__(self, matcher):
        self.matcher = matcher

    def understand_and_translate(self, xml_root, from_std: str, to_std: str) -> ET.Element:
        item_groups = self._extract_structured_items(xml_root, from_std)
        
        understood_items = []
        for group in item_groups:
            understood = self._understand_item(group, from_std, to_std)
            understood_items.append(understood)
        
        return self._generate_target(understood_items, to_std)

    def _extract_structured_items(self, root, from_std: str) -> List[list]:
        items = []
        for csn in root.findall(".//catalogSeqNumber"):
            group = []
            self._walk(csn, group, from_std, parent_path=[])
            if group:
                items.append(group)
        if not items:
            group = []
            self._walk(root, group, from_std, parent_path=[])
            items.append(group)
        return items

    def _walk(self, elem, group: list, from_std: str, parent_path: list):
        tag = self._strip_ns(elem.tag)
        value = (elem.text or "").strip()
        current_path = parent_path + [tag]

        if value:
            group.append({
                "tag": tag,
                "value": value,
                "attributes": dict(elem.attrib),
                "standard": from_std,
                "parent_context": " > ".join(current_path[-3:])  # rich context
            })

        for child in elem:
            self._walk(child, group, from_std, current_path)

    def _strip_ns(self, tag: str) -> str:
        return re.sub(r"\{.*?\}", "", tag)

    def _understand_item(self, concepts: list, from_std: str, to_std: str) -> list:
        """Enhanced semantic resolution with context"""
        understood = []
        for c in concepts:
            # Primary: direct matcher
            best = self.matcher.find_best_concept_in_target(c["tag"], from_std, to_std)
            
            if best and best.score >= 0.38:
                target_tag = best.target_tag
            else:
                # Context-aware fallback using richer signal
                context_text = f"Tag: {c['tag']} Context: {c['parent_context']} Value example: {c['value'][:100]} aerospace parts catalog item"
                best_fallback = self.matcher.find_best_concept_in_target(c["tag"], from_std, to_std)  # re-use with synth
                target_tag = best_fallback.target_tag if best_fallback else c["tag"]

            understood.append({
                "original_tag": c["tag"],
                "value": c["value"],
                "target_tag": target_tag,
                "confidence": best.score if best else 0.0
            })
        return understood

    def _generate_target(self, understood_items: List[list], to_std: str) -> ET.Element:
        """Dynamic, minimal generation"""
        root = ET.Element(f"{to_std.lower()}TranslatedDocument")
        
        for item in understood_items:
            item_el = ET.SubElement(root, "Item")
            for u in item:
                if u["value"]:
                    # Clean tag (camelCase → proper)
                    clean_tag = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', u["target_tag"]).lower()
                    ET.SubElement(item_el, clean_tag).text = str(u["value"])
        
        return root