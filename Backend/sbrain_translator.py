"""
S-Brain Translator v5.0 — Full Ontology Layer (True Semantic Brain)
"""

import xml.etree.ElementTree as ET
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from sbrain_crossmatch import SemanticCrossMatcher
from sbrain_ontology import SBrainOntology   # NEW

class OntologyDrivenTranslator:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.matcher = SemanticCrossMatcher.load(str(self.output_dir / "crossmatch"))
        self.ontology = SBrainOntology(self.matcher)   # Full ontology layer
        self.translation_log = []
        print(f"✅ S-Brain v5.0 Full Ontology Brain ready — {len(self.matcher.links)} links")

    def translate(self, xml_input: str, from_std: str, to_std: str, output_path: Optional[str] = None):
        self.translation_log = []
        print(f"\n{'='*100}\nFull Ontology Translation: {from_std} → {to_std}\n")

        root = self._parse_xml_tolerant(xml_input)

        # Pure ontology pipeline
        translated_root = self.ontology.understand_and_translate(root, from_std, to_std)

        # Metadata wrapper
        wrapper = ET.Element(f"{to_std.lower()}TranslatedDocument")
        wrapper.set("from", from_std)
        wrapper.set("to", to_std)
        wrapper.set("translatedAt", datetime.utcnow().isoformat() + "Z")
        wrapper.append(translated_root)

        ET.indent(wrapper, space="  ")
        xml_str = ET.tostring(wrapper, encoding="unicode", xml_declaration=True)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(xml_str)
            print(f"💾 Saved → {output_path}")

        # Safe coverage
        coverage = sum(1 for elem in translated_root.iter() 
                      if elem.text and elem.text.strip() and elem.tag != "Item")
        print(f"  Ontology Coverage: {coverage} meaningful concepts translated")

        return {"xml_string": xml_str, "log": self.translation_log}

    def _parse_xml_tolerant(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        replacements = {"&nbsp;":" ", "&rsquo;":"'", "&ldquo;":'"', "&rdquo;":'"', "&mdash;":"—", "&ndash;":"-"}
        for old, new in replacements.items():
            raw = raw.replace(old, new)
        raw = re.sub(r'<\?Pub[^>]*\?>', '', raw)
        try:
            return ET.fromstring(raw)
        except:
            wrapped = f"<root>{raw}</root>"
            root = ET.fromstring(wrapped)
            return list(root)[0] if len(root) > 0 else root

    def _calculate_coverage(self, translated_root) -> int:
        """Count how many tags were meaningfully mapped"""
        count = 0
        for elem in translated_root.iter():
            if elem.text and elem.text.strip() and elem.tag != "Item":
                count += 1
        return count

# CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description="S-Brain Translator v5.0 — Full Ontology")
    parser.add_argument("--input", required=True)
    parser.add_argument("--from", dest="from_std", required=True)
    parser.add_argument("--to", dest="to_std", required=True)
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    translator = OntologyDrivenTranslator()
    translator.translate(
        xml_input=args.input,
        from_std=args.from_std.upper(),
        to_std=args.to_std.upper(),
        output_path=args.output
    )

if __name__ == "__main__":
    main()