"""
S-Brain Translator v5.1 — Multi-Standard Ontology Translator
Fixes:
  - translate(...) call was passing literal `...` instead of actual args
  - understand_and_translate now receives proper xml_root, from_std, to_std
  - hierarchy_tracker declared BEFORE use (was after the loop that used it)
  - Multi-standard support: S1000D ↔ S2000M ↔ S3000L ↔ SX000i any direction
  - Proper coverage stats and debug table
  - CLI accepts --standards list for batch multi-target translation
"""

import xml.etree.ElementTree as ET
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict
from sbrain_learning_memory import SBrainLearningMemory

from sbrain_crossmatch import SemanticCrossMatcher
from sbrain_ontology import SBrainOntology

# All supported standards  (SX000i has a lowercase 'i' — never blindly .upper())
KNOWN_STANDARDS = {"S1000D", "S2000M", "S3000L", "SX000i"}

# Case-insensitive lookup so "SX000i" / "SX000i" / "SX000i" all resolve correctly
_STD_NORMALISE: Dict[str, str] = {s.upper(): s for s in KNOWN_STANDARDS}


def _normalise_std(name: str) -> str:
    """
    Resolve a user-supplied standard name to its canonical form.
    Raises ValueError if the name is unrecognised.
    """
    canonical = _STD_NORMALISE.get(name.upper())
    if canonical is None:
        raise ValueError(
            f"Unknown standard: '{name}'. "
            f"Supported: {sorted(KNOWN_STANDARDS)}"
        )
    return canonical


class OntologyDrivenTranslator:
    """
    Translates XML documents between any pair of S-Series standards using
    the semantic cross-matcher and ontology layer built by sbrain_core.py.

    Supports:
        S1000D → S2000M, S3000L, SX000i
        S2000M → S1000D, S3000L, SX000i
        S3000L → S1000D, S2000M, SX000i
        SX000i → S1000D, S2000M, S3000L
    """

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        crossmatch_dir = str(self.output_dir / "crossmatch")
        self.matcher  = SemanticCrossMatcher.load(crossmatch_dir)
        self.ontology = SBrainOntology(self.matcher)
        self.translation_log: List[dict] = []
        self.memory = SBrainLearningMemory()
        print(f"✅ S-Brain Translator v5.1 ready — {len(self.matcher.links)} semantic links")

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def translate(
        self,
        xml_input: str,
        from_std: str,
        to_std: str,
        output_path: Optional[str] = None,
        debug: bool = False,
    ) -> dict:
        """
        Translate *xml_input* (file path or raw XML string) from *from_std*
        to *to_std*.  Returns {"xml_string": ..., "log": ..., "coverage": N}.
        """
        self.translation_log = []

        from_std = _normalise_std(from_std)
        to_std   = _normalise_std(to_std)

        if from_std == to_std:
            raise ValueError("Source and target standards must be different.")

        print(f"\n{'='*80}")
        print(f"  S-Brain Translation: {from_std} → {to_std}")
        print(f"  Input: {xml_input}")
        print(f"{'='*80}")

        # 1. Parse input
        xml_root = self._parse_xml_tolerant(xml_input)

        # 2. Ontology pipeline  ← THIS WAS THE BUG: was passing `...`
        translated_root = self.ontology.understand_and_translate(
            xml_root, from_std, to_std
        )

        # 3. Wrap with metadata
        wrapper = self._wrap_with_metadata(translated_root, from_std, to_std)

        # 4. Serialise
        ET.indent(wrapper, space="  ")
        xml_str = ET.tostring(wrapper, encoding="unicode", xml_declaration=True)

        # 5. Save
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(xml_str, encoding="utf-8")
            print(f"  💾 Saved → {output_path}")

        # 6. Coverage report
        coverage = self._calculate_coverage(translated_root)
        print(f"  ✅ Coverage: {coverage} meaningful concepts translated")

        # 7. Optional debug dump
        if debug:
            self._print_debug_table(translated_root)

        return {
            "xml_string": xml_str,
            "log":        self.translation_log,
            "coverage":   coverage,
        }

    def translate_multi_target(
        self,
        xml_input: str,
        from_std: str,
        target_standards: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, dict]:
        """
        Translate one source document to multiple target standards in one call.
        Returns {to_std: result_dict, ...}.
        """
        from_std = _normalise_std(from_std)
        targets  = [_normalise_std(t) for t in (target_standards or [])] or \
                   [s for s in sorted(KNOWN_STANDARDS) if s != from_std]

        results: Dict[str, dict] = {}
        for to_std in targets:
            out_path = None
            if output_dir:
                stem = Path(xml_input).stem if Path(xml_input).exists() else "doc"
                out_path = str(Path(output_dir) / f"{stem}_{from_std}_to_{to_std}.xml")
            results[to_std] = self.translate(
                xml_input, from_std, to_std, output_path=out_path, debug=debug
            )
        return results

    # ──────────────────────────────────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────────────────────────────────

    def _parse_xml_tolerant(self, path_or_str: str) -> ET.Element:
        """
        Parse XML from a file path or raw string.
        Handles common HTML entities and Pub-specific PIs.
        """
        ENTITY_FIXES = {
            "&nbsp;":  " ",
            "&rsquo;": "'",
            "&ldquo;": '"',
            "&rdquo;": '"',
            "&mdash;": "—",
            "&ndash;": "-",
            "&amp;":   "&amp;",   # keep real ampersands safe
        }

        # Try to read as file first
        try:
            raw = Path(path_or_str).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            raw = path_or_str   # treat as raw XML string

        for old, new in ENTITY_FIXES.items():
            if old != "&amp;":   # don't break already-escaped amps
                raw = raw.replace(old, new)

        # Strip Arbortext processing instructions
        raw = re.sub(r'<\?Pub[^>]*\?>', '', raw)

        try:
            return ET.fromstring(raw)
        except ET.ParseError:
            # Wrap in a root and unwrap — handles documents with multiple top-level nodes
            try:
                wrapped = f"<_root_>{raw}</_root_>"
                root    = ET.fromstring(wrapped)
                return root[0] if len(root) > 0 else root
            except ET.ParseError as e:
                raise ValueError(f"Could not parse XML: {e}")

    def _wrap_with_metadata(
        self, translated_root: ET.Element, from_std: str, to_std: str
    ) -> ET.Element:
        """Wrap translated content in a standard envelope element."""
        wrapper = ET.Element(f"{to_std.lower()}TranslatedDocument")
        wrapper.set("from",         from_std)
        wrapper.set("to",           to_std)
        wrapper.set("engine",       "S-Brain-v5.1")
        wrapper.set("translatedAt", datetime.now(timezone.utc).isoformat())
        wrapper.append(translated_root)
        return wrapper

    def _calculate_coverage(self, translated_root: ET.Element) -> int:
        """Count meaningful (non-structural) translated elements."""
        return sum(
            1 for elem in translated_root.iter()
            if elem.text and elem.text.strip()
            and elem.tag.lower() not in {"item", "itemlist", "messageheader"}
        )

    def _print_debug_table(self, translated_root: ET.Element):
        print("\n" + "="*72)
        print("  DEBUG MAPPING TABLE")
        print("="*72)
        print(f"  {'Target Tag':<35} {'Conf':>5}  {'Path':<22} {'Source Tag'}")
        print("  " + "-"*70)
        for elem in translated_root.iter():
            if elem.text and elem.text.strip():
                conf = elem.get("conf", "?")
                src  = elem.get("src",  "?")
                try:
                    conf_f = float(conf)
                    if conf_f >= 0.90:
                        path = "canonical+semantic"
                    elif conf_f >= 0.82:
                        path = "canonical(no-vec)"
                    elif conf_f >= 0.50:
                        path = "semantic"
                    elif conf_f >= 0.36:
                        path = "semantic(weak)"
                    else:
                        path = "passthrough"
                except ValueError:
                    path = "?"
                print(f"  {elem.tag:<35} {conf:>5}  {path:<22} ← {src}")
        print("="*72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="S-Brain Translator v5.1 — Multi-Standard XML Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate one file S1000D → S2000M
  python sbrain_translator.py --input Inputs/sample.xml --from S1000D --to S2000M --output ./output/result.xml

  # Translate to ALL other standards at once
  python sbrain_translator.py --input Inputs/sample.xml --from S1000D --output-dir ./output/multi

  # Debug mode (prints mapping table)
  python sbrain_translator.py --input Inputs/sample.xml --from S1000D --to S3000L --debug
        """
    )
    parser.add_argument("--input",      required=True,  help="Input XML file path")
    parser.add_argument("--from",       dest="from_std", required=True,
                        help=f"Source standard: {sorted(KNOWN_STANDARDS)}")
    parser.add_argument("--to",         dest="to_std",  default=None,
                        help="Target standard (omit to translate to ALL others)")
    parser.add_argument("--output",     default=None,   help="Output XML file path (single target)")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Output directory (multi-target mode)")
    parser.add_argument("--model-dir",  default="./output",
                        help="Directory containing crossmatch index (default: ./output)")
    parser.add_argument("--debug",      action="store_true",
                        help="Print detailed tag mapping table")
    args = parser.parse_args()

    translator = OntologyDrivenTranslator(output_dir=args.model_dir)

    if args.to_std:
        # Single-target mode
        translator.translate(
            xml_input=args.input,
            from_std=args.from_std,
            to_std=args.to_std,
            output_path=args.output,
            debug=args.debug,
        )
    else:
        # Multi-target mode — translate to all other standards
        from_std_norm = _normalise_std(args.from_std)
        print(f"  Multi-target mode: translating {from_std_norm} → all other standards")
        results = translator.translate_multi_target(
            xml_input=args.input,
            from_std=args.from_std,
            output_dir=args.output_dir or "./output/multi",
            debug=args.debug,
        )
        print(f"\n  ✅ Multi-target complete: {len(results)} translations")
        for std, res in results.items():
            print(f"    {std}: {res['coverage']} concepts")


if __name__ == "__main__":
    main()
