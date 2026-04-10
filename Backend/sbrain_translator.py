"""
S-Brain Translator v1.0
Cross-Standard XML Translator: S1000D ↔ S2000M ↔ S3000L

Sits alongside sbrain_core.py in Backend/
Uses the already-extracted ontology_merged.json + XSD knowledge

Usage:
    python sbrain_translator.py --input myfile.xml --from S1000D --to S2000M
    python sbrain_translator.py --input myfile.xml --from S2000M --to S3000L
    python sbrain_translator.py --input myfile.xml --from S1000D --to S3000L
"""

import xml.etree.ElementTree as ET
import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Optional


# ─────────────────────────────────────────────
# MASTER FIELD MAPPING TABLE
# Key = canonical concept name
# Value = per-standard element/attribute names
# ─────────────────────────────────────────────
FIELD_MAP = {
    # ── Identity ──────────────────────────────
    "part_number": {
        "S1000D": ["partNumberValue", "partNumber", "descrPartNumber", "partNo"],
        "S2000M": ["partNumber", "partId", "pnr"],
        "S3000L": ["partNumber", "productItemPartNumber", "partNo"],
    },
    "nato_stock_number": {
        "S1000D": ["natoStockNumber", "fullNatoStockNumber"],
        "S2000M": ["nsn", "natoStockNumber", "nin"],
        "S3000L": ["nsn", "natoStockNumber"],
    },
    "item_number": {
        "S1000D": ["itemNumber", "indenture"],
        "S2000M": ["itemNumber", "lineItemNumber"],
        "S3000L": ["itemNumber", "lineItemNumber"],
    },
    "figure_number": {
        "S1000D": ["figureNumber", "figure"],
        "S2000M": ["figureNumber", "figNum"],
        "S3000L": ["figureNumber"],
    },

    # ── Nomenclature / Description ─────────────
    "description": {
        "S1000D": ["nomenclature", "descr"],
        "S2000M": ["nomenclature", "description"],
        "S3000L": ["description", "nomenclature"],
    },
    # techName and infoName are DISTINCT — must not both collapse to itemName
    "technical_name": {
        "S1000D": ["techName"],
        "S2000M": ["technicalName", "techName"],
        "S3000L": ["technicalName", "techName"],
    },
    "info_name": {
        "S1000D": ["infoName"],
        "S2000M": ["infoName", "dataModuleTitle"],
        "S3000L": ["infoName", "dataModuleTitle"],
    },
    "short_name": {
        "S1000D": ["shortName", "name"],
        "S2000M": ["itemName", "name"],
        "S3000L": ["name"],
    },
    "remarks": {
        "S1000D": ["remarks", "note"],
        "S2000M": ["remarks", "comment"],
        "S3000L": ["remarks", "note", "comment"],
    },

    # ── Quantity / Units ───────────────────────
    "quantity": {
        "S1000D": ["quantityValue", "qty", "quantity"],
        "S2000M": ["quantity", "orderQuantity", "quantityPerAssembly"],
        "S3000L": ["quantity", "quantityPerAssembly"],
    },
    "unit_of_measure": {
        "S1000D": ["quantityUnitOfMeasure", "unitOfMeasure"],
        "S2000M": ["unitOfIssue", "uom", "unitOfMeasure"],
        "S3000L": ["unitOfMeasure", "uom"],
    },

    # ── Supply / Logistics ─────────────────────
    "supplier_name": {
        "S1000D": ["manufacturerName", "vendorName"],
        "S2000M": ["supplierName", "vendorName", "manufacturerName"],
        "S3000L": ["supplierName", "vendorName"],
    },
    "supplier_code": {
        "S1000D": ["manufacturerCodeValue", "cage"],
        "S2000M": ["cageCode", "supplierCode", "manufacturerCode"],
        "S3000L": ["cageCode", "supplierCode"],
    },
    "lead_time": {
        "S1000D": ["leadTime"],
        "S2000M": ["leadTime", "leadTimeDays", "procurementLeadTime"],
        "S3000L": ["leadTime", "repairLeadTime"],
    },
    "unit_price": {
        "S1000D": ["unitPrice", "price"],
        "S2000M": ["unitPrice", "price", "standardPrice"],
        "S3000L": ["unitPrice", "replacementCost"],
    },
    "source_of_supply": {
        "S1000D": ["sourceOfSupply"],
        "S2000M": ["sourceOfSupply", "supplySource"],
        "S3000L": ["sourceOfSupply"],
    },

    # ── Maintenance / LSA ──────────────────────
    "maintenance_task": {
        "S1000D": ["taskCode", "maintenanceTask", "task"],
        "S2000M": ["maintenanceTask", "taskCode"],
        "S3000L": ["maintenanceTask", "taskIdentifier", "taskCode"],
    },
    "failure_mode": {
        "S1000D": ["failureMode", "faultCode"],
        "S2000M": ["failureMode"],
        "S3000L": ["failureMode", "failureModeIdentifier", "fmId"],
    },
    "mtbf": {
        "S1000D": ["mtbf", "meanTimeBetweenFailure"],
        "S2000M": ["mtbf"],
        "S3000L": ["mtbf", "meanTimeBetweenFailure", "reliabilityMTBF"],
    },
    "mttr": {
        "S1000D": ["mttr", "meanTimeToRepair"],
        "S2000M": ["mttr"],
        "S3000L": ["mttr", "meanTimeToRepair"],
    },
    "repair_level": {
        "S1000D": ["repairLevel", "maintenanceLevel"],
        "S2000M": ["repairLevel", "levelOfRepair"],
        "S3000L": ["repairLevel", "levelOfRepairAnalysis", "loraResult"],
    },

    # ── Data Module / Document Reference ──────
    "data_module_code": {
        "S1000D": ["dmCode", "dmc", "dataModuleCode"],
        "S2000M": ["dmReference", "dataModuleCode"],
        "S3000L": ["dmReference", "dataModuleCode"],
    },
    "applicability": {
        "S1000D": ["applicability", "applic", "applicRefId"],
        "S2000M": ["applicability", "applicCode"],
        "S3000L": ["applicability", "applicCode"],
    },
    "effectivity": {
        "S1000D": ["effectivity", "effect"],
        "S2000M": ["effectivity", "applicableFrom"],
        "S3000L": ["effectivity"],
    },

    # ── Classification ─────────────────────────
    "hazardous_material": {
        "S1000D": ["hazardousMaterial", "hazmat"],
        "S2000M": ["hazardousMaterial", "hazmatCode"],
        "S3000L": ["hazardousMaterial"],
    },
    "criticality": {
        "S1000D": ["criticality", "safetyItem"],
        "S2000M": ["criticality", "criticalityCode"],
        "S3000L": ["criticality", "itemCriticality"],
    },
}

# ─────────────────────────────────────────────
# ELEMENT-LEVEL CONTAINER MAPPINGS
# Top-level XML element names per standard
# ─────────────────────────────────────────────
CONTAINER_MAP = {
    # IPL / Parts List
    "ipl_root": {
        "S1000D": "illustratedPartsCatalog",
        "S2000M": "mmDataset",
        "S3000L": "lsaDataset",
    },
    "ipl_item": {
        "S1000D": "catalogSeqNumber",
        "S2000M": "sparePartsData",
        "S3000L": "hardwareElement",
    },
    "parts_list": {
        "S1000D": "partsList",
        "S2000M": "sparePartsList",
        "S3000L": "hardwarePartsList",
    },

    # Maintenance
    "maintenance_task_container": {
        "S1000D": "maintenanceProcedure",
        "S2000M": "maintenanceTask",
        "S3000L": "maintenanceTask",
    },

    # Supplier
    "supplier_container": {
        "S1000D": "manufacturerList",
        "S2000M": "supplierList",
        "S3000L": "supplierList",
    },
}

# ─────────────────────────────────────────────
# BUILD REVERSE LOOKUP: element name → canonical
# ─────────────────────────────────────────────
def build_reverse_lookup() -> dict:
    """
    Returns: { "partNumberValue": ("part_number", "S1000D"), ... }
    """
    reverse = {}
    for canonical, std_map in FIELD_MAP.items():
        for std, names in std_map.items():
            for name in names:
                key = name.lower()
                if key not in reverse:
                    reverse[key] = (canonical, std)
    return reverse

REVERSE_LOOKUP = build_reverse_lookup()


# ─────────────────────────────────────────────
# ONTOLOGY LOADER
# Loads the already-extracted ontology_merged.json
# to enrich translation with real extracted context
# ─────────────────────────────────────────────
class OntologyLoader:
    def __init__(self, ontology_path: str = "./output/ontology_merged.json"):
        self.ontology = {}
        self.entity_index = {}  # name_lower → canonical + standard list

        if os.path.exists(ontology_path):
            with open(ontology_path, "r", encoding="utf-8") as f:
                self.ontology = json.load(f)
            self._build_entity_index()
            print(f"  ✅ Ontology loaded: {len(self.ontology.get('entities', {}))} entities, "
                  f"{len(self.ontology.get('schemas', {}))} schemas")
        else:
            print(f"  ⚠️  Ontology not found at {ontology_path} — running without enrichment")

    def _build_entity_index(self):
        for canonical, data in self.ontology.get("entities", {}).items():
            self.entity_index[canonical.lower()] = {
                "canonical": canonical,
                "standards": [a["standard"] for a in data.get("appearances", [])],
                "descriptions": {
                    a["standard"]: a.get("description", "")
                    for a in data.get("appearances", [])
                }
            }

    def get_cross_standard_description(self, field_name: str, target_std: str) -> str:
        key = re.sub(r"[\s\-_]", "", field_name.lower())
        if key in self.entity_index:
            desc = self.entity_index[key]["descriptions"].get(target_std, "")
            if desc:
                return desc[:200]
        return ""

    def get_schema_children(self, element_name: str, standard: str) -> list:
        key = f"{standard}_{element_name}"
        schema = self.ontology.get("schemas", {}).get(key, {})
        return schema.get("children", [])


# ─────────────────────────────────────────────
# XML TRANSLATOR CORE
# ─────────────────────────────────────────────
class SBrainTranslator:
    def __init__(self, ontology_path: str = "./output/ontology_merged.json"):
        self.ontology = OntologyLoader(ontology_path)
        self.translation_log = []  # audit trail of every mapping decision

    # ── Main entry point ──────────────────────
    def translate(
        self,
        xml_input: str,
        from_std: str,
        to_std: str,
        output_path: str = None,
    ) -> dict:
        """
        Translate an XML file from one S-Series standard to another.

        Args:
            xml_input:   Path to input XML file
            from_std:    Source standard: S1000D | S2000M | S3000L
            to_std:      Target standard: S1000D | S2000M | S3000L
            output_path: Optional path to write translated XML

        Returns:
            dict with keys: xml_string, log, stats
        """
        if from_std == to_std:
            raise ValueError(f"Source and target standards are the same: {from_std}")

        print(f"\n{'='*65}")
        print(f"  Translating: {from_std} → {to_std}")
        print(f"  Input: {xml_input}")
        print(f"{'='*65}")

        self.translation_log = []

        # Parse input XML
        try:
            tree = ET.parse(xml_input)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")

        # Translate
        translated_root = self._translate_element(root, from_std, to_std, depth=0)

        # Add translation metadata as XML comment
        translated_root = self._add_metadata(translated_root, from_std, to_std, xml_input)

        # Serialise
        ET.indent(translated_root, space="  ")
        xml_string = ET.tostring(translated_root, encoding="unicode", xml_declaration=False)
        xml_string = f'<?xml version="1.0" encoding="UTF-8"?>\n' \
                     f'<!-- Translated by S-Brain Translator: {from_std} → {to_std} -->\n' \
                     f'<!-- Generated: {datetime.utcnow().isoformat()}Z -->\n\n' \
                     + xml_string

        # Write output file
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(xml_string)
            print(f"\n  💾 Translated XML saved: {output_path}")

        # Build stats
        mapped     = [l for l in self.translation_log if l["status"] == "mapped"]
        unmapped   = [l for l in self.translation_log if l["status"] == "unmapped"]
        containers = [l for l in self.translation_log if l["status"] == "container"]

        stats = {
            "from_standard":      from_std,
            "to_standard":        to_std,
            "total_elements":     len(self.translation_log),
            "mapped":             len(mapped),
            "unmapped":           len(unmapped),
            "containers_renamed": len(containers),
            "coverage_pct":       round(len(mapped) / max(len(self.translation_log), 1) * 100, 1),
        }

        self._print_summary(stats, unmapped)

        return {
            "xml_string": xml_string,
            "log":        self.translation_log,
            "stats":      stats,
        }

    # ── Recursive element translator ──────────
    def _translate_element(
        self, elem: ET.Element, from_std: str, to_std: str, depth: int
    ) -> ET.Element:

        original_tag = self._strip_namespace(elem.tag)
        new_tag      = self._map_element_name(original_tag, from_std, to_std)

        # Create new element with translated tag
        new_elem = ET.Element(new_tag)

        # Translate attributes — skip XML namespace/schema declarations
        PASSTHROUGH_PREFIXES = ("xmlns", "xsi", "noNamespaceSchemaLocation", "schemaLocation")
        for attr_name, attr_val in elem.attrib.items():
            stripped_attr = self._strip_namespace(attr_name)
            # Namespace declarations and schema locations pass through unchanged
            if any(attr_name.startswith(p) or stripped_attr.startswith(p)
                   for p in PASSTHROUGH_PREFIXES):
                new_elem.set(attr_name, attr_val)
            else:
                new_attr = self._map_element_name(stripped_attr, from_std, to_std)
                new_elem.set(new_attr, attr_val)

        # Preserve text content
        if elem.text and elem.text.strip():
            new_elem.text = elem.text

        if elem.tail and elem.tail.strip():
            new_elem.tail = elem.tail

        # Recurse into children
        for child in elem:
            translated_child = self._translate_element(child, from_std, to_std, depth + 1)
            new_elem.append(translated_child)

        return new_elem

    # ── Field name mapper ─────────────────────
    def _map_element_name(self, name: str, from_std: str, to_std: str) -> str:
        name_lower = name.lower()

        # 1. Check container map first
        for concept, std_map in CONTAINER_MAP.items():
            if std_map.get(from_std, "").lower() == name_lower:
                target = std_map.get(to_std, name)
                self._log(name, target, "container", concept)
                return target

        # 2. Check field map via reverse lookup
        if name_lower in REVERSE_LOOKUP:
            canonical, detected_std = REVERSE_LOOKUP[name_lower]
            target_names = FIELD_MAP.get(canonical, {}).get(to_std, [])
            if target_names:
                target = target_names[0]  # use primary (first) name for target standard
                self._log(name, target, "mapped", canonical)
                return target

        # 3. Try partial match — useful for camelCase variants
        partial = self._partial_match(name_lower, from_std, to_std)
        if partial:
            self._log(name, partial, "mapped", f"partial:{name_lower}")
            return partial

        # 4. No mapping found — keep original name but log it
        self._log(name, name, "unmapped", None)
        return name

    def _partial_match(self, name_lower: str, from_std: str, to_std: str) -> Optional[str]:
        """
        Partial match for camelCase variants.
        Requires minimum 6 chars to avoid false matches like 'name' hitting everything.
        Only matches if the source name appears as a whole word within the field name,
        not just as a substring (prevents techName matching 'name' → itemName collision).
        """
        if len(name_lower) < 6:
            return None  # too short — too many false positives

        for canonical, std_map in FIELD_MAP.items():
            source_names = std_map.get(from_std, [])
            for sname in source_names:
                sname_lower = sname.lower()
                # Must be an exact match or the full source name is contained
                # Avoid partial substring collisions (e.g. 'name' in 'techName')
                if name_lower == sname_lower:
                    target_names = std_map.get(to_std, [])
                    if target_names:
                        return target_names[0]
                # Allow if source name IS the field name with camelCase prefix stripped
                stripped = re.sub(r'^[a-z]+', '', sname_lower)
                if stripped and name_lower == stripped:
                    target_names = std_map.get(to_std, [])
                    if target_names:
                        return target_names[0]
        return None

    # ── Namespace stripper ────────────────────
    def _strip_namespace(self, tag: str) -> str:
        return re.sub(r"\{[^}]+\}", "", tag)

    # ── Metadata injector ─────────────────────
    def _add_metadata(
        self, root: ET.Element, from_std: str, to_std: str, source_file: str
    ) -> ET.Element:
        wrapper = ET.Element(f"{to_std.lower()}TranslatedDocument")
        wrapper.set("translatedFrom", from_std)
        wrapper.set("translatedTo", to_std)
        wrapper.set("sourceFile", Path(source_file).name)
        wrapper.set("translatedAt", datetime.utcnow().isoformat() + "Z")
        wrapper.set("translatorVersion", "SBrain-1.0")
        wrapper.append(root)
        return wrapper

    # ── Audit logger ──────────────────────────
    def _log(self, original: str, translated: str, status: str, canonical: str):
        self.translation_log.append({
            "original":   original,
            "translated": translated,
            "status":     status,
            "canonical":  canonical,
        })

    # ── Summary printer ───────────────────────
    def _print_summary(self, stats: dict, unmapped: list):
        print(f"\n  📊 Translation Summary")
        print(f"     {stats['from_standard']} → {stats['to_standard']}")
        print(f"     Total elements processed : {stats['total_elements']}")
        print(f"     Successfully mapped       : {stats['mapped']}")
        print(f"     Containers renamed        : {stats['containers_renamed']}")
        print(f"     Unmapped (kept original)  : {stats['unmapped']}")
        print(f"     Coverage                  : {stats['coverage_pct']}%")

        if unmapped:
            print(f"\n  ⚠️  Unmapped elements (review these):")
            seen = set()
            for u in unmapped:
                if u["original"] not in seen:
                    seen.add(u["original"])
                    print(f"     - {u['original']}")


# ─────────────────────────────────────────────
# BATCH TRANSLATOR
# Translate a whole folder of XML files
# ─────────────────────────────────────────────
class BatchTranslator:
    def __init__(self, ontology_path: str = "./output/ontology_merged.json"):
        self.translator = SBrainTranslator(ontology_path)

    def translate_folder(
        self,
        input_folder: str,
        from_std: str,
        to_std: str,
        output_folder: str = None,
    ) -> list:
        input_folder  = Path(input_folder)
        output_folder = Path(output_folder) if output_folder else input_folder / f"translated_{to_std}"
        output_folder.mkdir(parents=True, exist_ok=True)

        xml_files = list(input_folder.glob("*.xml"))
        if not xml_files:
            print(f"  No XML files found in {input_folder}")
            return []

        print(f"  Found {len(xml_files)} XML files to translate")
        all_results = []

        for xml_file in xml_files:
            out_name = xml_file.stem + f"_{to_std}.xml"
            out_path = output_folder / out_name
            try:
                result = self.translator.translate(
                    str(xml_file), from_std, to_std, str(out_path)
                )
                all_results.append({"file": xml_file.name, "result": result})
            except Exception as e:
                print(f"  ❌ Failed: {xml_file.name} — {e}")
                all_results.append({"file": xml_file.name, "error": str(e)})

        # Save batch report
        report_path = output_folder / "translation_report.json"
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "from_standard": from_std,
            "to_standard": to_std,
            "files_processed": len(all_results),
            "results": [
                {
                    "file": r["file"],
                    "stats": r.get("result", {}).get("stats", {}),
                    "error": r.get("error"),
                }
                for r in all_results
            ],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n  📋 Batch report saved: {report_path}")

        return all_results


# ─────────────────────────────────────────────
# INTERACTIVE TRANSLATION MODE
# ─────────────────────────────────────────────
def interactive_translate(translator: SBrainTranslator):
    print("\n🔄 S-Brain Translation Mode (type 'exit' to quit)\n")

    STANDARDS = ["S1000D", "S2000M", "S3000L"]

    while True:
        print("\nAvailable standards:", " | ".join(STANDARDS))
        xml_path = input("XML file path (or 'exit'): ").strip()
        if xml_path.lower() == "exit":
            break

        if not os.path.exists(xml_path):
            print(f"  ❌ File not found: {xml_path}")
            continue

        from_std = input("From standard (S1000D / S2000M / S3000L): ").strip().upper()
        to_std   = input("To standard   (S1000D / S2000M / S3000L): ").strip().upper()

        if from_std not in STANDARDS or to_std not in STANDARDS:
            print("  ❌ Invalid standard. Use S1000D, S2000M, or S3000L")
            continue

        out_name = Path(xml_path).stem + f"_{to_std}_translated.xml"
        out_path = str(Path(xml_path).parent / out_name)

        try:
            result = translator.translate(xml_path, from_std, to_std, out_path)
            print(f"\n  ✅ Done. Output: {out_path}")
            print(f"  Coverage: {result['stats']['coverage_pct']}% fields mapped")
        except Exception as e:
            print(f"  ❌ Translation failed: {e}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="S-Brain Cross-Standard XML Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python sbrain_translator.py --input myipl.xml --from S1000D --to S2000M

  # Single file, specify output location
  python sbrain_translator.py --input myipl.xml --from S1000D --to S3000L --output ./out/translated.xml

  # Batch translate a folder
  python sbrain_translator.py --batch ./my_xml_folder --from S2000M --to S3000L

  # Interactive mode
  python sbrain_translator.py --interactive
        """
    )

    parser.add_argument("--input",       help="Path to input XML file")
    parser.add_argument("--from",        dest="from_std", help="Source standard: S1000D | S2000M | S3000L")
    parser.add_argument("--to",          dest="to_std",   help="Target standard: S1000D | S2000M | S3000L")
    parser.add_argument("--output",      help="Output XML file path (optional)")
    parser.add_argument("--batch",       help="Folder of XML files to translate in batch")
    parser.add_argument("--ontology",    default="./output/ontology_merged.json",
                                         help="Path to ontology_merged.json (default: ./output/ontology_merged.json)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--log",         help="Save translation log as JSON to this path")

    args = parser.parse_args()

    # ── Interactive mode ──
    if args.interactive:
        translator = SBrainTranslator(args.ontology)
        interactive_translate(translator)
        return

    # ── Batch mode ──
    if args.batch:
        if not args.from_std or not args.to_std:
            parser.error("--batch requires --from and --to")
        batch = BatchTranslator(args.ontology)
        batch.translate_folder(args.batch, args.from_std.upper(), args.to_std.upper(), args.output)
        return

    # ── Single file mode ──
    if not args.input or not args.from_std or not args.to_std:
        parser.error("Single file mode requires --input, --from, and --to")

    translator = SBrainTranslator(args.ontology)

    result = translator.translate(
        xml_input=args.input,
        from_std=args.from_std.upper(),
        to_std=args.to_std.upper(),
        output_path=args.output,
    )

    # Save log if requested
    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            json.dump(result["log"], f, indent=2)
        print(f"  📋 Translation log saved: {args.log}")

    # Print translated XML to console if no output file specified
    if not args.output:
        print("\n" + "─"*65)
        print(result["xml_string"])


if __name__ == "__main__":
    main()