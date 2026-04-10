# Aerospace Standards Extraction Pipeline

Extracts structured knowledge from **S1000D**, **S2000M**, and **S3000L** PDFs
(1800–3000+ pages) into a clean, queryable ontology store.

## Architecture

```
PDF (any S-series)
    ↓
PDFParser           ← PyMuPDF (text) + pdfplumber (tables)
    ↓
SectionSegmenter    ← heading detection via font size + numbering
    ↓
Standard Extractor  ← S1000DExtractor / S2000MExtractor / S3000LExtractor
    ↓
Normalizer          ← dedup, canonical field naming
    ↓
Output JSON         ← per-standard + merged ontology
```

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Single PDF (auto-detect standard)
```bash
python extractor.py path/to/s1000d.pdf
```

### Multiple PDFs with explicit standards
```bash
python extractor.py s1000d.pdf s2000m.pdf s3000l.pdf \
    --standard S1000D S2000M S3000L \
    --output ./output
```

### Limit processing (for testing)
```bash
python extractor.py s1000d.pdf --standard S1000D --max-pages 50 --max-sections 30
```

### Validate pipeline (no PDF needed)
```bash
python test_pipeline.py
```

## Outputs (per standard)

```
output/
├── S1000D_extracted.json
├── S2000M_extracted.json
├── S3000L_extracted.json
└── ontology_merged.json      ← cross-referenced brain
```

### Per-standard JSON structure
```json
{
  "standard": "S1000D",
  "total_pages": 1847,
  "sections": [...],
  "entities": [
    {
      "name": "Data Module",
      "entity_type": "StandardEntity",
      "description": "...",
      "page": 42
    }
  ],
  "rules": [
    {
      "rule_text": "Each DM shall have a unique DMC.",
      "rule_type": "SHALL",
      "applies_to": "3.2 DATA MODULE REQUIREMENTS",
      "page": 42
    }
  ],
  "schemas": [...],
  "tables": [...]
}
```

### Merged Ontology
```json
{
  "entities": {
    "part_number": {
      "appearances": [
        {"standard": "S1000D", ...},
        {"standard": "S2000M", ...}
      ]
    }
  },
  "cross_references": [
    {
      "entity": "part_number",
      "appears_in": ["S1000D", "S2000M", "S3000L"],
      "connection_type": "shared_entity"
    }
  ]
}
```

## Extending to S4000P, S5000F, S6000T

1. Add to `STANDARD_PROFILES` dict in `extractor.py`
2. Create `S4000PExtractor(BaseExtractor)` class
3. Register in `EXTRACTOR_MAP`

All base extraction (rules, definitions, fields, schemas) comes for free from `BaseExtractor`.
