"""
S-Brain Learning Memory v1.0 - Agentic Persistent Layer
No circular imports. Standalone.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import re


def _norm_key(name: str) -> str:
    return re.sub(r'[\s\-_]', '', name.lower())


class SBrainLearningMemory:
    def __init__(self, memory_path: str = "./output/learning_memory.json"):
        self.path = Path(memory_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.memory: Dict[str, dict] = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    def record_correction(self, from_std: str, original_tag: str, validated_target: str,
                          to_std: str, example_value: Optional[str] = None):
        norm_tag = _norm_key(original_tag)
        key = f"{from_std.upper()}::{norm_tag}"

        entry = {
            "from_std": from_std.upper(),
            "original_tag": original_tag,
            "target_tag": validated_target,
            "target_std": to_std.upper(),
            "confidence": 1.0,
            "times_confirmed": self.memory.get(key, {}).get("times_confirmed", 0) + 1,
            "example_value": example_value,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "human_feedback",
        }

        self.memory[key] = entry
        self._save()
        print(f"✅ LEARNED: {from_std}::{original_tag} → {to_std}::{validated_target} "
              f"({entry['times_confirmed']} confirmations)")

    def get_validated_mapping(self, from_std: str, tag: str, to_std: str) -> Optional[dict]:
        """Returns dict if human-validated mapping exists"""
        norm_tag = _norm_key(tag)
        key = f"{from_std.upper()}::{norm_tag}"
        return self.memory.get(key)

    def get_stats(self) -> dict:
        return {
            "total_learned_mappings": len(self.memory),
            "last_updated": max((v.get("timestamp") for v in self.memory.values()), default="never")
        }
