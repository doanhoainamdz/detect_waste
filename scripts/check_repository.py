"""Check Python syntax, notebook hygiene, credentials, and optional weight hashes."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SECRET = re.compile(r"(?:api_key|password|access_token)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]", re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true", help="Also verify checkpoint size and SHA-256")
    args = parser.parse_args()
    errors = []
    sources = list(ROOT.glob("*.py"))
    for directory in ("module", "scripts", "tests"):
        sources.extend((ROOT / directory).rglob("*.py"))
    for path in sources:
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as error:
            errors.append(f"{path.relative_to(ROOT)}: {error.msg}")
        if SECRET.search(text):
            errors.append(f"{path.relative_to(ROOT)}: possible hardcoded credential")
    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    for path in notebooks:
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(data["cells"]):
            if SECRET.search("".join(cell.get("source", []))):
                errors.append(f"{path.name}, cell {index}: possible hardcoded credential")
            if cell.get("outputs") or cell.get("execution_count") is not None:
                errors.append(f"{path.name}, cell {index}: clear output and execution count")
    if args.weights:
        manifest = json.loads((ROOT / "models_manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["weights"]:
            path = ROOT / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                errors.append(f"{entry['path']}: missing or wrong size; run git lfs pull")
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry["sha256"]:
                errors.append(f"{entry['path']}: SHA-256 mismatch")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(1)
    print(f"OK: {len(sources)} Python files, {len(notebooks)} clean notebooks"
          + (", checkpoint hashes verified" if args.weights else ""))


if __name__ == "__main__":
    main()
