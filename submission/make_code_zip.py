"""Build group_XX_code/ and group_XX_code.zip for the Project 1 submission.

Rules from the brief: source code only, no datasets, no models, no external links.

    python submission/make_code_zip.py --group 07

Files are taken from `git ls-files`, so untracked data, .venv and caches never get in.
Commit (or at least `git add`) changes before building.
"""

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "submission"

MEMBER_ROOTS = ("hoanganh/", "lam/", "SC4020-Ky/", "weiyew/")

EXCLUDE = [
    "hoanganh/untitled10.py",        # Colab export, superseded by the notebooks
    "SC4020-Ky/_archive/*",          # per-point labels (model output)
    "SC4020-Ky/docs/*",              # individual draft report, not code
    "SC4020-Ky/report/*",
    "*.bak",
    "*.pdf",
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.joblib",
    "*.zip",
]

# Ky's data/raw/ holds experiment outputs, not datasets; ship them as results/.
RENAME = {"SC4020-Ky/data/raw/": "SC4020-Ky/results/"}

MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)[^)]+\)")
URL = re.compile(r"https?://[^\s\"'<>)\]\\]+")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def is_dataset(path):
    parts = path.split("/")
    return "data" in parts[:-1] and parts[-1] != "README.md"


def keep(path):
    if not path.startswith(MEMBER_ROOTS):
        return False
    if any(fnmatch.fnmatch(path, pat) for pat in EXCLUDE):
        return False
    if any(path.startswith(src) for src in RENAME):
        return True
    return not is_dataset(path)


def target(path):
    for src, dst in RENAME.items():
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def strip_md_links(text):
    return MD_LINK.sub(r"\1", text)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".md":
        dst.write_text(strip_md_links(src.read_text(encoding="utf-8")), encoding="utf-8")
    elif src.suffix == ".ipynb":
        nb = json.loads(src.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                cell["source"] = [strip_md_links(line) for line in cell["source"]]
        dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def url_audit(root):
    hits = []
    for f in sorted(root.rglob("*")):
        if f.suffix not in {".py", ".ipynb", ".md", ".txt", ".toml"}:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if f.suffix == ".ipynb":
            # outputs can hold base64 images; only look at cell sources
            nb = json.loads(text)
            text = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        for url in sorted(set(URL.findall(text))):
            hits.append((f.relative_to(root).as_posix(), url))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="XX", help="group ID, e.g. 07")
    args = ap.parse_args()

    name = f"group_{args.group}_code"
    out = HERE / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    files = [p for p in tracked_files() if keep(p)]
    for p in files:
        copy(REPO / p, out / target(p))
    copy(HERE / "README_code.md", out / "README.md")
    shutil.copy2(HERE / "requirements.txt", out / "requirements.txt")

    zip_path = HERE / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(name) / f.relative_to(out))

    print(f"{len(files) + 2} files -> {out.relative_to(REPO)}")
    print(f"zip: {zip_path.relative_to(REPO)} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    hits = url_audit(out)
    if hits:
        print(f"\nURLs left in the package ({len(hits)}), review before submitting:")
        for f, url in hits:
            print(f"  {f}: {url}")


if __name__ == "__main__":
    main()
