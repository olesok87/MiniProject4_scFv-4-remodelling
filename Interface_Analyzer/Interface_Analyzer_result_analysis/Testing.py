from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================
# PATHS (EDIT ONLY THESE)
# =========================
MUTANT_RESULTS_DIR = Path(
    r"C:\Users\aszyk\PycharmProjects\Miniproject 4 scaffold engineering\Miniproject4 scripts\Interface_Analyzer\Interface_Analyzer_result_analysis\Rosetta_sc_files"
)

WT_SCOREFILE = MUTANT_RESULTS_DIR / "WT.sc"

PDB_FILE = Path(
    r"C:\Users\aszyk\PycharmProjects\Miniproject 4 scaffold engineering\Miniproject4 scripts\Rosetta_file_preparation\Input files\cleaned_pdb\scFv4_chainC_renum.pdb"
)

# Output CSVs
ALL_RESULTS_CSV = MUTANT_RESULTS_DIR.parent / "all_results.csv"
FINAL_RESULTS_CSV = MUTANT_RESULTS_DIR.parent / "final_results.csv"
FINAL_BY_AA_CSV = MUTANT_RESULTS_DIR.parent / "final_results_by_aa.csv"

SORT_METRIC = "dG_separated"

SIGNIFICANT_COLUMNS = [
    "dG_separated",
    "dG_cross",
    "dG_separated/dSASAx100",
    "dG_cross/dSASAx100",
    "dSASA_int",
    "dSASA_hphobic",
    "dSASA_polar",
    "hbonds_int",
    "delta_unsatHbonds",
    "packstat",
    "complex_normalized",
    "total_score",
]


# =========================
# PDB PARSING
# =========================
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


def build_pdb_wt_map(pdb_path: Path) -> Dict[Tuple[str, int], str]:
    """
    Return mapping: (chain, resseq) -> WT one-letter amino acid
    Uses the first ATOM encountered per (chain, resseq).
    """
    wt_map: Dict[Tuple[str, int], str] = {}
    seen: set[Tuple[str, int]] = set()

    with pdb_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 26:
                continue

            chain = line[21].strip()
            resn3 = line[17:20].strip().upper()
            resseq_str = line[22:26].strip()
            if not chain or not resseq_str or not resseq_str.lstrip("-").isdigit():
                continue
            resseq = int(resseq_str)

            key = (chain, resseq)
            if key in seen:
                continue
            seen.add(key)

            wt_map[key] = AA3_TO_1.get(resn3, "X")

    if not wt_map:
        raise ValueError(f"Failed to parse any residues from PDB: {pdb_path}")
    return wt_map


# =========================
# ROSETTA SCORE PARSING
# =========================
def parse_rosetta_scorefile(scorefile: Path) -> Dict[str, str]:
    """
    Reads a Rosetta scorefile (.sc/.txt) and returns the FIRST SCORE row as dict.
    Looks for:
      SCORE: <headers...> description
      SCORE: <values...> <desc>
    """
    header: Optional[List[str]] = None
    data: Optional[List[str]] = None

    with scorefile.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2 or parts[0] != "SCORE:":
                continue

            if header is None and "description" in parts:
                header = parts[1:]  # skip SCORE:
                continue

            if header is not None:
                data = parts[1:]  # skip SCORE:
                break

    if header is None:
        raise ValueError(f"No SCORE header found in {scorefile}")
    if data is None:
        raise ValueError(f"No SCORE data row found in {scorefile}")

    if len(data) < len(header):
        data = data + [""] * (len(header) - len(data))
    elif len(data) > len(header):
        data = data[: len(header)]

    return dict(zip(header, data))


def try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def find_mutant_scorefiles(root: Path) -> List[Path]:
    exts = {".sc", ".txt"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.name.lower() not in {"wt.sc", "wt.txt"}:
            try:
                parse_rosetta_scorefile(p)
                files.append(p)
            except Exception:
                pass
    return sorted(files)


# =========================
# MUTATION TOKEN PARSING
# =========================
RE_COMPACT = re.compile(r"(?<![A-Za-z0-9])(?P<chain>[A-Za-z])(?P<pos>\d+)(?P<mut>[A-Za-z])(?![A-Za-z0-9])")
RE_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])(?P<chain>[A-Za-z])_(?P<pos>\d+)_(?P<mut>[A-Za-z])(?![A-Za-z0-9])")

# From our output variant labels: "C A100Y"
RE_VARIANT_LABEL = re.compile(r"^(?P<chain>[A-Za-z])\s+(?P<wt>[A-Za-z])(?P<pos>\d+)(?P<mut>[A-Za-z])$")


def extract_mutation(text: str) -> Optional[Tuple[str, int, str]]:
    for rx in (RE_UNDERSCORE, RE_COMPACT):
        m = rx.search(text)
        if m:
            return m.group("chain").upper(), int(m.group("pos")), m.group("mut").upper()
    return None


def make_variant_label(scorefile: Path, row: Dict[str, str], pdb_wt_map: Dict[Tuple[str, int], str]) -> str:
    token = extract_mutation(scorefile.stem) or extract_mutation(scorefile.name) or extract_mutation(row.get("description", ""))
    if not token:
        return scorefile.stem  # fallback

    chain, pos, mut = token
    wt = pdb_wt_map.get((chain, pos), "X")
    return f"{chain} {wt}{pos}{mut}"


def parse_variant_pos(variant: str) -> Optional[int]:
    """
    Return amino acid position from variant label like:
      - WT -> None
      - "C A100Y" -> 100
      - fallback formats -> None
    """
    m = RE_VARIANT_LABEL.match(variant.strip())
    if m:
        return int(m.group("pos"))
    return None


# =========================
# MAIN
# =========================
def main() -> None:
    print("Using:")
    print(f"  MUTANT_RESULTS_DIR: {MUTANT_RESULTS_DIR}")
    print(f"  WT_SCOREFILE      : {WT_SCOREFILE}")
    print(f"  PDB_FILE          : {PDB_FILE}")
    print(f"  ALL_RESULTS_CSV   : {ALL_RESULTS_CSV}")
    print(f"  FINAL_RESULTS_CSV : {FINAL_RESULTS_CSV}")
    print(f"  FINAL_BY_AA_CSV   : {FINAL_BY_AA_CSV}\n")

    if not PDB_FILE.exists():
        raise SystemExit(
            "ERROR: PDB file not found.\n"
            "Please make sure the file is named exactly 'scFv4_chainC_renum.pdb' and placed at:\n"
            f"  {PDB_FILE}"
        )
    if not MUTANT_RESULTS_DIR.exists():
        raise SystemExit(f"ERROR: mutants folder not found:\n  {MUTANT_RESULTS_DIR}")
    if not WT_SCOREFILE.exists():
        raise SystemExit(
            "ERROR: WT scorefile not found.\n"
            "Please make sure WT scorefile is named exactly 'WT.sc' and placed at:\n"
            f"  {WT_SCOREFILE}"
        )

    pdb_wt_map = build_pdb_wt_map(PDB_FILE)

    wt_row = parse_rosetta_scorefile(WT_SCOREFILE)
    mutant_files = find_mutant_scorefiles(MUTANT_RESULTS_DIR)

    if not mutant_files:
        raise SystemExit(f"ERROR: No mutant scorefiles found in {MUTANT_RESULTS_DIR}")

    # Build combined records (WT + mutants)
    records: List[Dict[str, str]] = []
    wt_rec = {"variant": "WT"}
    wt_rec.update(wt_row)
    records.append(wt_rec)

    missing_lookup = 0
    for sf in mutant_files:
        row = parse_rosetta_scorefile(sf)
        variant = make_variant_label(sf, row, pdb_wt_map)
        if " X" in variant:  # e.g. "C X100Y"
            missing_lookup += 1
        rec = {"variant": variant}
        rec.update(row)
        records.append(rec)

    if missing_lookup:
        print(f"WARNING: {missing_lookup} variants got WT residue 'X' (PDB lookup failed).")
        print("This usually means residue numbering or chain ID in the mutation token doesn't match the PDB.\n")

    # Columns (no file paths)
    base_cols = ["variant"]
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    other_cols = sorted([k for k in all_keys if k not in base_cols and k != "scorefile"])
    columns = base_cols + other_cols

    # Sort all_results: WT first; then by dG_separated ascending
    def sort_key(rec: Dict[str, str]):
        if rec.get("variant") == "WT":
            return (-1e30,)
        v = try_float(rec.get(SORT_METRIC, ""))
        return (v if v is not None else 1e30,)

    records_sorted = sorted(records, key=sort_key)

    with ALL_RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in records_sorted:
            w.writerow({c: r.get(c, "") for c in columns})

    # WT numeric for subtraction
    wt_numeric: Dict[str, float] = {}
    for c in SIGNIFICANT_COLUMNS:
        v = try_float(wt_row.get(c, ""))
        if v is not None:
            wt_numeric[c] = v

    ddg_sort_col = f"{SORT_METRIC}_mut_minus_WT" if SORT_METRIC in wt_numeric else None
    final_cols = ["variant"] + [f"{c}_mut_minus_WT" for c in wt_numeric.keys()] + [SORT_METRIC]

    final_rows: List[Dict[str, str]] = []
    for r in records_sorted:
        out: Dict[str, str] = {"variant": r["variant"], SORT_METRIC: r.get(SORT_METRIC, "")}
        for c, wt_val in wt_numeric.items():
            key = f"{c}_mut_minus_WT"
            if r["variant"] == "WT":
                out[key] = "0.0"
            else:
                mv = try_float(r.get(c, ""))
                out[key] = f"{(mv - wt_val):.6f}" if mv is not None else ""
        final_rows.append(out)

    def final_sort_key(fr: Dict[str, str]):
        if fr["variant"] == "WT":
            return (-1e30,)
        if ddg_sort_col is not None:
            v = try_float(fr.get(ddg_sort_col, ""))
            return (v if v is not None else 1e30,)
        v2 = try_float(fr.get(SORT_METRIC, ""))
        return (v2 if v2 is not None else 1e30,)

    final_rows_sorted = sorted(final_rows, key=final_sort_key)

    with FINAL_RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=final_cols)
        w.writeheader()
        for r in final_rows_sorted:
            w.writerow({c: r.get(c, "") for c in final_cols})

    # =========================================================
    # NEW: final_results_by_aa.csv
    # - Uses all_results.csv data (our in-memory records are identical)
    # - Sorted by amino acid number (position)
    # - Includes WT subtraction columns (mut - WT)
    # =========================================================
    by_aa_rows: List[Dict[str, str]] = []
    for r in records:
        variant = r["variant"]
        pos = parse_variant_pos(variant)

        out: Dict[str, str] = {"variant": variant, "aa_pos": "" if pos is None else str(pos), SORT_METRIC: r.get(SORT_METRIC, "")}
        for c, wt_val in wt_numeric.items():
            key = f"{c}_mut_minus_WT"
            if variant == "WT":
                out[key] = "0.0"
            else:
                mv = try_float(r.get(c, ""))
                out[key] = f"{(mv - wt_val):.6f}" if mv is not None else ""
        by_aa_rows.append(out)

    # Sort: WT first, then increasing aa_pos, then variant string
    def by_aa_sort_key(fr: Dict[str, str]):
        if fr["variant"] == "WT":
            return (-1, -1, fr["variant"])
        p = try_float(fr.get("aa_pos", ""))
        if p is None:
            return (1_000_000, 1_000_000, fr["variant"])
        return (0, int(p), fr["variant"])

    by_aa_sorted = sorted(by_aa_rows, key=by_aa_sort_key)

    by_aa_cols = ["variant", "aa_pos"] + [f"{c}_mut_minus_WT" for c in wt_numeric.keys()] + [SORT_METRIC]
    with FINAL_BY_AA_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=by_aa_cols)
        w.writeheader()
        for r in by_aa_sorted:
            w.writerow({c: r.get(c, "") for c in by_aa_cols})

    print("Done.")
    print(f"Wrote: {ALL_RESULTS_CSV}")
    print(f"Wrote: {FINAL_RESULTS_CSV}")
    print(f"Wrote: {FINAL_BY_AA_CSV}")


if __name__ == "__main__":
    main()
