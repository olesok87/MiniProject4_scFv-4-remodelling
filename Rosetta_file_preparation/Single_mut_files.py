import os
import csv
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch, Selection
import numpy as np
import sys

# compute script folder dynamically so paths stay valid when moving the project
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = SCRIPT_DIR
PROJECT_ROOT = SCRIPTS_ROOT  # script root == script directory

INPUT_DIR  = os.path.join(SCRIPTS_ROOT, "Input files")
OUTPUT_DIR = os.path.join(SCRIPTS_ROOT, "Output")
CLEANED_PDB_DIR = os.path.join(INPUT_DIR, "cleaned_pdb")  # always under `Input files`

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CLEANED_PDB_DIR, exist_ok=True)

# 0. Prep pdf file (I would recommend running this)

def clean_and_renumber_pdb(input_pdb, output_all_chains, output_single_chain, target_chain=None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)
    io = PDB.PDBIO()

    # Normalize chain input
    target_chain = (target_chain or "").strip()
    if target_chain == "":
        target_chain = None  # means "don't write single-chain file"

    # Collect chain IDs present (first model)
    model0 = structure[0]
    chain_ids = [ch.id for ch in model0.get_chains()]
    print("Chains found in PDB:", chain_ids)

    if target_chain is not None and target_chain not in chain_ids:
        raise ValueError(f"Chain '{target_chain}' not found in PDB. Available: {chain_ids}")

    class SelectAllChains(PDB.Select):
        def accept_residue(self, residue):
            return residue.id[0] == " "  # standard residues only
        def accept_atom(self, atom):
            return atom.element != "H"   # drop hydrogens

    class SelectSingleChain(PDB.Select):
        def accept_chain(self, chain):
            return (target_chain is None) or (chain.id == target_chain)
        def accept_residue(self, residue):
            return residue.id[0] == " "
        def accept_atom(self, atom):
            return atom.element != "H"

    # Renumber standard residues starting from 1 per chain
    for model in structure:
        for chain in model:
            i = 1
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                residue.id = (" ", i, " ")
                i += 1

    # Save all chains
    os.makedirs(os.path.dirname(output_all_chains) or ".", exist_ok=True)
    io.set_structure(structure)
    io.save(output_all_chains, select=SelectAllChains())
    print(f"✅ Saved cleaned+renumbered ALL chains to: {output_all_chains}")

    # Save only specified chain (only if user provided one)
    if target_chain is not None:
        os.makedirs(os.path.dirname(output_single_chain) or ".", exist_ok=True)
        io.set_structure(structure)
        io.save(output_single_chain, select=SelectSingleChain())

        # quick non-empty check
        with open(output_single_chain, "r") as fh:
            has_atom = any(line.startswith("ATOM") for line in fh)

        if not has_atom:
            raise RuntimeError(
                f"Single-chain output is empty. Chain '{target_chain}' matched, but no ATOM lines were written.\n"
                "This usually means the file contains only non-standard residues or selection filtered everything."
            )

        print(f"✅ Saved cleaned+renumbered chain {target_chain} to: {output_single_chain}")
    else:
        print("ℹ️ No chain provided -> skipped writing single-chain file.")


# 1. CamSol B-Factors

AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}


def get_insoluble_residues(pdb_path_camsol, bfactor_threshold=-0.6):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("structure", pdb_path_camsol)
    results = []

    for model in struct:
        for chain in model:
            for residue in chain:
                if not residue.has_id("CA"):
                    continue

                resname3 = residue.resname.upper()
                if resname3 not in AA3_TO_1:
                    continue

                wt = AA3_TO_1[resname3]

                b_vals = [atom.get_bfactor() for atom in residue]
                avg_b = np.mean(b_vals)

                if avg_b < bfactor_threshold:
                    results.append((
                        chain.id,
                        residue.id[1],
                        wt,                     # ← 1-letter code
                        round(avg_b, 2)
                    ))

    return results



# 2. AggregaScan scores
def get_aggregation_residues(input_file, score_threshold=0.6):
    results = []

    with open(input_file, "r") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        print("Header:", header)

        for row in reader:
            protein, chain, residue, residue_name, score = row
            score = float(score)

            if score > score_threshold:
                results.append((
                    chain,
                    int(residue),
                    residue_name,
                    round(score, 3)
                ))

    print(f"Found {len(results)} aggregation residues > {score_threshold}")
    return results



# 3. Merge CamSol and AggregaScan results
def merge_camsol_aggregascan(camsol_hits, aggs_hits):
    merged = {}

    # Add CamSol hits
    for chain, resnum, resname, camsol_score in camsol_hits:
        key = (chain, resnum)
        merged[key] = {
            "chain": chain,
            "resnum": resnum,
            "resname": resname,
            "camsol": camsol_score,
            "aggregascan": None,
            "source": "CamSol"
        }

    # Add AggregaScan hits
    for chain, resnum, resname, agg_score in aggs_hits:
        key = (chain, resnum)

        if key in merged:
            merged[key]["aggregascan"] = agg_score
            merged[key]["source"] = "CamSol+AggregaScan"
        else:
            merged[key] = {
                "chain": chain,
                "resnum": resnum,
                "resname": resname,
                "camsol": None,
                "aggregascan": agg_score,
                "source": "AggregaScan"
            }

    # Convert to list of tuples for CSV writing
    merged_list = [
        (
            v["chain"],
            v["resnum"],
            v["resname"],
            v["camsol"],
            v["aggregascan"],
            v["source"]
        )
        for v in merged.values()
    ]

    return merged_list


# 4. Write Rosetta mut files

ALL_AA_1 = [
    "A","C","D","E","F","G","H","I","K","L",
    "M","N","P","Q","R","S","T","V","W","Y"
]

def write_per_residue_mutfiles(
    input_csv,
    output_root="mutfiles"
):
    os.makedirs(output_root, exist_ok=True)

    with open(input_csv, newline="") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            chain = row["chain"]          # kept only for folder naming
            resnum = int(row["resnum"])
            wt = row["resname"].strip().upper()   # already 1-letter

            if wt not in ALL_AA_1:
                print(f"⚠️ Skipping unknown residue {wt} at {chain}{resnum}")
                continue

            # Folder per residue
            folder = os.path.join(
                output_root,
                f"{chain}_{resnum}_{wt}"
            )
            os.makedirs(folder, exist_ok=True)

            mutfile_path = os.path.join(folder, "mut_files.txt")

            mutations = [aa for aa in ALL_AA_1 if aa != wt]

            with open(mutfile_path, "w") as out:
                out.write(f"total {len(mutations)}\n")
                for aa in mutations:
                    out.write("1\n")
                    out.write(f"{wt} {resnum} {aa}\n")   # ← CORRECT FORMAT

            print(f"✅ Wrote {len(mutations)} mutations for {wt}{resnum}")





def get_instability_residues(pdb_path: str, chain_id: str):
    instability_map = {
        "ASN": ("Deamidation hotspot", "Mutate to D or A"),
        "GLN": ("Deamidation hotspot", "Mutate to E or A"),
        "MET": ("Oxidation-prone",      "Mutate to L or I"),
        "CYS": ("Free thiol",           "Mutate to S or A"),
        "PHE": ("Hydrophobic patch",    "Mutate to S or Y"),
        "TRP": ("Hydrophobic patch",    "Mutate to F or Y"),
        "TYR": ("Hydrophobic patch",    "Mutate to S or F"),
        "LEU": ("Hydrophobic patch",    "Mutate to T or S"),
        "ILE": ("Hydrophobic patch",    "Mutate to T or S"),
        "LYS": ("Protease site",        "Mutate to R")
    }

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    inst_residues = []

    # Check if chain exists
    if chain_id not in [chain.id for chain in model]:
        sys.exit(f"ERROR: Chain {chain_id} not found in {pdb_path}")

    for res in model[chain_id]:
        if res.id[0] != " ":  # Skip heteroatoms and waters
            continue

        resname = res.get_resname()
        if resname not in instability_map:
            continue

        inst_type, suggestion = instability_map[resname]
        resnum = res.id[1]

        inst_residues.append((chain_id, resnum, resname, inst_type, suggestion))

    return inst_residues

### Helpers: write to CSV / TXT
def write_list_to_csv(filepath, data, headers):
    with open(filepath, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"→ Wrote {len(data)} records to {filepath}")


# EXECUTION FLOW
if __name__ == "__main__":
    flex = []
    agg = []
    # pdb_file = input("🔍 PDB file path: ").strip()
    # fix
    input_pdb = os.path.join(INPUT_DIR, "6Y6C_BC_complex_cleaner.pdb")
    pdb_file_camsol = os.path.join(INPUT_DIR, "scFv4_only_PDB_cleaner_CamSol.pdb")
    agg_file = os.path.join(INPUT_DIR, "A4D_scores.csv")

    #results_dir = input("📂 Results folder path: ").strip()
    results_dir=OUTPUT_DIR
    os.makedirs(results_dir, exist_ok=True)
    output_all_chains_path = os.path.join(CLEANED_PDB_DIR, "scFv4_allchains_renum.pdb")
    # single-chain output depends on chain, but file name should differ:
    # we'll create it after user picks a chain


    # 0. Prep PDB file
    if input("✏️ Would you like to clean the pdb files and generate single chain pdb? (y/n): ").strip().lower() == 'y':
        chain_to_keep = input("Which chain would you like to analyse later in Rosetta? (e.g. C): ").strip().upper()
        output_single_chain_path = os.path.join(CLEANED_PDB_DIR, f"scFv4_chain{chain_to_keep}_renum.pdb")

        clean_and_renumber_pdb(
            input_pdb=input_pdb,
            output_all_chains=output_all_chains_path,
            output_single_chain=output_single_chain_path,
            target_chain=chain_to_keep
        )
    else:
        # if you skip cleaning, define what single-chain file you want to use
        output_single_chain_path = input_pdb

    # 1. Solubility residues
    if input("✏️ Would you like to generate solubility hotspots? (y/n): ").strip().lower() == 'y':
        #        bcut = float(input("📈 B-factor/solubility threshold (e.g., -0.3): ").strip())
        bcut=-0.6 # Pre-set threshold as per original script
        flex = get_insoluble_residues(pdb_file_camsol, bcut)
        out1 = os.path.join(results_dir, "camsol_residues.csv")
        write_list_to_csv(out1, flex, ["chain", "resnum", "resname", "avg_bfactor"])

    # 2. Aggregation-prone residues
    if input("✏️ Would you like to generate aggregation hotspots? (y/n): ").strip().lower() == 'y':
        #agg_file = input("📄 Path to aggregation_residues.txt: ").strip()
        #scut = float(input("📈 Aggregation score threshold (e.g., 0.3): ").strip())
        scut=0.6 # Pre-set threshold as per original script
        agg = get_aggregation_residues(agg_file, scut)
        out2 = os.path.join(results_dir, "aggregation_residues.csv")
        write_list_to_csv(out2, agg, ["chain", "resnum", "resname", "score"])

    # 3. Merge CamSol + AggregaScan
    merged = merge_camsol_aggregascan(flex, agg)

    out_merge = os.path.join(results_dir, "camsol_aggregascan_merged.csv")
    write_list_to_csv(
        out_merge,
        merged,
        headers=["chain", "resnum", "resname", "camsol_score", "aggregascan_score", "source"]
    )


    #4. Rosetta mut files

    write_per_residue_mutfiles(
        input_csv=out_merge,
        output_root=os.path.join(results_dir, "mutfiles")
    )


    # 5. Instability hotspots
    if input("✏️ Would you like to generate instability hotspots? (y/n): ").strip().lower() == 'y':
        instability_cif_path = input("Path to PDB file for instability analysis: ").strip()
        instability_chain_id = input("Chain to analyze for instability (e.g. A): ").strip()
        inst_list = get_instability_residues(instability_cif_path, instability_chain_id)
        out5_csv = os.path.join(results_dir, "instability_residues.csv")
        write_list_to_csv(out5_csv,inst_list,headers=["Chain","ResNum","ResName","Instability","Mutate to..."])
