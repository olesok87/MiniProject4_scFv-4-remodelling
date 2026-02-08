import os
import csv
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch, Selection
import numpy as np
import sys



# 0. Prep pdf file (can be skipped)
def clean_and_renumber_pdb(input_pdb, output_all_chains, output_single_chain, target_chain=None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    io = PDB.PDBIO()

    class SelectAllChains(PDB.Select):
        def accept_residue(self, residue):
            return residue.id[0] == " "  # Exclude heteroatoms and waters
        def accept_atom(self, atom):
            return atom.element != "H"  # Remove hydrogens

    class SelectSingleChain(PDB.Select):
        def accept_chain(self, chain):
            return chain.id == target_chain
        def accept_residue(self, residue):
            return residue.id[0] == " "
        def accept_atom(self, atom):
            return atom.element != "H"

    # Renumber all residues starting from 1 per chain
    for model in structure:
        for chain in model:
            for i, residue in enumerate(chain.get_residues(), start=1):
                residue.id = (" ", i, " ")

    # Save all chains
    io.set_structure(structure)
    io.save(output_all_chains, select=SelectAllChains())
    print("Replace input_pdb with output_all_chains_path")
    print(f"✅ Saved cleaned PDB with all chains to: {output_all_chains}")

    # Save only the specified chain
    io.set_structure(structure)
    io.save(output_single_chain, select=SelectSingleChain())
    print(f"✅ Saved cleaned PDB with chain {target_chain} to: {output_single_chain}")

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



# 3. merge CamSol and AggregaScan results
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



#from Bio import PDB
import sys

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
    #input_pdb = r"C:\Users\aszyk\PycharmProjects\Simple_helicase_mutant_selector\pdb\2P6R.pdb"
    pdb_file_camsol = r"C:\Users\aszyk\PycharmProjects\Miniproject 4 scaffold engineering\Miniproject4 scripts\Input files\scFv4_only_PDB_cleaner_CamSol.pdb"
    agg_file = r'C:\Users\aszyk\PycharmProjects\Miniproject 4 scaffold engineering\Miniproject4 scripts\Input files\A4D_scores.csv'

    #results_dir = input("📂 Results folder path: ").strip()
    results_dir=r"C:\Users\aszyk\PycharmProjects\Miniproject 4 scaffold engineering\Miniproject4 scripts\Output"
    os.makedirs(results_dir, exist_ok=True)
    #output_all_chains_path = r"C:\Users\aszyk\PycharmProjects\Simple_helicase_mutant_selector\pdb\cleaned\cleaned_all.pdb"
    #output_single_chain_path = r"C:\Users\aszyk\PycharmProjects\Simple_helicase_mutant_selector\pdb\cleaned\single_chain.pdb"


    # 0. Prep pdf file
    if input("✏️ Would you like to clean the pdb files and generate single chain pdb? (y/n): ").strip().lower() == 'y':
        chain_to_keep = input("Which chain would you like to analyse later in Rosetta?")  # Set to your desired chain
        clean_and_renumber_pdb(input_pdb, output_all_chains_path, output_single_chain_path, chain_to_keep)


    # 1. Solubility residues
    if input("✏️ Would you like to generate solubility hotspots? (y/n): ").strip().lower() == 'y':
        #        bcut = float(input("📈 B-factor/solubility threshold (e.g., -0.3): ").strip()
        bcut=-0.6
        flex = get_insoluble_residues(pdb_file_camsol, bcut)
        out1 = os.path.join(results_dir, "camsol_residues.csv")
        write_list_to_csv(out1, flex, ["chain", "resnum", "resname", "avg_bfactor"])

    # 2. Aggregation-prone residues
    if input("✏️ Would you like to generate aggregation hotspots? (y/n): ").strip().lower() == 'y':
        #agg_file = input("📄 Path to aggregation_residues.txt: ").strip()
        #scut = float(input("📈 Aggregation score threshold (e.g., 0.3): ").strip())
        scut=0.6
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

    # 4. Rosetta mut files
    write_per_residue_mutfiles(
        input_csv=out_merge,  # ← use the path you already created
        output_root=os.path.join(results_dir, "mutfiles")
    )


    # 5. Instability hotspots
    #    if input("✏️ Would you like to generate stability hotspots? (y/n): ").strip().lower() == 'y':
        #cif_path = input("Path to .exposed.pdf file: ").strip()
    #   cif_path = r"C:\Users\aszyk\PycharmProjects\Simple_helicase_mutant_selector\pdb\2P6R_exposed_surface.pdb"
    #    chain_id = input("Chain to analyze (e.g. A): ").strip()
    #    inst_list = get_instability_residues(cif_path, chain_id)
    #    out5_csv = os.path.join(results_dir, "instability_residues.csv")
    #    write_list_to_csv(out5_csv,inst_list,headers=["Chain","ResNum","ResName","Instability","Mutate to..."])





