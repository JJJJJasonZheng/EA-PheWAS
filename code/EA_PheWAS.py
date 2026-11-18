import os
import sys
import pickle
import argparse
import pandas as pd
from pyhpo import Ontology
from pyhpo.annotations import Gene

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from EmbedPheScan import phewas_by_embedding_with_zpvalues
from PheWAS import run_carrierlist_phewas_hybrid_cov
from ACAT_PheWAS import (
    add_acat_pvalues_from_embed_and_lr,
    acat_df_to_combined_ranking,
    build_phecode_mapping,
    get_gene_carriers_id,
    get_group_embed,
    make_weighted_phecode_embeddings,
)
from eval_hit_precision import evaluate_gene_from_summary_pkl
from embedding_model.embed_w2v import train_icd_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--gene", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--hesin", type=str, default=os.path.join(ROOT_DIR, "sample_data/hesin_sample.txt"))
parser.add_argument("--cov", type=str, default=os.path.join(ROOT_DIR, "sample_data/cov_adjust_sample.txt"))
parser.add_argument("--carrier_file", type=str, default=os.path.join(ROOT_DIR, "sample_data/carrier_sample.txt"))
parser.add_argument("--icd2phecode", type=str, default=os.path.join(ROOT_DIR, "sample_data/icd2phecode_des.csv"))
parser.add_argument("--phecode_map", type=str, default=os.path.join(ROOT_DIR, "sample_data/Phecode_map12_filtered.csv"))
parser.add_argument("--hpo_map", type=str, default=os.path.join(ROOT_DIR, "sample_data/hpo-phecode1.2_links.tsv"))
parser.add_argument("--icd_embed", type=str, default=None, help="Optional precomputed ICD embedding TSV/PKL")
parser.add_argument("--ind_embed", type=str, default=None, help="Optional precomputed individual embedding PKL")
parser.add_argument("--icd_freq", type=str, default=None, help="Optional ICD frequency CSV (weight_col='Freq')")
parser.add_argument("--embed_name", type=str, default="w2vicd500k_embed")
parser.add_argument("--min_cases", type=int, default=5)
parser.add_argument("--B", type=int, default=10000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sample_from", choices=["all", "noncarriers"], default="all")
parser.add_argument("--with_replacement", action="store_true", help="Sample pseudo-carriers with replacement (default: without replacement)")
args = parser.parse_args()

GENE = args.gene
embed_name = args.embed_name

# Load covariates, ICD-10 diagnosis records, and phecode mappings
cov_adjust = pd.read_csv(args.cov, sep="\t")
hesin_diag = pd.read_csv(args.hesin, sep="\t")
icd2phecode = pd.read_csv(args.icd2phecode)
phecode_mapping = build_phecode_mapping(pd.read_csv(args.phecode_map))

icd_to_phecode = icd2phecode.dropna(subset=["diag_icd10", "PheCode"]).drop_duplicates("diag_icd10").assign(PheCode=lambda x: x["PheCode"].astype(str).str.strip()).set_index("diag_icd10")["PheCode"].to_dict()
hesin_use = hesin_diag.assign(phecode=hesin_diag["diag_icd10"].map(icd_to_phecode))[["eid", "phecode"]].dropna().drop_duplicates()
eid_use_phewas = hesin_use["eid"].drop_duplicates().tolist()

# Build or load ICD-level, individual-level (loaded_embeddings), and phecode-level (phecode_embeddings) dictionaries
if args.icd_embed and args.icd_embed.endswith((".pkl", ".pickle")):
    with open(args.icd_embed, "rb") as f:
        icd_embed = pickle.load(f)
elif args.icd_embed:
    df_emb = pd.read_csv(args.icd_embed, sep="\t", index_col=0)
    icd_embed = {str(k): row.values for k, row in df_emb.iterrows()}
else:
    df_emb = train_icd_embeddings(hesin_diag, seed=args.seed)
    icd_embed = {str(k): row.values for k, row in df_emb.iterrows()}

if args.ind_embed:
    with open(args.ind_embed, "rb") as f:
        raw_ind = pickle.load(f)
    loaded_embeddings = raw_ind if f"ind_{embed_name}" in raw_ind else {f"ind_{embed_name}": raw_ind}
else:
    loaded_embeddings = {f"ind_{embed_name}": get_group_embed(eid_use_phewas, hesin_diag, icd_embed)}

freq_df = pd.read_csv(args.icd_freq, index_col=0) if args.icd_freq else None
phecode_embeddings = {f"icd_{embed_name}": make_weighted_phecode_embeddings(icd2phecode, icd_embed, freq_df=freq_df)}

c_embed = get_gene_carriers_id(args.carrier_file, hesin_use)
embed_res = phewas_by_embedding_with_zpvalues(
    c_embed, embed_name, loaded_embeddings, phecode_embeddings, phecode_mapping,
    B=args.B, seed=args.seed, sample_from=args.sample_from, replace=args.with_replacement,
)

c_phewas = get_gene_carriers_id(args.carrier_file, hesin_use)
phewas_res = run_carrierlist_phewas_hybrid_cov(c_phewas, hesin_use, 'phecode', eid_use_phewas, cov_adjust, phecode_mapping, min_cases=args.min_cases)

acat_df = add_acat_pvalues_from_embed_and_lr(embed_res, phewas_res)
combined_ranking = acat_df_to_combined_ranking(acat_df, keep_all=False)

gene_dir = os.path.join(args.out_dir, GENE)
os.makedirs(gene_dir, exist_ok=True)
out_path = os.path.join(gene_dir, f"{GENE}_summary.pkl")

with open(out_path, "wb") as f:
    pickle.dump({
        "gene": GENE, "embed_results": embed_res, "phewas_res": phewas_res, 
        "acat_df": acat_df, "combined_ranking": combined_ranking
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

_ = Ontology()
hpo_ids = []
try:
    hpo_ids = [t.id.replace(":", "_") for t in list(Gene.get(GENE).hpo_set())]
except KeyError:
    pass

hpo_map = pd.read_table(args.hpo_map)
gt_phecodes = hpo_map[hpo_map['hpo_code'].isin(hpo_ids)][['phecode1.2_code', 'phecode1.2_label', 'phecode1.2_category', 'hpo_code', 'hpo_label']].drop_duplicates()

with open(os.path.join(gene_dir, f"{GENE}_hpo.pkl"), "wb") as f:
    pickle.dump(gt_phecodes, f, protocol=pickle.HIGHEST_PROTOCOL)

summary, _ = evaluate_gene_from_summary_pkl(out_path, gt_phecodes, k_values=(5, 10, 15))
with open(os.path.join(gene_dir, f"{GENE}_eval.pkl"), "wb") as f:
    pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
print(summary)
