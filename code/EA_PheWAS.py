import os
import pickle
import argparse
import pandas as pd
from pyhpo import Ontology
from pyhpo.annotations import Gene

from EmbedPheScan_500k import phewas_by_embedding_with_zpvalues
from PheWAS_500k import run_carrierlist_phewas_hybrid_cov
from ACAT_PheWAS import add_acat_pvalues_from_embed_and_lr, acat_df_to_combined_ranking
from eval_hit_precision import evaluate_gene_from_summary_pkl

parser = argparse.ArgumentParser()
parser.add_argument("--gene", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--hpo_map", type=str, required=True)
args = parser.parse_args()

GENE = args.gene

# Assuming loaded_embeddings, phecode_embeddings, phecode_mapping, hesin_use, etc., are loaded in scope
c_embed = get_gene_500K_LoFcarriers_id_ANNOVAR(GENE, hesin_use)
embed_res = phewas_by_embedding_with_zpvalues(c_embed, "w2vicd500k_embed", loaded_embeddings, phecode_embeddings, phecode_mapping)

c_phewas = get_gene_500K_LoFcarriers_id_ANNOVAR(GENE, hesin_use)
phewas_res = run_carrierlist_phewas_hybrid_cov(c_phewas, hesin_use, 'phecode', eid_use_phewas, cov_adjust, phecode_mapping)

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

summary, _ = evaluate_gene_from_summary_pkl(out_path, gt_phecodes, (5, 10, 15))
with open(os.path.join(gene_dir, f"{GENE}_eval.pkl"), "wb") as f:
    pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)