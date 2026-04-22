import pandas as pd
from gensim.models import Word2Vec

def train_icd_embeddings(df_diag, eid_col='eid', icd_col='diag_icd10', size=100, window=5, min_count=1, workers=4):
    sentences = df_diag.groupby(eid_col)[icd_col].apply(list).tolist()
    model = Word2Vec(sentences, vector_size=size, window=window, min_count=min_count, workers=workers)
    
    codes = model.wv.index_to_key
    emb_df = pd.DataFrame(model.wv[codes], index=codes)
    emb_df.columns = [f"dim_{i+1}" for i in range(emb_df.shape[1])]
    
    return emb_df