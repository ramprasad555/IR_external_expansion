"""
Code for indexing with  hardcoded datasets.
"""

import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_colbert.indexing import ColBERTIndexer
import pandas as pd
from typing import Iterator, Dict, Any
from tqdm import tqdm

# indexing will done here, they are done offline.
index_root_tgt = "./index/tgt/nfcorpus"
index_name_tgt = "nfcorpus_colbertIndex"
dataset_tgt = pt.get_dataset("irds:beir/nfcorpus")

# default 150, stride 75
checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
indexer =  pt.text.sliding(text_attr="text", prepend_attr='title') >> ColBERTIndexer(checkpoint,index_root_tgt, index_name_tgt, chunksize=20)
indexer.index(dataset_tgt.get_corpus_iter())


# indexing for external corpus
index_root_ext = "./index/ext/msmarco_10k"
index_name_ext = "msmarco_colbertIndex"

class DocumentCollection:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_corpus_iter(self, verbose=True) -> Iterator[Dict[str, Any]]:
        """
        Returns an iter of dicts for this collection. If verbose=True, a tqdm pbar shows the progress over this iterator.
        """
        iterator = self.dataframe.iterrows()
        if verbose:
            iterator = tqdm(iterator, total=len(self.dataframe))

        for _, row in iterator:
            yield {
                'docno': row['docno'],
                'text': row['text']
            }
ps = pd.read_csv('./msmarco_passages_10k.tsv', sep='\t', names = ["docno", "text"])
ps["docno"]= ps["docno"].apply(str)

# Creating an instance of DocumentCollection
collection = DocumentCollection(ps)

checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
# we have to find a dataset with format docid and text (if we use a different format, we need to index it accordingly)
indexer = ColBERTIndexer(checkpoint,index_root_ext, index_name_ext, chunksize=20) # apply text sliding to the dataset if the dataset is not in passage format

indexer.index(collection.get_corpus_iter(verbose=False))

