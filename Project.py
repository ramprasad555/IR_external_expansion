"""
The project.py file, is taken from the code repository provided by the authors for the paper -
IPM paper: "Improving zero-shot retrieval using dense external expansion".

Original Authors - Xiao Wang, Craig Macdonald, Iadh Ounis

"""

import pyterrier as pt
if not pt.started():
    pt.init()
import faiss
import pyterrier_colbert.ranking
from pyterrier_colbert.ranking import ColbertPRF
from pyterrier_colbert.indexing import ColBERTIndexer
import pandas as pd
from tqdm import tqdm
import os
import torch
import argparse


def expansion_using_prf(topics_nfcorpus, pytcolbert, k, beta, fb_docs, name, cut_off):
    """
    This function uses Colbert PRF to generate new query embeddings based on the external corpus.
    param topics_nfcorpus: topics of nfcorpus dataset.
    param pytcolbert: instance of colbertFactory with external index.
    param beta : weighting factor for the external embeddings.
    param fb_docs: number of psuedo relevance feedback documents.
    param name: file name the result embeddings to be saved.
    param cutoff : cutoff value to be considered in the PRF pipeline.
    """
    dense_e2e = pytcolbert.set_retrieve() >> pytcolbert.index_scorer(query_encoded=True)
    PRF_ext = dense_e2e % cut_off >> ColbertPRF(pytcolbert, k=k, fb_docs=fb_docs, fb_embs=10, beta=beta)
    ## results are saved in the pkl
    result_ext = PRF_ext(topics_nfcorpus) # result of PRFtarget are stored in res, we have ["qid", "query", "query_embs", "query_toks", "query_weights"]
    pd.to_pickle(result_ext,"./embeddings/ext/" + name +".embs.pkl")

def transform(res_tgt, res_ext):
    """
    The function modifies the target query by incorporating additional query embeddings from external corpus
    param res_tgt: the query embeddings, result of colbert-PRF on target Corpus
    param res_ext: the query embeddings, result of colbert-PRF on external Corpus
    """
    first_row = res_ext.iloc[0]
    
    # concatenate the external embeddings to the target query embeddings 
    newemb = torch.cat((res_tgt.iloc[0].query_embs,res_ext.iloc[0].query_embs[32:]),0)

    # the weights column defines important of each query embedding
    newweights = torch.cat((res_tgt.iloc[0].query_weights, res_ext.iloc[0].query_weights[32:]),0)
    newtoks = res_tgt.iloc[0].query_toks+ (res_ext.iloc[0].query_toks)

    # generate the revised query dataframe row
    rtr = pd.DataFrame([
        [first_row.qid, 
        #  first_row.docno,
         first_row.query, 
         newemb, 
         newtoks, 
         newweights]
        ],
        columns=["qid", "query", "query_embs", "query_toks", "query_weights"])
    return rtr

def mergePRF(res_tgt_input, res_ext_input):
    """
    The function concatenates the query embeddings of target and external corpus.
    param res_tgt_input: the query embeddings, result of colbert-PRF on target Corpus
    param res_ext_input: the query embeddings, result of colbert-PRF on external Corpus
    """
    rtrMerge = pd.DataFrame(columns = ["qid", "query", "query_embs", "query_toks", "query_weights"])
    qidlist = res_ext_input.qid.unique()
    for qid in qidlist:
        res_tgt = res_tgt_input[res_tgt_input["qid"] == qid]
        res_ext = res_ext_input[res_ext_input["qid"] == qid]
        new_query_df = transform(res_tgt, res_ext)
        rtrMerge = pd.concat([rtrMerge, new_query_df])
    return rtrMerge

def process_file(file_path, pytcolbert_tgt, res_tgt, topics_nfcorpus, qrels_nfcorpus, name):
    """
    This function reads the saved the query embeddings of external corpus and merges it with query embeddings of external corpus 
    and runs experiments and evaluates the scores based on the retrieved results.
    param filepath: The file path to the query embeddings of external corpus 
    param pytcolbert_tgt: instance of colbertFactory with target index
    param res_tgt: the query embeddings, result of colbert-PRF on target Corpus
    param topics_nfcorpus: topics of nfcorpus dataset.
    param qrels_nfcorfus: qrels of nfcorpus dataset.
    param name: the file name to which the results are saved to.
    """

    res_ext = pd.read_pickle(file_path)
    resMerge = mergePRF(res_tgt, res_ext) 
    # resMerge.head()


    MergeTrans = pt.transformer.SourceTransformer(resMerge)
    pipeCE2_ranker = (MergeTrans 
                    >> pytcolbert_tgt.set_retrieve(query_encoded=True) 
                    >> (pytcolbert_tgt.index_scorer(query_encoded=True, add_ranks=True, batch_size=5000)
                    >> pt.text.max_passage() % 1000))
    
    # evaluation
    res_final = pd.concat([resPart for resPart in pipeCE2_ranker.transform_gen(topics_nfcorpus, batch_size=100)])
    pt.io.write_results(res_final, "./final_result/CE.nfcorpus"+name+".res.gz")
    evalMeasuresDict = pt.Utils.evaluate(res_final, qrels_nfcorpus, metrics=["ndcg_cut_10","ndcg_cut_20","ndcg_cut_1000", "map", "recip_rank"])

    return evalMeasuresDict # returns a dict

def process_all_files(folder_path, pytcolbert_tgt, res_tgt, topics_nfcorpus, qrels_nfcorpus ):
    """
    This function loops over all the files in the result/ext folder () and fetches the evaluation results.
    param folder_path: The folder path to the query embeddings of external corpus 
    param pytcolbert_tgt: instance of colbertFactory with target index
    param res_tgt: the query embeddings, result of colbert-PRF on target Corpus
    param topics_nfcorpus: topics of nfcorpus dataset.
    param qrels_nfcorpus: qrels of nfcorpus dataset.
    """

    # List to store all evaluation results
    all_evaluations = []
    
    # Process each file in the directory
    for entry in os.listdir(folder_path):
        print(entry)
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            eval_result = process_file(full_path,  pytcolbert_tgt, res_tgt, topics_nfcorpus, qrels_nfcorpus, entry)
            all_evaluations.append((entry, eval_result))
    return all_evaluations

def main():

    parser = argparse.ArgumentParser(description="Process multiple values for beta, fb_docs, and cutoff.")
    parser.add_argument('--beta', type=float, nargs='+', help="Space-separated list of beta values (e.g., --beta 0.5 0.7)")
    parser.add_argument('--fb_docs', type=int, nargs='+', help="Space-separated list of fb_docs values (e.g., --fb_docs 10 20)")
    parser.add_argument('--cutoff', type=int, nargs='+', help="Space-separated list of cutoff values (e.g., --cutoff 100 200)")
    args = parser.parse_args()

    # indexing will done here, they are done offline.
    # index_root_tgt = "./index/tgt/nfcorpus"
    # index_name_tgt = "nfcorpus_colbertIndex"
    # dataset_tgt = pt.get_dataset("irds:beir/nfcorpus")

    # # default 150, stride 75
    # checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    # indexer =  pt.text.sliding(text_attr="text", prepend_attr='title') >> ColBERTIndexer(checkpoint,index_root_tgt, index_name_tgt, chunksize=20)
    # indexer.index(dataset_tgt.get_corpus_iter())

    # # indexing for external corpus
 
    # # index_root="./nfs/indices/colbert_passage"
    # # index_name="external_colbertIndex"
    # index_root_ext = "./index/ext/msmarco_10k"
    # index_name_ext = "msmarco_colbertIndex"

    # # ps = pd.read_csv('/home/stu4/s6/rk1668/IR/msmarco_passages.tsv', sep='\t', names = ["docno", "text"])
    # # ps["docno"]= ps["docno"].apply(str)

    # checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"

    # # we have to find a dataset with format docid and text (if we use a different format, we need to index it accordingly)
    # indexer = ColBERTIndexer(checkpoint,index_root_ext, index_name_ext, chunksize=20) # apply text sliding to the dataset if the dataset is not in passage format

    # indexer.index(collection.get_corpus_iter(verbose=False))


    # load query sets 
    topics_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_topics()
    qrels_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_qrels()

    index_root_tgt = "./index/tgt/nfcorpus"
    index_name_tgt = "nfcorpus_colbertIndex"

    index_root_ext = "./index/ext/msmarco_10k"
    index_name_ext = "msmarco_colbertIndex"

    # expansion on colbert PRF target corpus
    pytcolbert_tgt= pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", \
                                                      index_root_tgt, \
                                                      index_name_tgt,faiss_partitions=100,memtype='mem')
    pytcolbert_tgt.faiss_index_on_gpu = False
    dense_e2eTgt = pytcolbert_tgt.set_retrieve()  >> pytcolbert_tgt.index_scorer(query_encoded=True)
    PRF_tgt = dense_e2eTgt % 10 >> ColbertPRF(pytcolbert_tgt, k=24, fb_docs=10, fb_embs=10, beta=1)  # returns a dataframe after performing PRF, used kmeans to find nearest doc embeds
    print("running the prf pipeline on target corpus")
    result_tgt = PRF_tgt(topics_nfcorpus) # result of PRFtarget are stored in res, we have ["qid", "query", "query_embs", "query_toks", "query_weights"]
    print("finished, saving output of prf pipeline on target corpus to files.")
    pd.to_pickle(result_tgt, "./embeddings/tgt/Tgt_BEIR.nfcorpus.embs.pkl")

    # expansion on colbert prf external 
    print("expanding on external started..............")
    pytcolbert_ext = pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", \
                                                    index_root_ext, \
                                                    index_name_ext,faiss_partitions=100,memtype='mem')
    pytcolbert_ext.faiss_index_on_gpu = False
    betas = args.beta
    cutoffs = args.cutoff
    fb_docs = args.fb_docs # number of feedback documents for the psuedo relavance feedback 
    for cut_off  in cutoffs:
        for beta in betas:
            for fd in fb_docs: 
                print(f"##################### \n beta {beta} fb_docs {fd}")
                expansion_using_prf(topics_nfcorpus, pytcolbert_ext, beta, fd, f"msmarco{beta}_{fd}", cut_off)   

    # reads the query embeddings generated on the target corpus
    res_tgt = pd.read_pickle("./embeddings/tgt/Tgt_BEIR.nfcorpus.embs.pkl")
    final_eval_result = process_all_files("./embeddings/ext", pytcolbert_tgt,  res_tgt, topics_nfcorpus, qrels_nfcorpus)

    print(final_eval_result)

    with open('./output.txt', 'w') as file:
        for item in final_eval_result:
            file.write(f"{item}\n")
    file.close()
    return 


if __name__ == "__main__":
    """
    This is the main guard of the program
    """
    main()