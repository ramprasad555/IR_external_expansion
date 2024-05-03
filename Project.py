"""
The project.py file, is taken from the code provided by the authors. 

"""


import pyterrier as pt
if not pt.started():
    pt.init()
import faiss
import pyterrier_colbert.ranking
from pyterrier_colbert.ranking import ColbertPRF
from pyterrier_colbert.indexing import ColBERTIndexer
import pandas as pd
from typing import Iterator, Dict, Any
from tqdm import tqdm
import os
import torch


def expansion_using_prf(topics_nfcorpus, pytcolbert, k, beta, fb_docs, name, cut_off):

    dense_e2e = pytcolbert.set_retrieve() >> pytcolbert.index_scorer(query_encoded=True)
    PRF_ext = dense_e2e % cut_off >> ColbertPRF(pytcolbert, k=k, fb_docs=fb_docs, fb_embs=10, beta=beta)
    # print(len(topics_nfcorpus))
    ## results are saved in the pkl
    result_ext = PRF_ext(topics_nfcorpus) # result of PRFtarget are stored in res, we have ["qid", "query", "query_embs", "query_toks", "query_weights"]
    pd.to_pickle(result_ext,"/home/stu4/s6/rk1668/IR_External_expansion/result/ext/" + name +".embs.pkl")

def transform(res_tgt, res_ext):
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
    rtrMerge = pd.DataFrame(columns = ["qid", "query", "query_embs", "query_toks", "query_weights"])
    qidlist = res_ext_input.qid.unique()
    for qid in qidlist:
        res_tgt = res_tgt_input[res_tgt_input["qid"] == qid]
        res_ext = res_ext_input[res_ext_input["qid"] == qid]
        new_query_df = transform(res_tgt, res_ext)
        # rtrMerge = rtrMerge.append(new_query_df)
        rtrMerge = pd.concat([rtrMerge, new_query_df])
    return rtrMerge

def process_file(file_path, pytcolbert_tgt, res_tgt, topics_nfcorpus, qrels_nfcorpus, name):
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
    pt.io.write_results(res_final, "/home/stu4/s6/rk1668/IR_External_expansion/final_result/CE.nfcorpus"+name+".res.gz")
    evalMeasuresDict = pt.Utils.evaluate(res_final, qrels_nfcorpus, metrics=["ndcg_cut_10","ndcg_cut_20","ndcg_cut_1000", "map", "recip_rank"])
    # evalMeasuresDict

    return evalMeasuresDict # returns a dict

def process_all_files(folder_path, pytcolbert_tgt, res_tgt, topics_nfcorpus, qrels_nfcorpus ):
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

def index_external():
    import pandas as pd
    from typing import Iterator, Dict, Any
    from tqdm import tqdm

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

    # Example usage
    if __name__ == "__main__":
        ps = pd.read_csv('/home/stu4/s6/rk1668/IR_External_expansion/msmarco_passages_10k.tsv', sep='\t', names = ["docno", "text"])
        ps["docno"]= ps["docno"].apply(str)

        # Creating an instance of DocumentCollection
        collection = DocumentCollection(ps)

def main():

    # indexing will done here, they are done offline.
    # write condition to check, if an index already exists
    index_root_tgt = "./index/tgt/nfcorpus"
    index_name_tgt = "nfcorpus_colbertIndex"
    dataset_tgt = pt.get_dataset("irds:beir/nfcorpus")
    # default 150, stride 75
    checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    indexer =  pt.text.sliding(text_attr="text", prepend_attr='title') >> ColBERTIndexer(checkpoint,index_root_tgt, index_name_tgt, chunksize=20)
    indexer.index(dataset_tgt.get_corpus_iter())

    # indexing for external corpus
    # write condition to check, if an index already exists
    from pyterrier_colbert.indexing import ColBERTIndexer
    import pandas as pd
    # index_root="./nfs/indices/colbert_passage"
    # index_name="external_colbertIndex"
    index_root_ext = "./index/ext/msmarco_10k"
    index_name_ext = "msmarco_colbertIndex"

    # ps = pd.read_csv('/home/stu4/s6/rk1668/IR/msmarco_passages.tsv', sep='\t', names = ["docno", "text"])
    # ps["docno"]= ps["docno"].apply(str)

    checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"

    # we have to find a dataset with format docid and text (if we use a different format, we need to index it accordingly)
    indexer = ColBERTIndexer(checkpoint,index_root_ext, index_name_ext, chunksize=20) # apply text sliding to the dataset if the dataset is not in passage format

    indexer.index(collection.get_corpus_iter(verbose=False))


    # load query sets 
    topics_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_topics()
    qrels_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_qrels()
    print(len(topics_nfcorpus))
    print(len(qrels_nfcorpus))

    index_root_tgt = "/home/stu4/s6/rk1668/IR_External_expansion/index/tgt/nfcorpus"
    index_name_tgt = "nfcorpus_colbertIndex"

    index_root_ext = "/home/stu4/s6/rk1668/IR_External_expansion/index/ext/msmarco_10k"
    index_name_ext = "msmarco_colbertIndex"

    # expansion on colbert PRF target corpus
    pytcolbert = pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", \
                                                      index_root_tgt, \
                                                      index_name_tgt,faiss_partitions=100,memtype='mem')
    pytcolbert.faiss_index_on_gpu = False

    # dense_e2eTgt = pytcolbert.set_retrieve()  >> pytcolbert.index_scorer(query_encoded=True)
    # PRF_tgt = dense_e2eTgt % 10 >> ColbertPRF(pytcolbert, k=24, fb_docs=10, fb_embs=10, beta=1)  # returns a dataframe after performing PRF, used kmeans to find nearest doc embeds
    # # print(len(topics_nfcorpus))
    # print("running the prf pipeline on target corpus")
    # result_tgt = PRF_tgt(topics_nfcorpus) # result of PRFtarget are stored in res, we have ["qid", "query", "query_embs", "query_toks", "query_weights"]
    # print("finished, save to files.")
    # pd.to_pickle(result_tgt, "/home/stu4/s6/rk1668/IR_External_expansion/result/tgt/Tgt_BEIR.nfcorpus.embs.pkl")


    # expansion on colbert prf external 
    print("expanding on external started..............")
    pytcolbert_ext = pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", \
                                                    index_root_ext, \
                                                    index_name_ext,faiss_partitions=100,memtype='mem')
    pytcolbert_ext.faiss_index_on_gpu = False
    betas = [0, 0.5, 1.0]  # 6
    cutoffs = [40] #3
    fb_docs = [3, 5, 10, 15, 20, 25, 30] # 4
    for cut_off  in cutoffs:
        for beta in betas:
            for fd in fb_docs:
                print(f"##################### \n beta {beta} fb_docs {fd}")
                expansion_using_prf(topics_nfcorpus, pytcolbert_ext, beta, fd, f"msmarco{beta}_{fd}", cut_off)    

    res_tgt = pd.read_pickle("/home/stu4/s6/rk1668/IR_External_expansion/result/tgt/Tgt_BEIR.nfcorpus.embs.pkl")
    
    final_eval_result = process_all_files("/home/stu4/s6/rk1668/IR_External_expansion/result/ext", pytcolbert,  res_tgt, topics_nfcorpus, qrels_nfcorpus)
    
    print(final_eval_result)
    
    with open('/home/stu4/s6/rk1668/IR_External_expansion/output1.txt', 'w') as file:
        for item in final_eval_result:
            file.write(f"{item}\n")
    file.close()
    
    return 


if __name__ == "__main__":
    # print("inside main hiiiiiii")
    main()
    # folder_path = "/home/stu4/s6/rk1668/IR_External_expansion/final_result"
    # /home/stu4/s6/rk1668/IR_External_expansion/final_result/CE.nfcorpusmsmarco0_3.embs.pkl.res.gz
    # all_evaluations = []
    # qrels_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_qrels()
    # for entry in os.listdir(folder_path):
    #     print(entry)
    #     full_path = os.path.join(folder_path, entry)
    #     if os.path.isfile(full_path):
    #         res_final = pd.read_csv(full_path)
    #         evalMeasuresDict = pt.Utils.evaluate(res_final, qrels_nfcorpus, metrics=["ndcg_cut_10","ndcg_cut_20","ndcg_cut_1000", "map", "recip_rank"])
    #         all_evaluations.append((entry, evalMeasuresDict))

    # with open('/home/stu4/s6/rk1668/IR_External_expansion/output1.txt', 'w') as file:
    #     for item in all_evaluations:
    #         file.write(f"{item}\n")
    # file.close()
    