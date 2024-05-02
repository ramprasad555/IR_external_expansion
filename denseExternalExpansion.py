import faiss
import pyterrier as pt
if not pt.started():
    pt.init()

import pyterrier_colbert.ranking
from pyterrier_colbert.ranking import ColbertPRF
from pyterrier_colbert.ranking import ColBERTFactory
import pandas as pd
from pyterrier_colbert.indexing import ColBERTIndexer


def transform(res_tgt, res_ext):
    import torch
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


# Indexing for the Target Corpus
def main():
    # the flow of the methodology as presented by the paper.
    # index the target and external corpus
    # load the query sets
    # apply colbert PRf on target and apply colbert prf on external
    # external dense expansion
        # get the saved query embds
        # concatenate the query embds for each query. 
        # Run the new retrievalpipeline.

    ### TO-DO
    # RQ - fb_embds and fb_docs, what are the optimal values for these ?
    # RQ1 - How do weights for internal embeddings vs external embeddings effect the performance (Effectiveness - NDCG@10, MRR)
    # hypothesis - 
    # write an experiment 
    # baseline BM25 on Target corpus
    #   

    # indexing the target corpus
    index_root="./nfs/indices/BEIR/nfcorpus"
    index_name="nfcorpus_colbertIndex"
    dataset = pt.get_dataset("irds:beir/nfcorpus")
    # default 150, stride 75
    checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    indexer =  pt.text.sliding(text_attr="text", prepend_attr='title') >> ColBERTIndexer(checkpoint,index_root, index_name, chunksize=20)
    indexer.index(dataset.get_corpus_iter())

    # Indexing for the External Corpus
    index_root="./nfs/indices/colbert_passage"
    # index_name="external_colbertIndex"
    index_name = "external_scifact"
    dataset = pt.get_dataset("irds:beir/scifact")
    checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"

    # we have to find a dataset with format docid and text (if we use a different format, we need to index it accordingly)
    indexer = pt.text.sliding(text_attr="text", prepend_attr='title') >> ColBERTIndexer(checkpoint,index_root, index_name, chunksize=20) # apply text sliding to the dataset if the dataset is not in passage format
    indexer.index(dataset.get_corpus_iter())

    # Loading query set
    topics_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_topics()
    qrels_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_qrels()
    print(len(topics_nfcorpus))
    print(len(qrels_nfcorpus))

    # ColBERT-PRF on Target Corpus
    # evaluate function needs this instance of pytcolbertTgt
    pytcolbertTgt = pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", \
                                                        "./nfs/indices/BEIR/nfcorpus", \
                                                        "nfcorpus_colbertIndex",faiss_partitions=100,memtype='mem')
    pytcolbertTgt.faiss_index_on_gpu = False
    dense_e2eTgt = pytcolbertTgt.set_retrieve() >> pytcolbertTgt.index_scorer(query_encoded=True)
    PRFTgt = dense_e2eTgt %10 >> ColbertPRF(pytcolbertTgt, k=24, fb_docs=10, fb_embs=10, beta=2)  # returns a dataframe after performing PRF, used kmeans to find nearest doc embeds
    # print(len(topics_nfcorpus))
    res = PRFTgt(topics_nfcorpus) # result of PRFtarget are stored in res, we have ["qid", "query", "query_embs", "query_toks", "query_weights"]
    # Save the COlBERT PRF embeddings on target corpus to pickle 
    pd.to_pickle(res,"./tgt/Tgt_BEIR.nfcorpus.embs.pkl")

    #  ColBERT-PRF on External Corpus
    pytcolbert = pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", \
                                                        "./nfs/indices/colbert_passage", \
                                                        "external_scifact",faiss_partitions=100,memtype='mem')
    pytcolbert.faiss_index_on_gpu = False
    dense_e2e = pytcolbert.set_retrieve() >> pytcolbert.index_scorer()
    PRF_ext = dense_e2e %10 >> ColbertPRF(pytcolbert, k=24, fb_docs=10, fb_embs=10, beta=4)
    res = PRF_ext(topics_nfcorpus)
    # Save the COlBERT PRF embeddings on external corpus to pickle 
    pd.to_pickle(res,"./ext/External_BEIR.nfscorpus.embs.pkl")


    # External Dense (ColBERTT-PRF) Expansion
    """
    #########################################
    #########################################
    #########################################
    """
    path_to_target_corpus_embds = "./tgt/Tgt_BEIR.nfcorpus.embs.pkl"
    path_to_external_corpus_embds = "./ext/External_BEIR.nfscorpus.embs.pkl"
    res_tgt = pd.read_pickle(path_to_target_corpus_embds)
    res_tgtTrans = pt.transformer.SourceTransformer(res_tgt)
    res_ext = pd.read_pickle(path_to_external_corpus_embds)
    res_extTrans = pt.transformer.SourceTransformer(res_ext)

    resMerge = mergePRF(res_tgt, res_ext) 
    MergeTrans = pt.transformer.SourceTransformer(resMerge)
    pipeCE2_ranker = (MergeTrans 
                        >> pytcolbertTgt.set_retrieve(query_encoded=True) 
                        >> (pytcolbertTgt.index_scorer(query_encoded=True, add_ranks=True, batch_size=5000)
                        >> pt.text.max_passage() % 1000))
    res = pd.concat([resPart for resPart in pipeCE2_ranker.transform_gen(topics_nfcorpus, batch_size=100)])
    pt.io.write_results(res, "./results/CE.nfcorpus.res.gz")
    evalMeasuresDict = pt.Utils.evaluate(res,qrels_nfcorpus,metrics=["ndcg_cut_10","ndcg_cut_20","ndcg_cut_1000", "map"])
    print(evalMeasuresDict)
    return 




def test():
    if not pt.started():
        pt.init()
    topics_nfcorpus=pt.get_dataset("irds:beir/nfcorpus/test").get_topics()
    pytcolbertTgt = pyterrier_colbert.ranking.ColBERTFactory("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", \
                                                        "./nfs/indices/BEIR/nfcorpus", \
                                                        "nfcorpus_colbertIndex",faiss_partitions=100,memtype='mem')
    pytcolbertTgt.faiss_index_on_gpu = False
    dense_e2eTgt = pytcolbertTgt.set_retrieve() >> pytcolbertTgt.index_scorer(query_encoded=True)
    PRFTgt = dense_e2eTgt %10 >> ColbertPRF(pytcolbertTgt, k=24, fb_docs=10, fb_embs=10, beta=2)
    res = PRFTgt(topics_nfcorpus)
    print("hiiii")
    print(res.head(10))

if __name__ == "__main__":
    test()
    # main()