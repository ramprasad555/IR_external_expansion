
This is the README for the project Report based on the IPM paper: "Improving zero-shot retrieval using dense external expansion".(https://www.sciencedirect.com/science/article/pii/S0306457322001364).

For the original README provided by the authors of the paper, please refer to - 
https://github.com/Xiao0728/DenseExternalExpansion_VirtualAppendix

# Clone the git repository of our code using.
using the command `git clone https://github.com/ramprasad555/IR_external_expansion.git`

# Install the dependencies
- pyterrier
- faiss
- torch
- pyterrier_colbert
- pyterrier_colbert.ranking
- pyterrier_colbert.indexing
- pandas
- os

# Download the MS MARCO dataset
Download `collection.tar.gz` from the MS MARCO passage ranking repository. Using the Command 
```
`wget -c --retry-connrefused --tries=0 --timeout=50 https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz⁠`
```
Unzip `collection.tar.gz` using the command `tar -xvf collection.tar.gz`

Now move the collection.tsv file into the folder `./data/collection`

# sample the MSMARCO dataset. 
we used msmarco_passage_10k.tsv as file name and the sample size is 10000
```
python3 subsample.py ./data/collection/collection.tsv <output filename> <sample size>
```

## indexing the Target Corpus
If the indices fo target and external corpus are not available offline, run the following command
```
python3 indexing.py
```
The above command indexes both target and external corpus separately and stores in the folder ``` ./index/tgt ``` and ```./index/ext``` for target and external corpus respectively.

By now you will have one index each for target and external corpus.

## Expansion, Inference and Evaluation
The file project.py contains the code for performing PRF with specified independent variables(number of feedback docs in PRF and Beta values) on target and external corpuses separately and stores the embeddings in pickle files at ``` ./embeddings ``` folder. Next The (1)query embeddings generated from the target corpus and the (2)external corpus are concatenated into a new set of (3)query embeddings. This new set is then used as input for the retrieval pipeline. The results of this pipeline are saved as pickles to ``` ./final_result``` folder.

Later, the results of the pipeline are used to calculate performance metrics like 
- nDCG - Normalised Discounted Cummulative Gain
- MAP - Mean Average Precision 

Run the following command for Expansion, Inference and Evaluation

``` python3 Project.py --beta 0 0.5 1 --fb_docs 3 5 10 15 20 25 30 --cutoff 40 ```


The evaluation results from the above run are stored in ``` output.txt ```

--- we have uploaded few results from our previous runs.