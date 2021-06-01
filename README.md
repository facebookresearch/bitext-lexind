# Bilingual Lexicon Inductionvia Unsupervised Bitext Construction and Word Alignment 

[Haoyue Shi](https://ttic.uchicago.edu/~freda), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz) and [Sida I. Wang](http://www.sidaw.xyz/)

## Requirements
PyTorch >= 1.7 <br> 
transformers == 4.0.0 <br>
fairseq (to run CRISS and extract CRISS-based features) <br>
chinese_converter (to convert between simplfied and traditional Chinese, fitting the different settings of CRISS and [MUSE](https://github.com/facebookresearch/MUSE)) <br>

See also [env/env.yml](./env/env.yml) for sufficient environment setup. 

## A Quick Example for the Pipeline of Lexicon Induction 

### Step 0: Download [CRISS](https://github.com/pytorch/fairseq/tree/master/examples/criss) 
The default setting assumes that the CRISS (3rd iteration) model is saved in `criss/criss-3rd.pt`. 

### Step 1: Unsupervised Bitext Construction with [CRISS](https://github.com/pytorch/fairseq/tree/master/examples/criss)
Let's assume that we have the following [bitext](./data/bitext.txt) (sentences separated by " ||| ", one pair per line):
```
Das ist eine Katze . ||| This is a cat .
Das ist ein Hund . ||| This is a dog .
```

### Step 2: Word Alignment with [SimAlign](https://github.com/cisnlp/simalign) 
Note: we use CRISS as the backbone of SimAlign and use [our own implmentation](./align/), you can also use other aligners---just make sure that the results are stored in [a json file](./data/bitext.txt.align) like follows:
```
{"inter": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], "itermax": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}
{"inter": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], "itermax": [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]}
```
where "inter" and "itermax" denote the *argmax* and *itermax* algorithm in SimAlign respectively. 
The output is in the same format as the json output of [SimAlign](https://github.com/cisnlp/simalign). 
See the code of SimAlign for more details. 

### Step 3: Training and Testing Lexicon Inducer 
#### Fully Unsupervised 
```
python src/fully_unsup.py \
    -b ./data/bitext.txt \
    -a ./data/bitext.txt.align \
    -te ./data/test.dict 
```

#### Weakly Supervised 
```
python src/weakly_sup.py \
    -b ./data/bitext.txt \
    -a ./data/bitext.txt.align \
    -tr ./data/train.dict \
    -te ./data/test.dict \
    -src de_DE \
    -trg en_XX
```

You would probably also like to specify a model folder by `-o $model_FOLDER` to save the statistices of bitext and alignment (default `./model`). 

`-src` and `-trg` specify the source and target language, where for the languages and corresponding codes that CRISS supports, check the language pairs in [this file](https://github.com/pytorch/fairseq/blob/master/examples/criss/unsupervised_mt/eval.sh). 

You will see the final model (`model.pt`, lexicon inducer) and the induced lexicon (`induced.weaklysup.dict`/`induced.fullyunsup.dict`) in the model folder, as well as a line of evaluation result (on the test set) like follows:
```
{'oov_number': 0, 'oov_rate': 0.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
```

## A Quick Example for the MLP-Based Aligner

#### Training 
Training an MLP-based aligner using the bitext and alignment shown above. 
```
python align/train.py \
    -b ./data/bitext.txt \
    -a ./data/bitext.txt.align \
    -src de_DE \
    -trg en_XX \
    -o model/
```

#### Testing
Testing the saved aligner on the same set (note: this is only used to show how the code works, and in real scenarios we test on a different dataset from the training set). 

The `-b` and `-a` should be the same as those used for training, to avoid potential error (in fact, if you did not delete anything after training, the `-b` and `-a` parameters will never be actually used). 
```
python align/test.py \
    -b ./data/bitext.txt \
    -a ./data/bitext.txt.align \
    -src de_DE \
    -trg en_XX \
    -m model/
```


For CRISS-SimAlign baseline, you can run a quick evaluation of CRISS-based SimAlign the above examples for German--English alignment, using the *argmax* inference algorithm 
```
python align/eval_simalign_criss.py
```

## License
MIT 