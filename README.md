# offenseval
Submission for SemEval 2019 Task: OffensEval

## Run Bert

To train the bert model run the following command from the source root directory.

```
python bert.py --data_dir data/ --output_dir output/ --do_train --do_lower_case --bert_model bert-base-uncased
```

To generate the test submission file by running the bert model on test set run the following command from the source root directory.

```
python bert.py --data_dir data/ --output_dir output/ --do_test --do_lower_case --bert_model bert-base-uncased
```

### Best Performance Observed

The best performance was observed by Bert and an ensemble model of Bert with two versions of a logistic regression classifier using character and word ngrams.

Trained BERT for 3 epochs, batch size = 1:
(The model predicted NOT class for all answers)

```
acc = 0.7209302325581395
f1_macor = 0.4189189189189189
f1_micro = 0.7209302325581395
```
Trained BERT again for 10 epochs, batch size = 2:

```
acc = 0.8023255813953488
f1_macor = 0.7304890278433223
f1_micro = 0.8023255813953488


##### Data source

dev.txt is from /offenseval-data/start-kit/trail-data
train.txt is from /offenseval-data/OLIDv1.0/olid-training-v1.0
test.txt is from /offenseval-data/OLIDv1.0/testset-levela.tsv and labels-levela.csv

