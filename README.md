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


Ideas:
* Read test data and create a dummy submission csv

* idea, use POS tagger to get all adjectives in offensive sentences
* use these adjectives as offensive features

Tasks:
* create a data reader

##### Data source

dev.txt is from /offenseval-data/start-kit/trail-data
train.txt is from /offenseval-data/start-kit/training-v1
test.txt is from /offenseval-data/Test A Release/

