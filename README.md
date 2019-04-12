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

### Performance Observed

Trained BERT for 3 epochs, batch size = 1:
(The model predicted NOT class for all answers)

```

04/11/2019 17:05:20 - INFO - __main__ -   ***** Eval results *****
04/11/2019 17:05:20 - INFO - __main__ -     acc = 0.7209302325581395
04/11/2019 17:05:20 - INFO - __main__ -     eval_loss = 1.028112765282479
04/11/2019 17:05:20 - INFO - __main__ -     f1_macor = 0.4189189189189189
04/11/2019 17:05:20 - INFO - __main__ -     f1_micro = 0.7209302325581395
04/11/2019 17:05:20 - INFO - __main__ -     global_step = 0
04/11/2019 17:05:20 - INFO - __main__ -     loss = None
04/11/2019 17:05:20 - INFO - __main__ -   ***** Test submission file *****
```
Trained BERT again for 10 epochs, batch size = 2:

```
04/11/2019 23:23:22 - INFO - __main__ -   ***** Eval results *****
04/11/2019 23:23:22 - INFO - __main__ -     acc = 0.8023255813953488
04/11/2019 23:23:22 - INFO - __main__ -     eval_loss = 1.436534363466604
04/11/2019 23:23:22 - INFO - __main__ -     f1_macor = 0.7304890278433223
04/11/2019 23:23:22 - INFO - __main__ -     f1_micro = 0.8023255813953488
04/11/2019 23:23:22 - INFO - __main__ -     global_step = 0
04/11/2019 23:23:22 - INFO - __main__ -     loss = None
04/11/2019 23:23:22 - INFO - __main__ -   ***** Test submission file *****
```




Ideas:
* Read test data and create a dummy submission csv

* idea, use POS tagger to get all adjectives in offensive sentences
* use these adjectives as offensive features

Tasks:
* create a data reader

##### Data source

dev.txt is from /offenseval-data/start-kit/trail-data
train.txt is from /offenseval-data/OLIDv1.0/olid-training-v1.0
test.txt is from /offenseval-data/OLIDv1.0/testset-levela.tsv and labels-levela.csv

