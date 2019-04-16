from logreg import Classifier
from reader import OffenseEvalData
from bertpredict import BertPredict
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
import argparse


def run_classifier():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Bert Required parameters
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    # train the classifier on the training data
    classifier = Classifier(args.data_dir, args.output_dir)
    bert = BertPredict(args)
    classifier.train()
    processor = OffenseEvalData()
    devel_indices = []
    predicted_indices = []
    with open(args.output_dir + "test_submission_logreg.csv", "w") as file:
        for sentence in processor.get_test_examples(args.data_dir):
            char_pred = classifier.predict_one_char(sentence.text_a)
            word_pred = classifier.predict_one_word(sentence.text_a)
            bert_pred = bert.predict_one(sentence)
            selected_label, _ = Counter([char_pred, word_pred, bert_pred]).most_common()[0]

            devel_indices.append(classifier.label_index(sentence.label))
            predicted_indices.append(classifier.label_index(selected_label))

            file.write("%s,%s\n" % (sentence.guid, selected_label))

    f1_macro = f1_score(devel_indices, predicted_indices, average="macro")
    f1_micro = f1_score(devel_indices, predicted_indices, average="micro")
    accuracy = accuracy_score(devel_indices, predicted_indices)

    # print out performance
    msg = "\n{:.1%} F1 macro {:.1%} F1 micro and {:.1%} accuracy on test data"
    print(msg.format(f1_macro, f1_micro, accuracy))


if __name__ == "__main__":
    run_classifier()

# usage
# python ensemble.py --data_dir data/ --output_dir output/ --bert_model_dir bert_trained_model/ --do_lower_case --bert_model bert-base-uncased