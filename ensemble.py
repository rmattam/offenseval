from logreg import Classifier
from reader import OffenseEvalData
from sklearn.metrics import f1_score, accuracy_score
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

    args = parser.parse_args()

    # train the classifier on the training data
    classifier = Classifier(args.data_dir, args.output_dir)
    classifier.train()
    processor = OffenseEvalData()
    devel_indices = []
    predicted_indices = []
    with open(args.output_dir + "test_submission_logreg.csv", "w") as file:
        for sentence in processor.get_test_examples(args.data_dir):
            char_pred = classifier.predict_one_char(sentence.text_a)
            word_pred = classifier.predict_one_word(sentence.text_a)

            selected_label = word_pred

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
