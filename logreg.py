from typing import Iterator, Sequence, Text, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from reader import OffenseEvalData
import argparse


class Classifier(object):
    def __init__(self, data_dir, output_dir):
        """Initializes the classifier."""
        # the classifier does not converge at default max iteration.
        self.clf = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=100)
        self.le = preprocessing.LabelEncoder()
        self.dv = DictVectorizer()
        self.cv = CountVectorizer()
        self.data_dir = data_dir
        self.output_dir = output_dir

    def train(self):
        features = []
        labels = []
        processor = OffenseEvalData()
        corpus = []
        for sentence in processor.get_train_examples(self.data_dir):
            labels.append(sentence.label)
            corpus.append(sentence.text_a)

        X = self.cv.fit_transform(corpus)
        y = self.le.fit_transform(labels)
        # train the log reg classifier
        self.clf.fit(X, y)

    def label_index(self, label: Text) -> int:
        return self.le.transform([label])[0]

    def predict(self):
        processor = OffenseEvalData()
        with open(self.output_dir + "test_submission_logreg.csv") as file:
            for sentence in processor.get_test_examples(self.data_dir):
                X = self.cv.transform(sentence.text_a)
                y = self.clf.predict(X)
                label = self.le.classes_[y]
                file.write("%s,%s\n" % (sentence.guid, label))


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
    classifier.predict()


if __name__ == "__main__":
    run_classifier()
