from typing import Iterator, Sequence, Text, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from reader import OffenseEvalData
from sklearn.metrics import f1_score, accuracy_score
import argparse
import re


def words_and_char_bigrams(text):
    words = re.findall(r'\w{3,}', text)
    n = 7
    for w in words:
        yield w
        for i in range(len(w) - n + 1):
            yield w[i:i + n]


class Classifier(object):
    def __init__(self, data_dir, output_dir):
        """Initializes the classifier."""
        # the classifier does not converge at default max iteration.
        self.clf_char = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=100)
        self.clf_word = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=100)
        self.le = preprocessing.LabelEncoder()
        self.cv_char = CountVectorizer(ngram_range=(2, 7), analyzer="char_wb")
        self.cv_word = CountVectorizer(ngram_range=(1, 7))

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

        X = self.cv_char.fit_transform(corpus)
        y = self.le.fit_transform(labels)
        # train the character ngram log reg classifier
        self.clf_char.fit(X, y)
        X = self.cv_word.fit_transform(corpus)
        # train the word ngram log reg classifier
        self.clf_word.fit(X, y)

    def label_index(self, label: Text) -> int:
        return self.le.transform([label])[0]

    def predict_one_char(self, test_input):
        X = self.cv_char.transform([test_input])
        y = self.clf_char.predict(X)
        return self.le.classes_[y[0]]

    def predict_char(self):
        processor = OffenseEvalData()
        devel_indices = []
        predicted_indices = []
        with open(self.output_dir + "test_submission_logreg_char.csv", "w") as file:
            for sentence in processor.get_test_examples(self.data_dir):
                X = self.cv_char.transform([sentence.text_a])
                y = self.clf_char.predict(X)

                devel_indices.append(self.label_index(sentence.label))
                predicted_indices.append(y[0])

                label = self.le.classes_[y[0]]
                file.write("%s,%s\n" % (sentence.guid, label))

        f1_macro = f1_score(devel_indices, predicted_indices, average="macro")
        f1_micro = f1_score(devel_indices, predicted_indices, average="micro")
        accuracy = accuracy_score(devel_indices, predicted_indices)

        # print out performance
        msg = "\n{:.1%} F1 macro {:.1%} F1 micro and {:.1%} accuracy on test data"
        print(msg.format(f1_macro, f1_micro, accuracy))

    def predict_one_word(self, test_input):
        X = self.cv_word.transform([test_input])
        y = self.clf_word.predict(X)
        return self.le.classes_[y[0]]

    def predict_word(self):
        processor = OffenseEvalData()
        devel_indices = []
        predicted_indices = []
        with open(self.output_dir + "test_submission_logreg_word.csv", "w") as file:
            for sentence in processor.get_test_examples(self.data_dir):
                X = self.cv_word.transform([sentence.text_a])
                y = self.clf_word.predict(X)

                devel_indices.append(self.label_index(sentence.label))
                predicted_indices.append(y[0])

                label = self.le.classes_[y[0]]
                file.write("%s,%s\n" % (sentence.guid, label))

        f1_macro = f1_score(devel_indices, predicted_indices, average="macro")
        f1_micro = f1_score(devel_indices, predicted_indices, average="micro")
        accuracy = accuracy_score(devel_indices, predicted_indices)

        # print out performance
        msg = "\n{:.1%} F1 macro {:.1%} F1 micro and {:.1%} accuracy on test data"
        print(msg.format(f1_macro, f1_micro, accuracy))


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
    classifier.predict_char()
    classifier.predict_word()


if __name__ == "__main__":
    run_classifier()
