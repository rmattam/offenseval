from typing import Iterator, Sequence, Text, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing


class Datapoint:
    def __init__(self, id, text, category):
        self.id = id
        self.text = text
        self.category = category


def read_data(path: str) -> Iterator[Datapoint]:
    with open(path, "r") as file:
        for line in file:
            tokens = line.split('\t')
            if tokens[0] == "id":
                continue
            dp = Datapoint(tokens[0], tokens[1], tokens[2])
            yield dp


class Classifier(object):
    def __init__(self):
        """Initializes the classifier."""
        # the classifier does not converge at default max iteration.
        self.clf = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=100)
        self.le = preprocessing.LabelEncoder()
        self.dv = DictVectorizer()

    # def train(self, tagged_sentences: Iterator[Tuple[TokenSeq, PosSeq]]) -> Tuple[NDArray, NDArray]:
    #     features = []
    #     labels = []
    #     for sentence in tagged_sentences:
    #         for i in range(0, len(sentence[0])):
    #             # utilize this toke iteration to create a labels list from training data
    #             labels.append(sentence[1][i])
    #             # create feature mapping
    #             features.append({
    #                 'token': sentence[0][i],
    #                 'token-1': sentence[0][i-1] if i > 1 else '<s>',
    #                 'pos-1': sentence[1][i-1] if i > 1 else '<s>'
    #             })
    #
    #     X = self.dv.fit_transform(features)
    #     y = self.le.fit_transform(labels)
    #     # train the log reg classifier
    #     self.clf.fit(X, y)
    #     return X, y
    #
    # def feature_index(self, feature: Text) -> int:
    #     return self.dv.vocabulary_[feature]
    #
    # def label_index(self, label: Text) -> int:
    #     return self.le.transform([label])[0]
    #
    # def predict(self, tokens: TokenSeq) -> PosSeq:
    #     _, pos_tags = self.predict_greedy(tokens)
    #     # _, _, pos_tags = self.predict_viterbi(tokens)
    #     return pos_tags
