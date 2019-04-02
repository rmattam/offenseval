from typing import Iterator, Sequence, Text, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from scipy.sparse import vstack

import numpy as np
from scipy.sparse import spmatrix

NDArray = Union[np.ndarray, spmatrix]
TokenSeq = Sequence[Text]
PosSeq = Sequence[Text]


def read_data(path: str) -> Iterator[Tuple[TokenSeq, PosSeq]]:
    """Reads sentences from a Penn TreeBank .tagged file.
    Each sentence is a sequence of tokens and part-of-speech tags.

    Penn TreeBank .tagged files contain one token per line, with an empty line
    marking the end of each sentence. Each line is composed of a token, a tab
    character, and a part-of-speech tag. Here is an example:

        What	WP
        's	VBZ
        next	JJ
        ?	.

        Slides	NNS
        to	TO
        illustrate	VB
        Shostakovich	NNP
        quartets	NNS
        ?	.

    :param ptbtagged_path: The path of a Penn TreeBank .tagged file, formatted
    as above.
    :return: An iterator over sentences, where each sentence is a tuple of
    a sequence of tokens and a corresponding sequence of part-of-speech tags.
    """
    token_seq = []
    pos_seq = []
    with open(path, "r") as file:
        for line in file:
            if line.isspace():
                yield token_seq, pos_seq
                token_seq = []
                pos_seq = []
            else:
                tokens = line.split('\t')
                token_seq.append(tokens[0])
                # slicing up to -1, ignoring the last new line character.
                pos_seq.append(tokens[1][:-1])

    # return the last token and pos sentence tuple from the file.
    yield token_seq, pos_seq


class Classifier(object):
    def __init__(self):
        """Initializes the classifier."""
        # the classifier does not converge at default max iteration.
        self.clf = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=100)
        self.le = preprocessing.LabelEncoder()
        self.dv = DictVectorizer()

    def train(self, tagged_sentences: Iterator[Tuple[TokenSeq, PosSeq]]) -> Tuple[NDArray, NDArray]:
        """Trains the classifier on the part-of-speech tagged sentences,
        and returns the feature matrix and label vector on which it was trained.

        The feature matrix should have one row per training token. The number
        of columns is up to the implementation, but there must at least be 1
        feature for each token, named "token=T", where "T" is the token string,
        and one feature for the part-of-speech tag of the preceding token,
        named "pos-1=P", where "P" is the part-of-speech tag string, or "<s>" if
        the token was the first in the sentence. For example, if the input is:

            What	WP
            's	VBZ
            next	JJ
            ?	.

        Then the first row in the feature matrix should have features for
        "token=What" and "pos-1=<s>", the second row in the feature matrix
        should have features for "token='s" and "pos-1=WP", etc. The alignment
        between these feature names and the integer columns of the feature
        matrix is given by the `feature_index` method below.

        The label vector should have one entry per training token, and each
        entry should be an integer. The alignment between part-of-speech tag
        strings and the integers in the label vector is given by the
        `label_index` method below.

        :param tagged_sentences: An iterator over sentences, where each sentence
        is a tuple of a sequence of tokens and a corresponding sequence of
        part-of-speech tags.
        :return: A tuple of (feature-matrix, label-vector).
        """
        features = []
        labels = []
        for sentence in tagged_sentences:
            for i in range(0, len(sentence[0])):
                # utilize this toke iteration to create a labels list from training data
                labels.append(sentence[1][i])
                # create feature mapping
                features.append({
                    'token': sentence[0][i],
                    'token-1': sentence[0][i-1] if i > 1 else '<s>',
                    'pos-1': sentence[1][i-1] if i > 1 else '<s>'
                })

        X = self.dv.fit_transform(features)
        y = self.le.fit_transform(labels)
        # train the log reg classifier
        self.clf.fit(X, y)
        return X, y

    def feature_index(self, feature: Text) -> int:
        """Returns the column index corresponding to the given named feature.

        The `train` method should always be called before this method is called.

        :param feature: The string name of a feature.
        :return: The column index of the feature in the feature matrix returned
        by the `train` method.
        """
        return self.dv.vocabulary_[feature]

    def label_index(self, label: Text) -> int:
        """Returns the integer corresponding to the given part-of-speech tag

        The `train` method should always be called before this method is called.

        :param label: The part-of-speech tag string.
        :return: The integer for the part-of-speech tag, to be used in the label
        vector returned by the `train` method.
        """
        return self.le.transform([label])[0]

    def predict(self, tokens: TokenSeq) -> PosSeq:
        """Predicts part-of-speech tags for the sequence of tokens.

        This method delegates to either `predict_greedy` or `predict_viterbi`.
        The implementer may decide which one to delegate to.

        :param tokens: A sequence of tokens representing a sentence.
        :return: A sequence of part-of-speech tags, one for each token.
        """
        _, pos_tags = self.predict_greedy(tokens)
        # _, _, pos_tags = self.predict_viterbi(tokens)
        return pos_tags