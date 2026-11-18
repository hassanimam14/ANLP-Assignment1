
from collections import defaultdict
import math


STOPWORDS = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'can', 'need', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then'
])


class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    def __init__(self, classes=None, class_priors=None, word_probs=None, vocab=None, k=1, feature_type=0):
        """Initialises a new classifier."""
        self.classes = classes or []
        self.class_priors = class_priors or {}
        self.word_probs = word_probs or {}
        self.vocab = vocab or set()
        self.k = k
        self.feature_type = feature_type
    ####################################################################


    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################

        if not self.classes:
            return None

        if self.feature_type == 1:

            words = [
                w.lower() for w in x
                if w.lower() not in STOPWORDS and w.isalpha()
            ]

        elif self.feature_type == 2:
            words = list(x)

            for i in range(len(x) - 1):
                words.append(x[i] + '_' + x[i + 1])

        else:
            words = x

        best_class = None
        best_score = float('-inf')

        for c in self.classes:
            score = self.class_priors[c]

            for w in words:
                if w in self.vocab:
                    score += self.word_probs[c][w]

            if score > best_score:
                best_score = score
                best_class = c

        return best_class

        ############################################################


    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """
        ##################### STUDENT SOLUTION #####################

        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))
        vocab = set()

        total_docs = len(data)

        for words, label in data:
            class_counts[label] += 1

            for word in words:
                word_counts[label][word] += 1
                vocab.add(word)

        classes = list(class_counts.keys())
        V = len(vocab)

        class_priors = {}

        for c in classes:
            class_priors[c] = math.log(class_counts[c] / total_docs)

        word_probs = {}

        for c in classes:
            total_words = sum(word_counts[c].values())
            denominator = total_words + k * V

            word_probs[c] = {}

            for word in vocab:
                numerator = word_counts[c][word] + k
                word_probs[c][word] = math.log(numerator / denominator)

        return cls(
            classes=classes,
            class_priors=class_priors,
            word_probs=word_probs,
            vocab=vocab,
            k=k
        )

        ############################################################



def features1(data, k=1):
    """
    Feature set 1: lowercasing, stop word removal, alphabetic tokens only.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################

    processed = []

    for words, label in data:
        new_words = [
            w.lower() for w in words
            if w.lower() not in STOPWORDS and w.isalpha()
        ]

        processed.append((new_words, label))

    model = NaiveBayes.train(processed, k=k)
    return dict(
        classes=model.classes,
        class_priors=model.class_priors,
        word_probs=model.word_probs,
        vocab=model.vocab,
        k=k,
        feature_type=1
    )

    ##################################################################


def features2(data, k=1):
    """
    Feature set 2: unigrams and bigrams (joined with an underscore).

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################

    processed = []

    for words, label in data:
        new_words = list(words)

        for i in range(len(words) - 1):
            new_words.append(words[i] + '_' + words[i + 1])

        processed.append((new_words, label))

    model = NaiveBayes.train(processed, k=k)
    return dict(
        classes=model.classes,
        class_priors=model.class_priors,
        word_probs=model.word_probs,
        vocab=model.vocab,
        k=k,
        feature_type=2
    )

    ##################################################################
