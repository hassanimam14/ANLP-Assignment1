import numpy as np


class LogReg:
    def __init__(self, eta=0.01, num_iter=30, C=0.1, seed=0):
        self.eta = eta
        self.seed = seed
        self.num_iter = num_iter
        self.C = C
        self.weights = None
        self.classes = ['offensive', 'nonoffensive']
        self.w2i = {}
        self.F = 0

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs: ... x K
        :return: same shape probabilities
        """
        # stability fix
        tmp = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_inputs = np.exp(tmp)
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

    def train(self, X, Y):
        #################### STUDENT SOLUTION ###################
        N, F = X.shape
        K = Y.shape[1]
        self.weights = np.zeros((F, K))
        batch_size = 100
        rng = np.random.RandomState(self.seed)

        for epoch in range(self.num_iter):
            indices = rng.permutation(N)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = indices[start:end]
                X_batch = X[batch]
                Y_batch = Y[batch]
                m = len(X_batch)
                scores = X_batch @ self.weights
                probabilities = self.softmax(scores)
                error = probabilities - Y_batch
                gradient = X_batch.T @ error
                gradient += self.C * self.weights
                self.weights -= self.eta * gradient

        return self
        #########################################################

    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        scores = X @ self.weights
        return self.softmax(scores)
        ############################################################

    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        if isinstance(X, list):
            x_vector = np.zeros((1, self.F))
            for word in set(X):
                if word in self.w2i:
                    x_vector[0, self.w2i[word]] = 1
            probabilities = self.p(x_vector)
            return self.classes[np.argmax(probabilities[0])]

        probabilities = self.p(X)
        predictions = np.argmax(probabilities, axis=1)
        return [self.classes[i] for i in predictions]
        #############################################################


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################
    w2i = {}
    for i, w in enumerate(sorted(vocab)):
        w2i[w] = i
    return w2i
    ############################################################


def featurize(data, train_data=None, w2i=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.
        w2i: optional prebuilt word to index

    Returns:
        Matrix X and Y.
    """
    # YOUR CODE HERE
    ##################### STUDENT SOLUTION #######################
    if w2i is None:
        if train_data is None:
            train_data = data
        vocab = set()
        for words, label in train_data:
            vocab.update(words)
        w2i = buildw2i(vocab)

    N = len(data)
    F = len(w2i)
    X = np.zeros((N, F), dtype=np.float32)
    Y = np.zeros((N, 2))

    for i, (words, label) in enumerate(data):
        for word in set(words):
            if word in w2i:
                X[i, w2i[word]] = 1
        if label == 'offensive':
            Y[i, 0] = 1
        else:
            Y[i, 1] = 1

    return X, Y
    ##############################################################