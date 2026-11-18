
import matplotlib.pyplot as plt
import numpy as np

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize, buildw2i
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    ks = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accs = []
    f1s = []

    for k in ks:
        print(f"Training NB with k={k}...")

        model = NaiveBayes.train(train_data, k=k)

        acc = accuracy(model, test_data)
        f1 = f_1(model, test_data)

        accs.append(acc)
        f1s.append(f1)

        print(f"  k={k}: Acc={acc:.4f}, F1={f1:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(ks, accs, 'o-', label='Accuracy')
    plt.plot(ks, f1s, 's-', label='F1-score')
    plt.xscale('log')
    plt.xlabel('Smoothing parameter k')
    plt.ylabel('Score')
    plt.title('Naive Bayes performance vs smoothing k')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('smoothing_k_plot.png', dpi=150)
    plt.close()

    print("Saved plot to smoothing_k_plot.png")
    ####################################################################


def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag-of-words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################

    print("=== Baseline bag-of-words ===")
    nb0 = NaiveBayes.train(train_data, k=1)
    print(f"Acc: {accuracy(nb0, test_data):.4f}, F1: {f_1(nb0, test_data):.4f}")

    print("=== Feature set 1: stopword removal and lowercasing ===")
    nb1 = NaiveBayes(**features1(train_data, k=1))
    print(f"Acc: {accuracy(nb1, test_data):.4f}, F1: {f_1(nb1, test_data):.4f}")

    print("=== Feature set 2: unigrams and bigrams ===")
    nb2 = NaiveBayes(**features2(train_data, k=1))
    print(f"Acc: {accuracy(nb2, test_data):.4f}, F1: {f_1(nb2, test_data):.4f}")
    #####################################################################


def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################

    print("Featurizing training data...")

    # every word of the training set gets an index
    vocab = set()
    for words, label in train_data:
        vocab.update(words)

    w2i = buildw2i(vocab)

    X_train, Y_train = featurize(train_data, w2i=w2i)

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

    print("Training Logistic Regression (10 epochs, C=0.1)...")

    model = LogReg(eta=0.01, num_iter=10, C=0.1)
    model.w2i = w2i
    model.F = len(w2i)

    model.train(X_train, Y_train)

    print("Evaluating on test set...")

    acc = accuracy(model, test_data)
    f1 = f_1(model, test_data)

    print(f"LogReg Accuracy: {acc:.4f}")
    print(f"LogReg F1: {f1:.4f}")
    #####################################################################
