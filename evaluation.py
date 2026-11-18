
def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################

    if not data:
        return 0.0

    correct = 0

    for x, y in data:
        prediction = classifier.predict(x)

        if prediction == y:
            correct += 1

    return correct / len(data)

    ################################################################


def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################

    tp = 0
    fp = 0
    fn = 0

    for x, y in data:
        prediction = classifier.predict(x)

        if prediction == 'offensive' and y == 'offensive':
            tp += 1
        elif prediction == 'offensive' and y == 'nonoffensive':
            fp += 1
        elif prediction == 'nonoffensive' and y == 'offensive':
            fn += 1

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

    ################################################################
