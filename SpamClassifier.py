import random
import string

import NaiveBayes

# load dataset from file
def load_dataset(filename, train_size=0.75):
    train_ham = []
    test_ham = []
    train_spam = []
    test_spam = []

    # need in order to remove puncuation
    table = str.maketrans(dict.fromkeys(string.punctuation))

    with open(filename) as fp:
        lines = fp.readlines()

    # don't shuffle is entire dataset is being used to build model
    if train_size != 1:
        random.shuffle(lines)

    # the number of items in the training set
    num_lines = int(len(lines) * train_size)

    for i, line in enumerate(lines):
        # remove puncuation from line and get it's label
        label, cleaned_line = line.split(maxsplit=1)
        cleaned_line = cleaned_line.strip().translate(table).lower()

        # add lines to relevant lists
        if label == 'ham' and len(cleaned_line) > 0:
            if i < num_lines:
                train_ham.append(cleaned_line)
            else: 
                test_ham.append(cleaned_line)
        elif label == 'spam' and len(cleaned_line) > 0:
            if i < num_lines:
                train_spam.append(cleaned_line)
            else: 
                test_spam.append(cleaned_line)

    training_dict = {
        'ham': NaiveBayes.ItemSet('ham', train_ham),
        'spam': NaiveBayes.ItemSet('spam', train_spam)
    }

    testing_dict = {
        'ham': NaiveBayes.ItemSet('ham', test_ham),
        'spam': NaiveBayes.ItemSet('spam', test_spam)
    }

    return training_dict, testing_dict

# calculate model metrics
def test_model(naive_bayes, testing_dict):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for sentence in testing_dict['spam'].items:
        predicted, _ = naive_bayes.classify(sentence)
        if 'spam' == predicted:
            true_positive += 1
        else:
            false_negative += 1

    for sentence in testing_dict['ham'].items:
        predicted, _ = naive_bayes.classify(sentence)
        if 'ham' == predicted:
            true_negative += 1
        else:
            false_positive += 1

    precision = true_positive / (true_positive + false_positive)
    recall =    true_positive / (true_positive + false_negative)
    fscore = (2 * precision * recall) / (precision + recall)
    accuracy = (true_positive + true_negative) / (len(testing_dict['ham'].items) + len(testing_dict['spam'].items))

    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F-Score: {}'.format(fscore))
    print('True Positive: {}'.format(true_positive))
    print('False Positive: {}'.format(false_positive))
    print('True Negative: {}'.format(true_negative))
    print('False Negative: {}'.format(false_negative))

# ask for user input to classify
def prompt_input(naive_bayes):
    while True:
        user_input = input("SMS to test: ")

        # exit when user only enters 'q'
        if user_input == 'q':
            break

        # predict input
        label, probability = naive_bayes.classify(user_input)
        print('{} {}\n'.format(label, probability))

if __name__ == '__main__': 
    # load the dataset
    training_dict, testing_dict = load_dataset('SMSSpamDataset')

    # create the model
    naive_bayes = NaiveBayes.NaiveBayes(training_dict)

    # use the model
    test_model(naive_bayes, testing_dict)
    # prompt_input(naive_bayes)

    
