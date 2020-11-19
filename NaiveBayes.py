import string
import math

class ItemSet:
    def __init__(self, label, items):
        self.label = label
        self.items = items

class CountSet:
    def __init__(self, label, items, total, probability=None):
        self.label = label
        self.items = items
        self.total = total
        self.probability = probability

class NaiveBayes:
    def __init__(self, count_set):
        self.count_set = count_set
        self.count_dict = {}

        self.count()

    # Count the occurances of a word
    def count(self):
        dataset_total = 0

        # for each class
        for _, current_class in self.count_set.items():
            label_counts = {}
            label_total = 0

            # for each sentence
            for line in current_class.items:

                # for each word
                for word in line.split():
                    label_counts.setdefault(word, 0)
                    label_counts[word] += 1

            # number of words for each label
            label_total = sum(label_counts.values())
            dataset_total += label_total

            self.count_dict[current_class.label] = CountSet(current_class.label, label_counts, label_total)

        # calculate the probability of each class
        for _, current_class in self.count_dict.items():
            current_class.probability = current_class.total / dataset_total

    # classify a sentence
    def classify(self, sentence):
        prob_dict = {}

        # remove puncuation and split sentence into a list
        table = sentence.maketrans(dict.fromkeys(string.punctuation))
        words = sentence.strip().translate(table).lower().split()

        # for Laplacian correction
        correction = 0
        for _, current_class in self.count_dict.items():
            # count the number of unique words the model has not seen to prevent double counting
            correction += sum([ (0 if word in current_class.items else 1) for word in set(words) ])

            # since each word is its own case, count the number of unique words
            correction += len(current_class.items)

        # do naive bayes
        for _, current_class in self.count_dict.items():
            # use log to avoid underflow
            probability = math.log(current_class.probability)

            for word in words:
                # Calculate the probability of a word by dividing the number of occurances for the word by class size.
                # Default word count to zero if it is new
                # Apply Laplacian correction by adding one to word count 
                # and the number of unique words in the sentence to the class size since each word is its own case.
                # Log to avoid underflow; this also changes the formula to a summation
                probability += math.log((current_class.items.get(word, 0) + 1) / (current_class.total + correction))

            prob_dict[current_class.label] = probability

        # return the classified class and the probabilities
        return max(prob_dict, key=lambda label: prob_dict[label]), prob_dict
