import numpy as np
from sklearn.metrics import balanced_accuracy_score

class MetricsCalculator:
    def __init__(self, classes):
        self.predictions = []
        self.labels = []
        self.classes = len(classes)

    def add_result(self, prediction, label):
        self.predictions.append(prediction)
        self.labels.append(label)

    def calculate_balanced_accuracy(self):
        return balanced_accuracy_score(self.labels, self.predictions)

    def calculate_g_mean(self):
        num_classes = self.classes
        class_counts = np.zeros(num_classes)
        class_correct_counts = np.zeros(num_classes)

        for pred, true_label in zip(self.predictions, self.labels):
            class_counts[true_label] += 1
            if pred == true_label:
                class_correct_counts[true_label] += 1

        class_accs = np.where(class_counts == 0, 1, class_correct_counts / class_counts)
        return np.sqrt(np.prod(class_accs))

    def calculate_f1_score(self):
        num_classes = self.classes
        class_counts = np.zeros(num_classes)
        true_positive_counts = np.zeros(num_classes)
        false_positive_counts = np.zeros(num_classes)
        false_negative_counts = np.zeros(num_classes)

        for pred, true_label in zip(self.predictions, self.labels):
            pred = int(pred)
            class_counts[true_label] += 1
            if pred == true_label:
                true_positive_counts[true_label] += 1
            else:
                false_positive_counts[pred] += 1
                false_negative_counts[true_label] += 1

        class_precisions = np.where(class_counts == 0, 0,
                                    true_positive_counts / (true_positive_counts + false_positive_counts))
        class_recalls = np.where(class_counts == 0, 0,
                                 true_positive_counts / (true_positive_counts + false_negative_counts))

        class_f1_scores = np.where((class_precisions == 0) | (class_recalls == 0), 0,
                                   2 * ((class_precisions * class_recalls) / (class_precisions + class_recalls)))

        weighted_f1_score = np.sum(class_f1_scores * class_counts) / np.sum(class_counts)

        return weighted_f1_score

    def calculate_accuracy(self):
        correct_predictions = sum(pred == label for pred, label in zip(self.predictions, self.labels))
        total_predictions = len(self.predictions)
        accuracy = correct_predictions / total_predictions
        return accuracy
