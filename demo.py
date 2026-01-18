import random
import numpy as np
import matplotlib.pyplot as plt
from multiclass_evaluator import MetricsCalculator
from incre_space_data_stream import incre_stream
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from subspace_classifier import SubspaceResidualTrapezoidalClassifier

random.seed(42)

in_stream = incre_stream("data/wave.csv", change_times=5)
inc_data, fix_data, all_data, label = in_stream.next_sample(500, return_fixed=True, return_all=True)

classes = in_stream.target_values
target_classifier = SubspaceResidualTrapezoidalClassifier(n_subspaces = 7, js_threshold = 0.001)
compare_classifier = DynamicWeightedMajorityClassifier()

target_classifier.partial_fit(inc_data, label, classes=classes)
compare_classifier.partial_fit(fix_data, label, classes=classes)

matrix = MetricsCalculator(classes = classes)
matrix2 = MetricsCalculator(classes = classes)


data_size = 0
result_list = []
result_list2 = []
t_list = []

index = -1

while in_stream.has_more_samples() and data_size < 10000000:
    inc_data, fix_data, label = in_stream.next_sample(return_fixed=True)

    predict = target_classifier.predict(inc_data)
    matrix.add_result(predict, label)

    predict = compare_classifier.predict(fix_data)
    matrix2.add_result(predict, label)

    if random.random() < 1:
        target_classifier.partial_fit(inc_data, label)
        compare_classifier.partial_fit(fix_data, label)


    data_size = data_size + 1

    if data_size % 500 == 0.:
        result_list.append(matrix.calculate_accuracy())
        result_list2.append(matrix2.calculate_accuracy())
        matrix = MetricsCalculator(classes= classes)
        matrix2 = MetricsCalculator(classes= classes)
        t_list.append(data_size)
        plt.plot(t_list, result_list, c='r', ls='-', marker='o', mec='b', mfc='w', label='SALF')
        plt.plot(t_list, result_list2, c='b', ls='-', marker='o', mec='b', mfc='w', label='DWM')
        if index == -1:
            plt.legend()
            index = 0
        plt.pause(0.01)
        print(data_size)

print(np.mean(result_list))
print(np.mean(result_list2))

plt.show()
