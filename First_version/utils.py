import numpy as np


def my_print(print_sentence, need_to_print):
    if need_to_print:
        print(print_sentence)


def write_metric_tb(tb, metrics, epoch):
    for label, metric in metrics.items():
        tb.add_scalar(label, np.array(metric), epoch)
