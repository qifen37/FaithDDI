from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, accuracy_score
import numpy as np

def multi_class_eval(labels, pred):
    pred = [sample.argmax() for sample in pred]
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='macro')
    precision = precision_score(labels, pred, average='macro', zero_division=0)
    recall = recall_score(labels, pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels, pred)

    return acc * 100, f1 * 100, precision * 100, recall * 100, kappa * 100
