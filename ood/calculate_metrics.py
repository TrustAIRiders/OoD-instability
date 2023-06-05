import numpy as np
from sklearn import metrics


class CalculateMetrics:
    def __init__(self):
        self._number_of_init_probes = 50

    def run(self, openset_method):
        (fpr_95, fpr_90), _, _ = self._calculate_openset_metrics(openset_method)
        tnr, auc, _, auin, auout, max_bin_acc, acc_close, acc_open = self._calculate_metric_dmd(openset_method)

        metrics_result = {}
        metrics_result["acc_close"] = acc_close
        metrics_result["acc_open"] = acc_open
        metrics_result["max_bin_acc"] = max_bin_acc
        metrics_result["auc"] = auc
        metrics_result["aupr_in"] = auin
        metrics_result["aupr_out"] = auout
        metrics_result["fpr_95"] = fpr_95
        metrics_result["fpr_90"] = fpr_90
        metrics_result["tnr_95"] = tnr

        for m in metrics_result.keys():
            metrics_result[m] = self._nice_number(metrics_result[m])

        return metrics_result

    def _nice_number(self, x):
        return int(x * 100000) / 100000

    def _calculate_metric_dmd(self, openset_method):
        # Based on https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py

        known = openset_method.known_out
        novel = openset_method.unknown_out

        # tp, fp
        known = np.asarray(known)
        novel = np.asarray(novel)
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp = -np.ones([num_k + num_n + 1], dtype=int)
        fp = -np.ones([num_k + num_n + 1], dtype=int)
        tp[0], fp[0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[l + 1:] = tp[l]
                fp[l + 1:] = np.arange(fp[l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[l + 1:] = np.arange(tp[l] - 1, -1, -1)
                fp[l + 1:] = fp[l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[l + 1] = tp[l]
                    fp[l + 1] = fp[l] - 1
                else:
                    k += 1
                    tp[l + 1] = tp[l] - 1
                    fp[l + 1] = fp[l]

        # TNR
        tpr95_pos = np.abs(tp / num_k - .95).argmin()
        tnr = 1. - fp[tpr95_pos] / num_n

        # AUROC
        tpr = np.concatenate([[1.], tp / tp[0], [0.]])
        fpr = np.concatenate([[1.], fp / fp[0], [0.]])
        auroc = -np.trapz(1. - fpr, tpr)

        # ACC BIN
        ACC = (tp + fp[0] - fp) / (tp[0] + fp[0])
        best_acc = np.argmax(ACC)
        max_bin_acc = ACC[best_acc]

        acc_close = tpr[best_acc]
        acc_open = 1. - fpr[best_acc]

        # DTACC
        dtacc = .5 * (tp / tp[0] + 1. - fp / fp[0]).max()

        # AUIN
        denom = tp + fp
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp / denom, [0.]])
        auin = -np.trapz(pin[pin_ind], tpr[pin_ind])

        # AUOUT
        denom = tp[0] - tp + fp[0] - fp
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
        auout = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])

        return tnr, auroc, dtacc, auin, auout, max_bin_acc, acc_close, acc_open

    def _calculate_openset_metrics(self, openset_method):
        pos = -openset_method.unknown_out
        neg = -openset_method.known_out
        examples = np.concatenate((pos, neg), axis=None)
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        auroc = metrics.roc_auc_score(labels, examples)
        aupr = metrics.average_precision_score(labels, examples)
        fpr = self._fpr_and_fdr_at_recall(labels, examples)

        return fpr, auroc, aupr

    def _fpr_and_fdr_at_recall(self, y_true, y_score, recall_level=[0.95, 0.9], pos_label=None):
        # Based on https://github.com/nazim1021/OOD-detection-using-OECC

        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        y_true = (y_true == pos_label)

        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        tps = self._stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        thresholds = y_score[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

        divisor = (np.sum(np.logical_not(y_true)))
        res = [fps[np.argmin(np.abs(recall - r))] / divisor for r in recall_level]
        return res

    def _stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        # Based on https://github.com/nazim1021/OOD-detection-using-OECC
        # Use high precision for cumsum and check that final value matches sum

        out = np.cumsum(arr, dtype=np.float64)
        expected = np.sum(arr, dtype=np.float64)
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: its last element does not correspond to sum')
        return out

    def _get_problem_area(self, specificity, sensitivity, threshold=0.01):
        size = len(specificity)
        max_delta = 0
        index_max = 0

        for i in range(size - 1):
            dspec = - specificity[i + 1] + specificity[i]
            desens = - sensitivity[i] + sensitivity[i + 1]
            delta = min(dspec, desens)
            if delta > max_delta:
                max_delta = delta
                index_max = i
        if threshold < max_delta:
            return index_max
        return -1

    def get_p_acc_bin_for_graphs(self, openset_method):
        results = []
        all_metrics = self._calculate_all_basic_metrics(openset_method)
        results.append(["best_bin", all_metrics["probes"][all_metrics["bin_acc"].argmax()], all_metrics["bin_acc"].max()])

        best_i = 0
        diff = 1
        for i in range(len(all_metrics["acc_close"])):
            if abs(0.95 - all_metrics["acc_close"][i]) < diff:
                diff = abs(0.95 - all_metrics["acc_close"][i])
                best_i = i
        results.append(["close_95", all_metrics["probes"][best_i], all_metrics["bin_acc"][best_i]])

        best_i = 0
        diff = 1
        for i in range(len(all_metrics["acc_close"])):
            if abs(all_metrics["acc_close"][i] - all_metrics["acc_open"][i]) < diff:
                diff = abs(all_metrics["acc_close"][i] - all_metrics["acc_open"][i])
                best_i = i
        results.append(["min_diff", all_metrics["probes"][best_i], all_metrics["bin_acc"][best_i]])

        for i in range(len(results)):
            results[i][2] = self._nice_number(results[i][2])
            results[i] = tuple(results[i])

        return results

    def _calculate_all_basic_metrics(self, openset_method):
        probes = np.array([x / self._number_of_init_probes for x in range(self._number_of_init_probes + 1)])
        all_metrics = {}
        for m in ["probes", "acc_all", "acc_close", "acc_open", "bin_acc", "sensitivity", "specificity", "bin_f1"]:
            all_metrics[m] = np.empty(len(probes))

        for index, p in enumerate(probes):
            new_basic_metrics = self._calculate_basic(openset_method, p)
            for m in all_metrics.keys():
                all_metrics[m][index] = new_basic_metrics[m]

        index = self._get_problem_area(all_metrics["specificity"], all_metrics["sensitivity"])
        while index > -1:
            probes = all_metrics["probes"]
            new_p = (probes[index] + probes[index + 1]) / 2

            new_basic_metrics = self._calculate_basic(openset_method, new_p)
            for m in all_metrics.keys():
                all_metrics[m] = np.insert(all_metrics[m], index + 1, new_basic_metrics[m])

            if len(all_metrics["sensitivity"]) > 3 * self._number_of_init_probes:
                break
            index = self._get_problem_area(all_metrics["specificity"], all_metrics["sensitivity"])

        return all_metrics

    def _calculate_basic(self, openset_method, p):
        correct = [label for label in openset_method.known_df["original_label"]]
        correct += [-1] * len(openset_method.unknown_df)

        predicted = [label if not is_openset else -1 for is_openset, label in
                     zip(openset_method.verify(openset_method.known_out, p), openset_method.known_df["predicted_label"])
                     ]
        predicted += [label if not is_openset else -1 for is_openset, label in
                      zip(openset_method.verify(openset_method.unknown_out, p), openset_method.unknown_df["predicted_label"])
                      ]

        correct_close = [1] * len(openset_method.known_df["original_label"])
        predicted_close = [1 if predicted[i] != -1 else 0 for i in range(len(correct)) if correct[i] != -1]

        correct_open = [-1] * len(openset_method.unknown_df)
        predicted_open = [predicted[i] for i in range(len(correct)) if correct[i] == -1]

        correct_binary = [0 if correct[i] != -1 else 1 for i in range(len(correct))]
        predicted_binary = [0 if predicted[i] != -1 else 1 for i in range(len(predicted))]

        tn, fp, fn, tp = metrics.confusion_matrix(correct_binary, predicted_binary).ravel()

        metrics_result = {}
        metrics_result["probes"] = p
        metrics_result["acc_all"] = metrics.accuracy_score(correct, predicted)
        metrics_result["acc_close"] = metrics.accuracy_score(correct_close, predicted_close)
        metrics_result["acc_open"] = metrics.accuracy_score(correct_open, predicted_open)

        metrics_result["bin_acc"] = metrics.accuracy_score(correct_binary, predicted_binary)
        metrics_result["sensitivity"] = tp / (tp + fn)
        metrics_result["specificity"] = tn / (tn + fp)
        metrics_result["bin_f1"] = (2 * tp) / (2 * tp + fp + fn)

        return metrics_result