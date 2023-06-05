import glob
import numpy as np
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def metric_DMD(known, novel):
    """ Based on https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py
    """

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
    TNR = 1. - fp[tpr95_pos] / num_n

    # AUROC
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    AUROC = -np.trapz(1. - fpr, tpr)

    # DTACC
    DTACC = .5 * (tp / tp[0] + 1. - fp / fp[0]).max()

    # AUIN
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp / denom, [0.]])
    AUIN = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    AUOUT = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])

    result = [AUROC, DTACC, TNR, (AUIN+AUOUT)/2]
    return result


def get_seeds(a_type, fe="cifar10_ResNet"):
    path = "features_and_scores"
    if a_type == "train_seed_old":
        seeds = range(10)
    elif a_type == "all_epochs":
        if fe.split("_")[0] == "cifar10":
            seeds = range(0, 147)
        else:
            seeds = range(0, 215)
        seeds = [
            f"{path}/EPOCH_epoch_{'{:03d}'.format(s)}_{fe}_seed_0/" for s in seeds]
    elif a_type == "epochs":
        if fe.split("_")[0] == "cifar10":
            seeds = range(92, 147)
        else:
            seeds = range(120, 215)
        seeds = [
            f"{path}/EPOCH_epoch_{'{:03d}'.format(s)}_{fe}_seed_0/" for s in seeds]
    elif a_type == "train_seed":
        seeds = [filename for filename in glob.iglob(
            f"{path}/{fe}_seed*")]
    elif a_type == "split":
        seeds = [filename for filename in glob.iglob(
            f"{path}/SPLIT*{fe}*")]
    elif a_type == "type":
        seeds = [filename for filename in glob.iglob(
            f"{path}/*{fe}*type*")]
    elif a_type == "aug":
        seeds = [filename for filename in glob.iglob(
            f"{path}/AUG*{fe}*")]
    else:
        raise Exception("Unknown a_type")
    return seeds


def load_data(seed, method, ood):
    path = f"{seed}/scores/{method}/"
    x_ood = np.load(f"{path}ood_{ood}.npy")
    x_test = np.load(f"{path}test.npy")
    return x_ood, x_test


def get_close_acc(seed):
    x = np.load(seed + "/features/test.pickle", allow_pickle=True)
    ol = x["original_label"].to_numpy()
    u = [
        1 if np.argmax(u) == l else 0 for l,
        u in zip(
            x["original_label"].to_numpy(),
            x["classifier"].to_numpy())]
    return (sum(u) / len(u)) * 100


def get_indexes(n, m):
    indexes = np.asarray(range(n))
    np.random.shuffle(indexes)
    ind = indexes[:int(m)]
    return ind


def get_data_indexes(ood_data, number_of_div=1, seed=44):
    n = 50000
    m = 10000
    if ood_data == "svhn":
        n = 73257
    results = []
    for i in range(number_of_div):
        results.append({})
    indexes = []
    np.random.seed(seed)
    for i in range(number_of_div):
        indexes.append(get_indexes(n, m))
    if number_of_div == 1:
        return indexes[0]
    return indexes


def generate_results(method, ood_data, seeds, indexes, metric_type=0):
    results = []
    for j, seed in enumerate(seeds):
        try:
            ood, test = load_data(method=method, seed=seed, ood=ood_data)
            last_ood = ood
            last_test = test
        except Exception as e:
            print(e)
            ood = last_ood
            test = last_test
        results.append(metric_DMD(test, ood[indexes])[metric_type])
    return results


def analyse_results(results, ood_methods, verbose=True):
    ranks = {method: [] for method in ood_methods}
    raws = {method: [] for method in ood_methods}
    for j, result in enumerate(results):
        table = [result[method] for method in ood_methods]
        orders = [
            i[0] for i in sorted(
                enumerate(table),
                key=lambda x:x[1],
                reverse=True)]
        s_orders = [0 for i in range(len(orders))]
        for i, pos in enumerate(orders):
            s_orders[pos] = i
        for i, method in enumerate(ood_methods):
            raws[method].append(table[i] * 100)
            ranks[method].append(int(s_orders[i]))
    a_results = {}
    for method in ood_methods:
        rank = ranks[method]
        raw = raws[method]
        r_rank = analyse_metric(rank)
        r_raw = analyse_metric(raw)
        a_results[method] = {}
        a_results[method]["rank"] = r_rank
        a_results[method]["raw"] = r_raw
        if verbose:
            print(method, r_rank, r_raw)
    return a_results


def analyse_metric(raw):
    return {
        "mean": np.mean(raw),
        "std": np.std(raw),
        "max": np.max(raw),
        "min": np.min(raw)}


ood_methods = [
    "KNN",
    "Mahalanobis",
    "MaxLogits",
    "MaxSoftmax",
    "lof_cosine",
    "lof_euclidean",
    "FreeEnergy_t=1.0"]
models = ["cifar100_MobileNet", "cifar10_ResNet"]


def experiment(a_type, metric_type=0):
    a_results = {}
    for fe in models:
        a_results[fe] = {}
        seeds = get_seeds(a_type, fe)
        closed_acc = [get_close_acc(seed) for seed in seeds]
        results = analyse_metric(closed_acc)
        a_results[fe]["ACC"] = results
        print("Model", fe, "closed_acc", results)
        oods = ["svhn", "cifar100"]
        if fe.startswith("cifar100"):
            oods = ["svhn", "cifar10"]
        for ood in oods:
            print("OOD", ood)
            indexes = get_data_indexes(ood)
            results = [{} for i in range(len(seeds))]
            for method in ood_methods:
                rs = generate_results(method, ood, seeds, indexes, metric_type)
                for i, r in enumerate(rs):
                    results[i][method] = r
            a_results[fe][ood] = analyse_results(results, ood_methods)
        print()
    with open(a_type + ".json", "wt") as f:
        json.dump(a_results, f, cls=NpEncoder)


def generate_result_multi_index(
        method,
        ood_data,
        seed,
        indexes,
        metric_type=0):
    results = []
    ood, test = load_data(method=method, seed=seed, ood=ood_data)
    for j, index in enumerate(indexes):
        results.append(metric_DMD(test, ood[index])[metric_type])
    return results


def experiment_ood_random(metric_type=0):
    number = 100
    a_results = {}
    a_type = "train_seed"
    for fe in models:
        oods = ["svhn", "cifar100"]
        if fe.startswith("cifar100"):
            oods = ["svhn", "cifar10"]
        a_results[fe] = {}
        seeds = get_seeds(a_type, fe)
        for seed in seeds:
            if seed[-1] != '0':
                continue
            for ood in oods:
                o_results = {}
                a_results[fe][ood] = o_results
                indexes = get_data_indexes(ood, number_of_div=number)
                results = [{} for i in range(number)]
                for method in ood_methods:
                    rs = generate_result_multi_index(
                        method, ood, seed, indexes, metric_type)
                    for i, r in enumerate(rs):
                        results[i][method] = r
                a_results[fe][ood] = analyse_results(results, ood_methods)
    print(a_results)
    with open("ood_random.json", "wt") as f:
        json.dump(a_results, f, cls=NpEncoder)


def experiment_type(metric_type=0):
    results = []
    a_type = "type"
    fe = "cifar10_ResNet"
    seeds = get_seeds(a_type, fe)
    for seed in seeds:
        s_results = {}
        results.append(s_results)
        s_results["ACC"] = get_close_acc(seed)
        oods = ["svhn", "cifar100"]
        for ood in oods:
            o_results = {}
            s_results[ood] = o_results
            indexes = get_data_indexes(ood)
            for method in ood_methods:
                rs = generate_results(
                    method, ood, [seed], indexes, metric_type)
                o_results[method] = rs[0] * 100
    print(results)
    with open(a_type + ".json", "wt") as f:
        json.dump(results, f, cls=NpEncoder)


def experiment_aug(metric_type=0):
    a_results = {}
    a_type = "aug"
    for fe in models:
        results = []
        a_results[fe] = results
        seeds = get_seeds(a_type, fe)
        for seed in reversed(seeds):
            name = seed.split("_")[3]
            s_results = {}
            results.append(s_results)
            s_results["ACC"] = get_close_acc(seed)
            s_results["name"] = name
            oods = ["svhn", "cifar100"]
            if fe.startswith("cifar100"):
                oods = ["svhn", "cifar10"]
            for ood in oods:
                o_results = {}
                s_results[ood] = o_results
                indexes = get_data_indexes(ood)
                for method in ood_methods:
                    try:
                        rs = generate_results(
                            method, ood, [seed], indexes, metric_type)
                        rs = rs[0] * 100
                    except BaseException:
                        rs = -1
                    o_results[method] = rs
    print(a_results)
    with open(a_type + "2.json", "wt") as f:
        json.dump(a_results, f, cls=NpEncoder)


def epochs(a_type, fe, ood, metric_type):
    indexes = get_data_indexes(ood)
    seeds = get_seeds(a_type, fe)
    results = []
    for i, seed in enumerate(seeds):
        acc = get_close_acc(seed)
        res = {"ACC": acc, "epoch": i}
        for method in ood_methods:
            oe_ood, oe_test = load_data(seed=seed, method=method, ood=ood)
            metric = metric_DMD(oe_test, oe_ood[indexes])[metric_type]
            res[method] = metric
        results.append(res)

    with open(a_type + "_" + fe + "_" + ood + ".json", "wt") as f:
        json.dump(results, f, cls=NpEncoder)


def experiment_epochs(metric_type=0):
    a_type = "all_epochs"
    for fe in models:
        oods = ["svhn", "cifar100"]
        if fe.startswith("cifar100"):
            oods = ["svhn", "cifar10"]
        for ood in oods:
            epochs(a_type, fe, ood, metric_type)


def all_experiments(metric_type=0):
    # metric_type = 0 AUC
    #               1 DTACC
    #               2 TNR
    #               3 AUPR

    print("Table 1")
    experiment_type(metric_type)

    print("Table 2")
    experiment("train_seed", metric_type)

    print("Table 3")
    experiment_ood_random(metric_type)

    print("Table 4")
    experiment("split", metric_type)

    print("Table 5")
    experiment_aug(metric_type)

    print("Figure1")
    experiment_epochs(metric_type)


all_experiments()
