import os


def score_cvsm(result_filename):
    # score_instances should be a tuple of (stuff, label, score)
    score_instances = []
    target_relation = None
    with open(result_filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            target_relation, entity_pair_idx, score, label = line.split("\t")
            score = float(score)
            label = int(label)
            score_instances.append(((target_relation, entity_pair_idx), label, score))
    print("Computing AP, RR, ACC for relation", target_relation, "for CVSM")
    print("total number of predictions:", len(score_instances))
    ap, rr, acc = compute_scores(score_instances)
    print("AP:", ap, "\nRR:", rr, "\nACC:", acc)
    return ap, rr, acc


def compute_ap_and_rr(score_instances):
    """
    Given a list of scored instances [(stuff, label, score)], this method computes AP and RR.
    AP is none if no positive instance is in scored instances.

    :param score_instances:
    :return:
    """
    # sort score instances based on score from highest to lowest
    sorted_score_instances = sorted(score_instances, key=lambda score_instance: score_instance[2])[::-1]
    total_predictions = 0.0
    total_corrects = 0.0
    total_precisions = []
    first_correct = -1
    for stuff, label, score in sorted_score_instances:
        # print(stuff, label, score)
        total_predictions += 1
        if label == 1:
            total_corrects += 1
            if first_correct == -1:
                first_correct = total_predictions
            total_precisions.append(total_corrects/total_predictions)
    ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(total_precisions) > 0 else None
    rr = 0.0 if first_correct == -1 else 1.0 / first_correct
    return ap, rr


def compute_scores(score_instances):
    """
    Given a list of scored instances [(stuff, label, score)], this method computes Average Precision, Reciprocal Rank,
    and Accuracy.
    AP is none if no positive instance is in scored instances.

    :param score_instances:
    :return:
    """
    # sort score instances based on score from highest to lowest
    sorted_score_instances = sorted(score_instances, key=lambda score_instance: score_instance[2])[::-1]
    total_predictions = 0.0
    total_correct_pos = 0.0
    total_precisions = []
    first_correct = -1
    total_correct = 0.0
    for stuff, label, score in sorted_score_instances:
        # print(stuff, label, score)
        if abs(score - label) < 0.5:
            total_correct += 1
        total_predictions += 1
        # debug
        if label > 0:
        # if label == 1:
            total_correct_pos += 1
            if first_correct == -1:
                first_correct = total_predictions
            total_precisions.append(total_correct_pos/total_predictions)
    ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(total_precisions) > 0 else None
    rr = 0.0 if first_correct == -1 else 1.0 / first_correct
    acc = total_correct / len(score_instances)
    return ap, rr, acc
