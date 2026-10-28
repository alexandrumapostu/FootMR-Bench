"""Retrieval metrics for ranked event_id lists against a relevance set."""


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant) / k


def average_precision(ranked_ids: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precisions = []
    for i, eid in enumerate(ranked_ids, start=1):
        if eid in relevant:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    import math

    if not relevant:
        return 0.0
    dcg = 0.0
    for i, eid in enumerate(ranked_ids[:k], start=1):
        if eid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, eid in enumerate(ranked_ids, start=1):
        if eid in relevant:
            return 1.0 / i
    return 0.0


def auc_roc(ranked_ids: list[str], relevant: set[str]) -> float | None:
    """Area under the ROC curve for this query's ranking, treating each
    candidate event as a binary classification (relevant vs not). Computed
    directly from rank order via the standard Mann-Whitney U equivalence -
    AUC = (concordant positive/negative pairs) / (n_positive * n_negative) -
    so it needs no raw model scores, only the ranking, same as every other
    metric here. Returns None when undefined (every candidate in the match
    is relevant, or none are) - callers should skip such queries rather
    than treat None as 0.
    """
    neg_total = sum(1 for eid in ranked_ids if eid not in relevant)
    pos_total = len(ranked_ids) - neg_total
    if pos_total == 0 or neg_total == 0:
        return None
    concordant = 0
    neg_below = 0  # negatives seen so far, walking from the worst-ranked end
    for eid in reversed(ranked_ids):
        if eid in relevant:
            concordant += neg_below
        else:
            neg_below += 1
    return concordant / (pos_total * neg_total)


def roc_curve_points(ranked_ids: list[str], relevant: set[str]):
    """(fpr, tpr) point lists tracing this single query's ROC curve, swept
    one rank position at a time (threshold = top-1, top-2, ... top-n). None
    when undefined, same condition as auc_roc. Used to build macro-averaged
    ROC curves across many queries (interpolate each query's curve onto a
    common FPR grid, then average) - see scripts/plot_roc_curves.py.
    """
    neg_total = sum(1 for eid in ranked_ids if eid not in relevant)
    pos_total = len(ranked_ids) - neg_total
    if pos_total == 0 or neg_total == 0:
        return None
    tp = fp = 0
    fpr, tpr = [0.0], [0.0]
    for eid in ranked_ids:
        if eid in relevant:
            tp += 1
        else:
            fp += 1
        fpr.append(fp / neg_total)
        tpr.append(tp / pos_total)
    return fpr, tpr


def precision_recall_points(ranked_ids: list[str], relevant: set[str]):
    """(recall, precision) point lists tracing this single query's
    precision-recall curve - one point per relevant item retrieved, in rank
    order. None when `relevant` is empty. Used to build macro-averaged PR
    curves across many queries (interpolate each query's curve onto a common
    recall grid via the standard max-precision-at-or-above-recall envelope,
    then average) - see scripts/plot_pr_curves.py. The area under the
    macro-averaged curve approximates, but is not numerically identical to,
    mAP (average_precision averaged over queries without interpolation) -
    same relationship this project's AUC-ROC has to roc_curve_points.
    """
    if not relevant:
        return None
    n_relevant = len(relevant)
    hits = 0
    recall, precision = [0.0], [1.0]
    for i, eid in enumerate(ranked_ids, start=1):
        if eid in relevant:
            hits += 1
            recall.append(hits / n_relevant)
            precision.append(hits / i)
    return recall, precision


def evaluate_queries(queries: list[dict], rank_fn, ks=(1, 5, 10)) -> dict:
    """rank_fn(query) -> list[event_id] sorted by descending predicted score,
    scoped to that query's match. Returns mean metrics over all queries.
    """
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals["map"] = 0.0
    totals["mrr"] = 0.0
    totals["auc"] = 0.0
    totals.update({f"ndcg@{k}": 0.0 for k in ks})
    n = 0
    n_auc = 0  # separate count - auc_roc() is undefined (None) when a match has no
    # negatives (every event relevant) or no positives, which can't happen for the
    # other metrics once `relevant` is non-empty, so it needs its own denominator.
    for q in queries:
        relevant = set(q["relevant"])
        if not relevant:
            continue
        ranked = rank_fn(q)
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)
        totals["map"] += average_precision(ranked, relevant)
        totals["mrr"] += mrr(ranked, relevant)
        auc = auc_roc(ranked, relevant)
        if auc is not None:
            totals["auc"] += auc
            n_auc += 1
        n += 1
    if n == 0:
        return {k: 0.0 for k in totals}
    out = {k: v / n for k, v in totals.items() if k != "auc"}
    out["auc"] = totals["auc"] / n_auc if n_auc else 0.0
    return out
