# FootMR-Bench

A benchmark for **match-level football moment retrieval**: given a natural-language
query and a full football match, rank that match's annotated events by relevance
to the query. Unlike clip-level moment retrieval, every query is resolved against
an entire 90-plus-minute match, which is what makes ordinal queries ("the first
corner") and event-anchored queries ("the foul before the second goal") answerable
at all — their answer depends on events outside any single clip.

Built on top of [SoccerNet](https://www.soccer-net.org/)'s action-spotting
annotations: 48,938 queries over 500 matches, spanning 18 query categories
(single-event, multi-answer, ordinal, event-anchored, player/team-scoped), with
5 categories held out of training entirely to measure generalization to unseen
query structure.

This repository releases the full benchmark (events + queries, no NDA or
gating required — it's annotation metadata, not redistributed video), the
pipeline that built it, and the training/evaluation/reranking code used in
the paper.

## Task

Each match $m$ contributes a set of candidate events $\mathcal{E}_m$, each
observed only through a short video window around its timestamp (never its
class, half, or timestamp directly). A query $q$ has a non-empty relevant set
$\mathcal{R}(q) \subseteq \mathcal{E}_m$, possibly more than one event. A
model scores every candidate and is evaluated on how well the induced ranking
recovers $\mathcal{R}(q)$.

## Dataset

| Split | Matches | Queries |
|---|---|---|
| train | 300 | 26,995 |
| valid | 100 | 10,761 |
| test  | 100 | 11,182 |

`dataset/<split>/events.jsonl` — one event per line:
```json
{"match_id": "...", "event_id": "e000", "half": 1, "t": 339, "label": "Shots off target",
 "team": "home", "player": null, "caption": null, "window": [329, 349], "visibility": "visible"}
```

`dataset/<split>/queries.jsonl` — one query per line:
```json
{"query_id": "q000446", "match_id": "...", "text": "the yellow card Ander Herrera received",
 "template_id": "player_single", "relevant": ["e058"], "n_relevant": 1,
 "template_seen_in_train": true, "text_for_training": "Ander Herrera yellow card"}
```
`relevant` is the exact set of `event_id`s that answer the query — use it as
ground truth; `template_seen_in_train=false` marks the five held-out
generalization categories. `text_for_training` is the (possibly paraphrased)
query text actually used for training/eval; fall back to `text` if absent.

See [`pipeline/README.md`](pipeline/README.md) for how this was built from
raw SoccerNet annotations, if you want to reproduce or extend it.

## Metrics

Recall@1/5/10, MRR, mAP, nDCG@10, and per-query ROC-AUC, macro-averaged over
queries. mAP is the headline metric (~45% of queries have more than one
correct answer, and mAP is the only one of these that penalizes burying a
second correct answer). `mAP(U)` restricts scoring to the five held-out
categories, to report generalization separately.

## Main results

Four pre-trained vision-language backbones (CLIP4Clip, X-CLIP, SigLIP,
MetaCLIP), each evaluated zero-shot, LoRA fine-tuned, and fine-tuned plus our
ordinal-aware reranker (test split):

| Model | Stage | R@1 | mAP | AUC |
|---|---|---|---|---|
| CLIP4Clip | Zero-shot | 0.056 | 0.179 | 0.651 |
| CLIP4Clip | Fine-tuned (LoRA) | 0.180 | 0.438 | 0.901 |
| CLIP4Clip | + reranking | **0.207** | **0.463** | **0.904** |
| X-CLIP | Zero-shot | 0.024 | 0.108 | 0.567 |
| X-CLIP | Fine-tuned (LoRA) | 0.176 | 0.440 | 0.905 |
| X-CLIP | + reranking | **0.208** | **0.469** | **0.908** |
| SigLIP | Zero-shot | 0.030 | 0.110 | 0.539 |
| SigLIP | Fine-tuned (LoRA) | 0.212 | 0.486 | 0.918 |
| SigLIP | + reranking | **0.232** | **0.503** | **0.920** |
| MetaCLIP | Zero-shot | 0.039 | 0.142 | 0.589 |
| MetaCLIP | Fine-tuned (LoRA) | 0.167 | 0.418 | 0.894 |
| MetaCLIP | + reranking | **0.194** | **0.443** | **0.896** |

Full table (R@10, mAP(U), nDCG@10, significance tests) in the paper.

## Repository structure

```
footmr-bench/
├── dataset/            benchmark: events.jsonl + queries.jsonl per split
├── pipeline/            construction pipeline (raw SoccerNet -> dataset/)
├── src/                 shared library: data loading, metrics, rankers, reranker
├── scripts/              training and full-pipeline evaluation entry points
└── eval/                 standalone, model-agnostic scoring (no GPU required)
```

## Installation

```bash
pip install -r requirements.txt
```
`torch` with CUDA is recommended for training/inference but not required for
`eval/`, which only needs the predictions you've already generated.

## Usage

### Evaluate predictions you already have (no GPU needed)

Predictions are JSON Lines: `{"query_id": ..., "ranked_event_ids": [...]}`,
one line per query, `ranked_event_ids` sorted most-relevant-first over that
query's match candidates. See `eval/example/` for a worked example.

```bash
python eval/evaluate_predictions.py --predictions your_predictions.jsonl --split test
```

### Fine-tune a backbone with LoRA

```bash
python scripts/train_lora.py --model siglip --epochs 5 --window-radius 5 --num-frames 8 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --lr 1e-4 \
    --output-dir cache/lora/siglip-r16
```
`--model` accepts `clip4clip`, `clip4clip-large`, `xclip`, `siglip`, `metaclip`.

### Run zero-shot / fine-tuned / reranked evaluation end-to-end

```bash
python scripts/run_video_baselines.py --split test --model siglip \
    --lora-adapter cache/lora/siglip-r16/epoch5 --lora-family siglip \
    --ordinal-rerank-softmax --save-predictions-dir predictions/
```
This reproduces the numbers in the table above and writes predictions in the
format `eval/evaluate_predictions.py` expects, for independent scoring.

## Citation

```bibtex
@inproceedings{apostu2027footmrbench,
  title     = {FootMR-Bench: A Novel Benchmark for Match-Level Football Moment Retrieval},
  author    = {Apostu, Alexandru-Mihai and Danciu, Valentin-Nicolae and Ionescu, Radu Tudor},
  booktitle = {Proceedings of the 49th European Conference on Information Retrieval (ECIR)},
  year      = {2027}
}
```

## License

Code is released under the [MIT License](LICENSE). The benchmark (events and
queries) is derived from SoccerNet's Labels-v2 and SoccerNet-Caption
annotations; usage of the underlying match video remains subject to
[SoccerNet's own terms](https://www.soccer-net.org/) — this repository does
not redistribute video.
