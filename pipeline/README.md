# Benchmark construction pipeline

You don't need to run any of this to use FootMR-Bench — `dataset/{train,valid,test}/{events,queries}.jsonl`
is already the finished benchmark. This directory exists for transparency and
in case you want to rebuild or extend it (e.g. new query templates, a
different event-class selection).

### 1. Download raw SoccerNet

Requires access to SoccerNet (free, request via the [SoccerNet
website](https://www.soccer-net.org/)). Place `Labels-v2.json` and the
`caption-2023/` commentary files under `data/soccernet/`, matching the layout
the `SoccerNet` pip package downloads to.

### 2. Extract events

```bash
python pipeline/build_event_collection.py --window 10 --caption-tolerance 30
```
Reads every `Labels-v2.json`, attaches aligned commentary where available, and
writes one JSON file per match to `data/processed/events/`.

### 3. Flatten to one file

```bash
python pipeline/flatten_events.py
```
Reads `data/processed/events/**/*.json` and writes
`data/processed/events_all.jsonl` — one event per line.

### 4. Generate queries

```bash
python pipeline/generate_queries.py
```
Reads `data/processed/events/`, applies the 18 query templates, and writes
`data/processed/queries.jsonl` with each query's `relevant` event IDs already
resolved.

### 5. Paraphrase (optional)

```bash
ollama pull llama3.1:8b   # once
ollama serve               # local server, no API key
python pipeline/rephrase_queries.py --split valid
python pipeline/rephrase_queries.py --split test
```
Diversifies phrasing for the validation/test queries via a local LLM
(Ollama, free, no external API key needed). Training queries are left
untouched. Skipping this step leaves the raw template text in place.

### 6. Build the final split

```bash
python pipeline/build_dataset.py
```
Reads `data/processed/events_all.jsonl` and `data/processed/queries.jsonl`,
partitions matches into train/valid/test, and writes the clean
`dataset/<split>/{events,queries}.jsonl` files — the same files already
shipped in this repository.
