# Vietnamese Robot Vacuum Facebook Comments NLP

Reproducible pipeline required by `CODEX_INSTRUCTIONS_ROBOT_VACUUM_NLP.md`.

## Run

```bash
python -m pip install -r requirements.txt
python src/run_nlp.py --input robothutbui_facebookscrape.xlsx --sheet comments --output outputs --seed 42
```

The pipeline reads only `comments.text`, keeps source Excel row IDs, masks contact PII in exports, audits commercial content, runs Vietnamese sentiment inference, BERTopic, charts, report, metadata, and acceptance checks.

## Deliverables

See `outputs/` for the CSV, PNG, Markdown and JSON deliverables specified by the instruction file. Raw source text is not exported; `text_raw` exists only in the in-memory/intermediate audit and is masked before CSV export.

## Verification

The script asserts sentiment totals, topic totals including topic -1, crosstab totals, non-empty PNGs, and no phone/email leakage in exported text artifacts. No human-labeled sentiment validation is claimed; the commercial review sample is a review aid only.
