# DeskTriage-Intent — Support Ticket Intent Classifier

**Sector:** Customer support  
**Task:** Multi-class text classification

Classifies synthetic support messages into billing, account access, refunds, order status, technical issues, shipping, and cancellation intents.

## Data and provenance

The offline demo is generated locally by `desktriage/data.py`. It uses hand-written synthetic templates and does not include or download real support tickets, resumes, applicant records, or job postings. No external dataset license applies to these generated examples.

## Models and evaluation

- **Baseline:** word-count features with Multinomial Naive Bayes.
- **Stronger model:** word and bigram TF-IDF with class-balanced logistic regression.
- **Split:** stratified random holdout of independently generated messages.
- **Metrics:** accuracy, macro F1, confusion matrix and error count; binary matching additionally reports ROC-AUC, average precision, precision, and recall at the default 0.5 probability threshold for binary matching.
- **Error analysis:** confusion matrix and overall misclassification count; for matching, diagnostic subgroup rates are included.

## Limitations

The templates contain a small hand-written vocabulary and do not represent real customer language, languages, accessibility needs, or evolving product terminology. Treat the output as triage support only; a person should review routing and edge cases.

## Run

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m desktriage.train --seed 42 --samples 1600 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` are shortcuts. Python 3.9 or newer is required.

## Layout

- `desktriage/data.py`: deterministic synthetic text generator.
- `desktriage/split.py`: stratified or candidate-group split.
- `desktriage/model.py`: text pipelines and evaluation.
- `desktriage/train.py`: train/evaluate CLI.
- `tests/`: data, split, model, and CLI tests.
- `.github/workflows/ci.yml`: CI test and CLI smoke run.
