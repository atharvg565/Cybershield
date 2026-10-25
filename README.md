# CyberShield

**Explainable phishing and digital fraud detection for SMS, email and links.**

Paste a suspicious message or link. CyberShield returns a **Low / Medium / High** risk rating, a score, the specific reasons behind it, and clear advice on what to do next.

Built for the hackathon problem statement **CC6-AI05: Intelligent Phishing and Digital Fraud Detection**.

> **Status: prototype.** It runs locally. See [Limitations](#limitations) and [Roadmap](#roadmap) for what is and is not done.

---

## Why CyberShield

Blocklists only catch attacks that are already known, and black-box scores don't tell users *why* something is dangerous. CyberShield layers three defences and shows its reasoning at every step.

1. **Known threats:** links are checked against a local threat-feed blocklist. A match is rated High immediately.
2. **Unknown links:** a machine-learning model scores each URL from its structure, and rules catch look-alike brand domains such as `paypa1.com`.
3. **Scam language:** rules score the message text for urgency, threats, requests for passwords or OTPs, payment demands, prize offers and advance-fee wording.

## Features

- Risk rating (Low / Medium / High) with a numeric score
- Explainable alerts listing each indicator that fired
- URL analysis: length, subdomains, IP-as-host, `@`, punycode, HTTPS, risky TLDs, shorteners, look-alike brands
- Threat-intelligence lookup against local feed files (OpenPhish / URLhaus format)
- Fraud scenarios: fake bank alerts, government impersonation, fake job offers, investment and crypto scams, parcel or delivery scams, advance-fee scams
- Privacy: emails, phone numbers, card numbers and Aadhaar numbers are masked before analysis; raw messages are not stored
- Streamlit dashboard with message checking, session history (CSV export) and a metrics view
- REST API with interactive docs

## How it works

```
Message
  -> privacy.py        mask personal data
  -> api.py            FastAPI endpoint
  -> threat_intel.py   feed lookup (a hit gives score 1.0)
  -> scoring.py        URL model + URL rules + text rules
  -> Low (<0.35) / Medium (0.35 to 0.70) / High (>0.70) + reasons
```

Score = `0.4 x URL model + 0.3 x URL rules + 0.2 x text rules`. If a message has several links, the riskiest one drives the URL part. If it has no links, the text-rule score is used.

## Project structure

```
Cybershield/
├── app/
│   ├── api.py            FastAPI service (/health, /analyze)
│   ├── scoring.py        combines all signals into a risk verdict
│   ├── url_features.py   URL features and readable reasons
│   ├── text_rules.py     scam-language rules
│   ├── threat_intel.py   threat-feed lookup
│   └── privacy.py        PII masking and message fingerprinting
├── train/
│   ├── train_url.py      trains the URL model
│   └── train_text.py     trains a text model (optional, not yet connected)
├── eval/
│   └── evaluate_messages.py   labelled evaluation and metrics
├── data/                 datasets and threat-feed files (not in git)
├── models/               trained models (not in git)
├── dashboard.py          Streamlit dashboard
├── index.html            minimal web page that calls the API
└── README.md
```

## Setup

Requires **Python 3.10 or newer**. These steps are for Windows PowerShell; on macOS or Linux use `source venv/bin/activate`.

```powershell
git clone https://github.com/atharvg565/Cybershield.git
cd Cybershield
python -m venv venv
venv\Scripts\activate
python -m pip install fastapi uvicorn streamlit scikit-learn xgboost pandas tldextract rapidfuzz requests joblib
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

### Get the data

The dataset and trained model are not stored in git.

1. Download the **PhiUSIIL Phishing URL Dataset** (UCI Machine Learning Repository) and save it as `data/PhiUSIIL_Phishing_URL_Dataset.csv`.
2. Train the URL model:
   ```powershell
   python -m train.train_url
   ```
   Use `python -m train.train_url --sample 20000` for a quick test run. This writes `models/url_model.joblib` and `eval/url_metrics.json`.
3. Threat feeds (optional): put one URL or domain per line in `data/openphish_feed.txt`, `data/urlhaus_feed.txt` or `data/custom_blocklist.txt`. Lines starting with `#` are comments.

## Run

Use two terminals, both from the project folder with the virtual environment active.

**Terminal 1: API**
```powershell
python -m uvicorn app.api:app --reload
```
Health check: http://127.0.0.1:8000/health  
Interactive docs: http://127.0.0.1:8000/docs

**Terminal 2: dashboard**
```powershell
python -m streamlit run dashboard.py
```
Opens at http://localhost:8501.

Always run through `python -m ...`. Launcher scripts such as `uvicorn.exe` break if the project folder is renamed or moved.

### API example

```
POST /analyze
{"text": "URGENT: your SBI account is locked. Verify now at http://sbi-secure.xyz/login"}
```

The response contains `risk`, `score`, `reasons`, `urls`, `text_score`, `feed_override` and `pii_masked`.

## Evaluation

Run the labelled evaluation (writes `eval/message_metrics.json`, which the dashboard's Metrics tab reads):

```powershell
python -m eval.evaluate_messages
```

It reports the confusion matrix (counts and rates), accuracy, precision, recall, specificity, F1, Matthews correlation coefficient, ROC AUC, PR AUC, a threshold sweep, per-rule hit counts, latency, and every false positive and false negative.

Current results:

| Test | Size | Result |
|---|---|---|
| URL model (PhiUSIIL, held-out 20%) | 47,074 URLs | Accuracy 99.61%, precision 99.89%, recall 99.19%, F1 0.9954 |
| Message set (written by the team) | 40 messages | Accuracy 0.75, precision 0.87, recall 0.62, F1 0.72, MCC 0.53, ROC AUC 0.79 |

Treat the URL figure as an upper bound: the public dataset is easy to separate. The message set is small and team-written, so it is a prototype benchmark, not proof of real-world performance.

## Tech stack

Python, FastAPI, Uvicorn, XGBoost, scikit-learn, pandas, tldextract, rapidfuzz, joblib, Streamlit.

## Privacy

Emails, phone numbers, card numbers and Aadhaar numbers are replaced with placeholders before a message is analysed. Raw messages are not written to disk. The dashboard history keeps only the masked text for the current session.

## Limitations

- **One trained model.** The URL detector is a trained XGBoost model. Message analysis is currently rule-based, so rephrased scams can be missed. Recall on the message set is about 62%, and the misses are mostly text-only scams with no link.
- **Dataset bias.** The URL model learned from a public dataset in which legitimate links are mostly homepages, so some genuine deep links can score as risky.
- **Hand-written rules.** The text rules, look-alike checks and score overrides were written by the team and can be evaded.
- **Small threat feed.** The bundled blocklist is a demo list. Load a real OpenPhish or URLhaus dump for meaningful coverage.
- **Scope.** English only. No live page-content analysis, WHOIS domain-age check or certificate check.
- **Local only.** Not yet deployed.

## Roadmap

- Trained text model (TF-IDF + logistic regression) to catch rephrased scams
- Trusted-domain allowlist, and reasons that quote the matched words
- Live threat feed and a larger, held-out evaluation set
- Deployment: static page on Vercel, API on Render, dashboard on a Python host
- User login, feedback on verdicts, and submission of new phishing URLs and phrases, with an admin review queue that updates the blocklist
- `requirements.txt`, tests and a hosted demo

## Team

Hackathon project, prototype built in 24 hours. Repository: https://github.com/atharvg565/Cybershield
