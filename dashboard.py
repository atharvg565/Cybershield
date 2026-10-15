import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from app.privacy import mask_pii

API_URL = "http://127.0.0.1:8000/analyze"
ROOT = Path(__file__).resolve().parent
COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
ADVICE = {
    "High": "Do not click any link or reply. Verify through the official app or phone number, then report the message.",
    "Medium": "Be careful. Check the sender independently before acting on this message.",
    "Low": "No strong warning signs. Still verify unexpected requests for money or personal details.",
}
EXAMPLES = {
    "Google (trusted site)": "Check this out: https://google.com",
    "Flipkart order (legitimate)": "Your order has shipped. Track it at https://www.flipkart.com/orders",
    "SBI phishing link (in threat feed)": "URGENT: your SBI account is locked. Verify now at http://sbi-secure.xyz/login",
    "Lookalike brand fine (paypa1)": "Pay the Rs 500 fine within 24 hours or face arrest: http://paypa1.com/pay",
    "Advance-fee scam (20 million)": "20 million credited to your account. Pay 1 million to be able to withdraw.",
    "Personal data (masking demo)": "Call me on +91 98765 43210 or mail ravi.k@gmail.com",
}

st.set_page_config(page_title="CyberShield", page_icon="🛡️", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "msg" not in st.session_state:
    st.session_state.msg = ""


def load_example():
    choice = st.session_state.example
    if choice in EXAMPLES:
        st.session_state.msg = EXAMPLES[choice]


def call_api(text: str) -> dict:
    r = requests.post(API_URL, json={"text": text}, timeout=10)
    r.raise_for_status()
    return r.json()


def load_json(name: str):
    path = ROOT / "eval" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def record(text: str, d: dict):
    masked, _ = mask_pii(text)
    st.session_state.history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Risk": d["risk"],
        "Score": d["score"],
        "Reasons": len(d["reasons"]),
        "Message (masked)": masked[:80] + ("..." if len(masked) > 80 else ""),
    })


def render_result(d: dict):
    color = COLORS[d["risk"]]
    st.markdown(f"<h2 style='color:{color};margin-bottom:0'>{d['risk']} risk</h2>", unsafe_allow_html=True)
    st.write(f"Risk score **{round(d['score'] * 100)}%**")
    st.progress(min(max(d["score"], 0.0), 1.0))

    if d["feed_override"]:
        st.error("A link in this message is on a threat-intelligence blocklist.")

    if d.get("critical_rules"):
        st.markdown("**Critical rules triggered**")
        for c in d["critical_rules"]:
            st.write(f"🚨 {c}")

    st.markdown("**Why it was flagged**")
    if d["reasons"]:
        for r in d["reasons"]:
            st.write(f"• {r}")
    else:
        st.success("No warning signs found.")

    if d["urls"]:
        st.markdown("**Links checked**")
        st.dataframe(pd.DataFrame([{
            "Link": u["url"],
            "Trusted domain": "Yes" if u.get("trusted_domain") else "No",
            "ML phishing probability": u["ml_score"],
            "Rule hits": len(u["reasons"]),
        } for u in d["urls"]]), hide_index=True)

    if any(d["pii_masked"].values()):
        st.caption(f"Personal data masked before analysis: {d['pii_masked']}")

    st.markdown("**What to do**")
    st.write(ADVICE[d["risk"]])


def render_message_metrics(m: dict):
    st.subheader("Message test set")
    st.caption(
        f"{m['total']} messages ({m['positives_scam']} scam, {m['negatives_legit']} legitimate). "
        f"Decision threshold: score above {m['decision_threshold']} is flagged. "
        "These messages were written by the team, so treat the numbers as a prototype benchmark."
    )

    a, b, c, d_, e = st.columns(5)
    a.metric("Accuracy", f"{m['accuracy']:.1%}", help="Share of all decisions that were correct.")
    b.metric("Precision", f"{m['precision']:.1%}", help="Of messages flagged, the share that were really scams.")
    c.metric("Recall (TPR)", f"{m['recall']:.1%}", help="Of real scams, the share that were caught.")
    d_.metric("Specificity (TNR)", f"{m['specificity']:.1%}", help="Of legitimate messages, the share correctly passed.")
    e.metric("F1", f"{m['f1']:.2f}", help="Balance of precision and recall.")

    f, g, h = st.columns(3)
    f.metric("MCC", f"{m['matthews_corrcoef']:.2f}", help="Matthews correlation: -1 is worst, 0 is chance, 1 is perfect. Reliable on imbalanced data.")
    g.metric("ROC AUC", f"{m['roc_auc']:.2f}" if m["roc_auc"] is not None else "n/a", help="Chance that a random scam scores higher than a random legitimate message.")
    h.metric("PR AUC", f"{m['pr_auc']:.2f}" if m["pr_auc"] is not None else "n/a", help="Precision-recall area. Higher is better when scams are the minority class.")

    cm = m["confusion_matrix"]
    rates = m["confusion_matrix_rates"]
    st.markdown("**Confusion matrix (counts)**")
    st.table(pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual: legitimate", "Actual: scam"],
        columns=["Predicted: safe", "Predicted: flagged"],
    ))
    st.markdown("**Confusion matrix (rates)**")
    st.table(pd.DataFrame(
        [[rates["true_negative_rate"], rates["false_positive_rate"]],
         [rates["false_negative_rate"], rates["true_positive_rate"]]],
        index=["Actual: legitimate", "Actual: scam"],
        columns=["Predicted: safe", "Predicted: flagged"],
    ).round(3))
    st.caption(
        "TN = legitimate passed. FP = legitimate flagged (false alarm). "
        "FN = scam missed (dangerous). TP = scam caught."
    )

    sweep = pd.DataFrame(m["threshold_sweep"])
    if not sweep.empty:
        st.markdown("**ROC curve (true positive rate vs false positive rate)**")
        roc = sweep.sort_values("fpr").set_index("fpr")[["tpr_recall"]]
        st.line_chart(roc)
        st.markdown("**Precision and recall across thresholds**")
        st.line_chart(sweep.set_index("threshold")[["precision", "tpr_recall"]].rename(columns={"tpr_recall": "recall"}))
        st.caption("Lowering the threshold catches more scams (recall) but raises false alarms. Our system uses 0.35.")

    hits = m.get("rule_hits", {})
    if hits:
        st.markdown("**Rule hits: which rules fire on scams vs legitimate messages**")
        hit_df = pd.DataFrame(
            [{"Rule": k, "On scams": v["scam"], "On legitimate": v["legit"]} for k, v in hits.items()]
        ).sort_values("On scams", ascending=False)
        st.dataframe(hit_df, hide_index=True, use_container_width=True)

    st.markdown(f"**Scams missed (false negatives): {len(m['false_negatives'])}**")
    for x in m["false_negatives"]:
        st.write(f"• [{x['risk']} {x['score']}] {x['text']}")

    st.markdown(f"**Legitimate messages flagged (false positives): {len(m['false_positives'])}**")
    for x in m["false_positives"]:
        st.write(f"• [{x['risk']} {x['score']}] {x['text']}")

    st.caption(f"Average scoring latency: {m['avg_latency_ms']} ms per message.")


def render_url_metrics(u: dict):
    st.subheader("URL model (PhiUSIIL public dataset)")
    st.write(
        f"Held-out test URLs: {u['test_size']:,} · Accuracy {u['accuracy']:.2%} · "
        f"Precision {u['precision']:.2%} · Recall {u['recall']:.2%} · F1 {u['f1']:.4f}"
    )
    cm = u["confusion_matrix"]
    st.write(f"Confusion: TN={cm['tn']:,} FP={cm['fp']:,} FN={cm['fn']:,} TP={cm['tp']:,}")
    st.caption("Public dataset, so this is an upper bound. Real-world performance is lower, especially on new attacks.")


st.title("🛡️ CyberShield")
st.caption("Explainable phishing and fraud detection for SMS, email and links")

tab_check, tab_history, tab_metrics = st.tabs(["Check a message", "History", "Metrics"])

with tab_check:
    left, right = st.columns([1, 1.4])
    with left:
        st.selectbox(
            "Try an example",
            ["Choose an example..."] + list(EXAMPLES),
            key="example",
            on_change=load_example,
        )
        st.text_area("Message or link", key="msg", height=180)
        go = st.button("Check message", type="primary")

    with right:
        if go:
            text = st.session_state.msg.strip()
            if not text:
                st.warning("Paste a message first.")
            else:
                try:
                    d = call_api(text)
                    record(text, d)
                    render_result(d)
                except requests.RequestException:
                    st.error(
                        "Could not reach the CyberShield API. In a terminal, run: "
                        "python -m uvicorn app.api:app --reload"
                    )
        else:
            st.info("The result appears here: risk level, score, the exact reasons, and each link checked.")

with tab_history:
    hist = st.session_state.history
    if not hist:
        st.info("No checks yet in this session.")
    else:
        df = pd.DataFrame(hist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Checks", len(df))
        c2.metric("High", int((df["Risk"] == "High").sum()))
        c3.metric("Medium", int((df["Risk"] == "Medium").sum()))
        c4.metric("Low", int((df["Risk"] == "Low").sum()))
        st.bar_chart(df["Risk"].value_counts())
        st.dataframe(df, hide_index=True)
        st.download_button("Download history as CSV", df.to_csv(index=False), "cybershield_history.csv", "text/csv")

with tab_metrics:
    m = load_json("message_metrics.json")
    u = load_json("url_metrics.json")
    if m is None:
        st.warning("No message evaluation found. Run: python -m eval.evaluate_messages")
    else:
        render_message_metrics(m)
    st.divider()
    if u is not None:
        render_url_metrics(u)