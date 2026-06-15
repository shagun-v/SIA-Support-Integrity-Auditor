import streamlit as st
import pandas as pd
import numpy as np
import json, os, re, pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import QuantileTransformer

st.set_page_config(page_title='Support Integrity Auditor (SIA)', page_icon='🔍', layout='wide')

PRIORITY_ORD = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
WEIGHTS = {'nlp': 0.30, 'resolution': 0.25, 'embedding': 0.20, 'zeroshot': 0.25}

SEVERITY_LEXICON = {
    'outage':4,'system down':4,'cannot access':4,'security breach':4,
    'data breach':4,'account hacked':4,'payment failed':4,'fraud':4,
    'urgent':3,'error':3,'crash':3,'failed':3,'blocked':3,'asap':3,
    'slow':2,'delayed':2,'issue':2,'problem':2,'wrong':2,
    'question':1,'inquiry':1,'help':1,'feedback':1,
}

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r"http\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

def compute_nlp_severity(text):
    best, cum = 0.0, 0.0
    for phrase, w in SEVERITY_LEXICON.items():
        if phrase in text:
            best = max(best, w)
            cum += w
    damped = min(cum / 10.0, 4.0)
    return float(np.clip((0.6 * best + 0.4 * damped) / 4.0, 0, 1))

@st.cache_resource
def load_models():
    artifact_dir = Path("model_artifacts")
    models = {}
    for name in ["xgb_model","tfidf_vectorizer","scaler",
                 "label_encoder_category","label_encoder_channel",
                 "quantile_transformer","kmeans_cluster"]:
        path = artifact_dir / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models

st.title("🔍 Support Integrity Auditor (SIA)")
st.caption("MARS Open Projects 2026 — Detecting Priority Mismatch in Customer Support Tickets")
models = load_models()

tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket", "📂 Batch Upload", "📊 Dashboard"])

with tab1:
    st.subheader("Single Ticket Mismatch Analysis")
    col1, col2 = st.columns(2)
    with col1:
        subject     = st.text_input("Ticket Subject", "Cannot access my account")
        description = st.text_area("Ticket Description", "Locked out for 3 days.")
        priority    = st.selectbox("Assigned Priority", ["Low","Medium","High","Critical"])
    with col2:
        category  = st.selectbox("Issue Category", ["Technical","Billing","Account","General Inquiry","Fraud"])
        channel   = st.selectbox("Ticket Channel", ["Chat","Email","Web Form"])
        res_hours = st.slider("Resolution Time (Hours)", 1, 120, 48)
        sat_score = st.slider("Satisfaction Score", 1, 5, 3)
    if st.button("🔍 Analyse Ticket", type="primary"):
        clean = clean_text(subject + " " + description)
        nlp_s = compute_nlp_severity(clean)
        qt = models.get("quantile_transformer")
        res_s = float(qt.transform([[res_hours]])[0][0]) if qt else 0.5
        final = WEIGHTS["nlp"]*nlp_s + WEIGHTS["resolution"]*res_s + WEIGHTS["embedding"]*0.5 + WEIGHTS["zeroshot"]*nlp_s
        q = [0.25, 0.50, 0.75]
        inferred = "Low" if final<=q[0] else "Medium" if final<=q[1] else "High" if final<=q[2] else "Critical"
        p_ord = PRIORITY_ORD[priority]
        i_ord = PRIORITY_ORD[inferred]
        delta = i_ord - p_ord
        mismatch = abs(delta) >= 1
        if mismatch:
            mtype = "Hidden Crisis" if delta > 0 else "False Alarm"
            st.error(f"⚠️ MISMATCH DETECTED — {mtype}")
        else:
            st.success("✅ Priority Consistent — No Mismatch")
        ca, cb, cc = st.columns(3)
        ca.metric("Assigned", priority)
        cb.metric("Inferred", inferred, delta=f"{delta:+d} level(s)")
        cc.metric("Score", f"{final:.3f}")

with tab2:
    st.subheader("Batch CSV Analysis")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_b = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_b):,} rows")
        st.dataframe(df_b.head())

with tab3:
    st.subheader("Priority Mismatch Dashboard")
    dpath = Path("mismatch_dossiers.json")
    if dpath.exists():
        with open(dpath) as f:
            dossiers = json.load(f)
        dd = pd.DataFrame([{"mismatch_type":d["mismatch_type"],"confidence":float(d["confidence"])} for d in dossiers])
        st.metric("Total Mismatches", len(dd))
        fig = px.pie(dd, names="mismatch_type", title="Mismatch Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the full notebook first to generate dossiers.")
