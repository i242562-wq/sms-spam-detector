"""
space/app.py

Streamlit front-end for the SMS spam classifier.
Loads the saved sklearn Pipeline (model.joblib) and serves predictions.

Run locally:
    streamlit run app.py
"""

import json
import os

import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

EXAMPLES = {
    "Spam: prize": "WINNER!! You have been selected to receive a free £900 prize. Call 09061701461 now to claim.",
    "Spam: offer": "URGENT! Your mobile number has won a cash reward. Text CLAIM to 80062 to collect.",
    "Ham: casual": "Hey, are we still meeting for lunch tomorrow? Let me know what time works.",
    "Ham: reminder": "Don't forget to bring the notes from class, I'll pick you up at 6.",
}

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


model = load_model()

st.title("📩 SMS Spam Detector")
st.write(
    "A TF-IDF + Logistic Regression model trained in a Docker container "
    "on the UCI SMS Spam Collection. Paste a message and see if it looks like spam."
)

if model is None:
    st.error(
        "`model.joblib` was not found. Train the model first and copy it "
        "into this folder next to `app.py`."
    )
    st.stop()

# Example buttons fill the text box via session state.
if "message" not in st.session_state:
    st.session_state["message"] = ""

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state["message"] = text

message = st.text_area(
    "Message",
    key="message",
    height=130,
    placeholder="Type or paste an SMS message here...",
)

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        proba = model.predict_proba([message])[0]
        spam_p = float(proba[1])
        ham_p = float(proba[0])
        is_spam = spam_p >= 0.5

        if is_spam:
            st.error(f"🚨 Spam ({spam_p:.1%} confidence)")
        else:
            st.success(f"✅ Not spam ({ham_p:.1%} confidence)")

        st.progress(spam_p, text=f"Spam probability: {spam_p:.1%}")

        # Show which words pushed the prediction, using the model weights.
        try:
            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            vec = tfidf.transform([message])
            terms = tfidf.get_feature_names_out()
            coefs = clf.coef_[0]

            contributions = [
                (terms[i], float(vec[0, i] * coefs[i])) for i in vec.nonzero()[1]
            ]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            if contributions:
                with st.expander("Why this prediction? (top contributing words)"):
                    for word, score in contributions[:8]:
                        direction = "→ spam" if score > 0 else "→ not spam"
                        st.write(f"**{word}**  {direction}  ({score:+.3f})")
        except Exception:
            pass  # Explanation is optional; never break the app over it.

metrics = load_metrics()
if metrics:
    st.divider()
    st.subheader("Model performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Spam precision", f"{metrics['precision_spam']:.1%}")
    m3.metric("Spam recall", f"{metrics['recall_spam']:.1%}")
    m4.metric("Spam F1", f"{metrics['f1_spam']:.1%}")
    st.caption(
        f"Trained on {metrics['train_rows']:,} messages, evaluated on "
        f"{metrics['test_rows']:,}. Last trained: {metrics['trained_at']}."
    )