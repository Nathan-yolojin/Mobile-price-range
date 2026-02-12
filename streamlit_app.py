#================================
#Import necessary libraries
#================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

#================================
#Title and configuration
#================================
st.set_page_config(page_title="Mobile Price Range Predictor", page_icon="📱", layout="wide")

#================================
#Cache and helper functions
#================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("decision_tree_pipeline.pkl")   # can be pipeline or model
    feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

@st.cache_resource
def load_feature_ranges():
    return joblib.load("feature_ranges.pkl")

#================================
# Build full input DataFrame
#================================
def build_full_input(user_input: dict, feature_cols: list) -> pd.DataFrame:
    full_input = {col: user_input.get(col, 0) for col in feature_cols}
    return pd.DataFrame([full_input], columns=feature_cols)

#================================
# Get actual model from pipeline
#================================
def get_model(obj):
    if hasattr(obj, "named_steps"):
        return list(obj.named_steps.values())[-1]
    return obj

#================================
# Main app
#================================
debug = st.sidebar.checkbox("Show debug info", value=False)

#================================
# Load model and artifacts
#================================
try:
    pipeline, feature_cols = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load model files.")
    st.write("Make sure these files exist in the SAME folder as app.py:")
    st.code("decision_tree_pipeline.pkl\nfeature_columns.pkl")
    st.exception(e)
    st.stop()

try:
    feature_ranges = load_feature_ranges()
except Exception as e:
    st.error("❌ Failed to load feature_ranges.pkl")
    st.write("Export feature_ranges.pkl from your notebook and put it in the same folder as streamlit_app.py")
    st.exception(e)
    st.stop()

#================================
# Get actual model
#================================
model = get_model(pipeline)

#================================
# Debug info
#================================
if debug:
    st.write("Current folder:", os.getcwd())
    st.write("Files here:", os.listdir())
    st.write("Loaded object type:", type(pipeline))
    if hasattr(pipeline, "named_steps"):
        st.write("Pipeline steps:", pipeline.named_steps)
    st.write("Actual model type:", type(model))
    st.write("Feature ranges loaded:", feature_ranges)

st.title("📱 Mobile Price Range Predictor")

#================================
# Layout: Description and Instructions
#================================
top_left, top_right = st.columns([1, 1], gap="large")

with top_left:
    st.subheader("Description")
    st.write("""This is to predict the price range of a mobile phone based on its specifications. Based in categorical price ranges (0–3), 
             it uses features like RAM, camera quality, battery power, and network support to estimate the price category. 
             The model is a Decision Tree trained on a dataset of mobile phone specs and their corresponding price ranges.""")

with top_right:
    st.subheader("How to use")
    st.markdown("""
1. Enter the phone specifications on the **left**.
2. Click **Predict**.
3. The model outputs a **price range (0–3)** on the **right**.

**Price range meaning:**
- 0 = Low  
- 1 = Mid  
- 2 = High  
- 3 = Very High  

3G / 4G Support use 0 = No, 1 = Yes.
""")

#================================
# Fields + Labels
#================================
selected_fields = ["ram", "battery_power", "px_height", "px_width", "four_g", "three_g", "fc", "pc"]

label_map = {
    "ram": "RAM (MB)",
    "px_height": "Pixel Height",
    "px_width": "Pixel Width",
    "four_g": "4G Support",
    "three_g": "3G Support",
    "fc": "Front Camera (MP)",
    "pc": "Primary Camera (MP)",
    "battery_power": "Battery Power (mAh)"
}

range_label = {0: "Low", 1: "Mid", 2: "High", 3: "Very High"}

defaults = {
    "ram": 3000,
    "px_height": 1000,
    "px_width": 1000,
    "four_g": 1,
    "three_g": 1,
    "fc": 8,
    "pc": 12,
    "battery_power": 1500
}

#================================
# Session state initialization
#================================
st.session_state.setdefault("form_id", 0)
st.session_state.setdefault("pred", None)
st.session_state.setdefault("proba", None)
st.session_state.setdefault("conf", None)
st.session_state.setdefault("last_input_df", None)

left, right = st.columns([1, 1], gap="large")

#================================
# Input form (with validation that blocks prediction)
#================================
with left:
    st.subheader("🧾 Inputs")

    with st.form(f"input_form_{st.session_state.form_id}"):
        user_input = {}
        c1, c2 = st.columns(2)

        for i, col in enumerate(selected_fields):
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                lo, hi = feature_ranges.get(col, (0, 999999))

                if col in ["four_g", "three_g"]:
                    user_input[col] = st.selectbox(
                        label_map.get(col, col),
                        [0, 1],
                        index=int(defaults[col])
                    )
                else:
                    user_input[col] = st.number_input(
                        label_map.get(col, col),
                        value=int(defaults[col]),
                        min_value=int(lo),
                        max_value=int(hi),
                        step=1
                    )
                    st.caption(f"Dataset range: {int(lo)}–{int(hi)}")

        submitted = st.form_submit_button("Predict")

    if submitted:
    # ✅ Clear old results (so right side won't show stale prediction)
        st.session_state.pred = None
        st.session_state.proba = None
        st.session_state.conf = None
        st.session_state.last_input_df = None

    errors = []

    # ✅ Validate all inputs using feature_ranges.pkl
    for col in selected_fields:
        lo, hi = feature_ranges.get(col, (None, None))
        if lo is None:
            continue

        val = user_input[col]

        # handle int/float just in case
        if val < lo or val > hi:
            errors.append(
                f"{label_map.get(col, col)} is out of range. "
                f"Allowed: {int(lo)}–{int(hi)} (you entered {val})."
            )

    # extra logical rule
    if user_input["fc"] > user_input["pc"]:
        errors.append("Front Camera (MP) should not be higher than Primary Camera (MP).")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ✅ If valid, predict
    input_df = build_full_input(user_input, feature_cols)

    try:
        pred = int(pipeline.predict(input_df)[0])
        st.session_state.pred = pred
        st.session_state.last_input_df = input_df

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0]
            st.session_state.proba = proba
            st.session_state.conf = float(np.max(proba))
        else:
            st.session_state.proba = None
            st.session_state.conf = None

    except Exception as e:
        st.error("❌ Prediction failed (feature mismatch or preprocessing issue).")
        st.exception(e)


#================================
# Prediction display
#================================
with right:
    st.subheader("📊 Prediction")

    if st.button("Reset/Default"):
        st.session_state.pred = None
        st.session_state.proba = None
        st.session_state.conf = None
        st.session_state.last_input_df = None

        st.session_state.form_id += 1
        st.rerun()

    if st.session_state.pred is None:
        st.info("Enter inputs on the left and click **Predict**.")
    else:
        pred = st.session_state.pred
        st.success(f"✅ Predicted price range: **{pred} ({range_label.get(pred, 'Unknown')})**")

        if st.session_state.conf is not None:
            st.write(f"Prediction probability: **{st.session_state.conf:.2%}**")
            st.caption("Note: Decision Trees may output 100% probability if the input falls into a pure leaf.")

        if st.session_state.proba is not None:
            st.write("Class probabilities:")
            st.write(st.session_state.proba)

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            available = [c for c in selected_fields if c in imp.index]
            top3 = imp[available].sort_values(ascending=False).head(3)

            st.write("### 🔍 Top drivers (model)")
            for feat, score in top3.items():
                st.write(f"- **{label_map.get(feat, feat)}**: {score:.3f}")

        st.write("### Input Summary")

        if st.session_state.last_input_df is not None:
            with st.expander("Show input values (sent to model)"):
                st.dataframe(
                    st.session_state.last_input_df[selected_fields].rename(columns=label_map),
                    use_container_width=True
                )