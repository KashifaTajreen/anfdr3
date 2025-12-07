# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

st.set_page_config(page_title="ANFDR- AI Based Nano Fertilizers Dosage Regulator - Safe nano-fertilizer recommender", layout="wide")

st.title("NanoDose — nano-fertilizer recommender (safety-first)")
st.markdown("""
**What this app does:**
- Creates a conservative synthetic dataset representing safe nano-fertilizer experiments.
- Trains a regression model to **recommend nano dosage (ml)** from environmental and plant inputs.
- Saves a locked model for farmer predictions.
\n
**Important safety notice:** This is a prototype using synthetic data. **Do not** treat numeric recommendations as definitive dosing instructions. Always follow product labels and consult an agronomist or extension service. If you are a minor, use adult supervision before handling fertilizers.
""")

# ---------------------------
# Config / constants
# ---------------------------
MODEL_PATH = "nano_model.joblib"
DEV_PASSWORD = st.secrets.get("dev_password", "devpass123") if "dev_password" in st.secrets else "devpass123"
# conservative safe max by crop (ml per plant) -- conservative values for prototype only
SAFE_MAX_BY_CROP = {
    "generic": 2.0,
    "fenugreek": 2.0,
    "mint": 1.8,
    "coriander": 2.0,
    "spinach": 1.5,
}

# ---------------------------
# Synthetic dataset generator
# ---------------------------
def generate_synthetic_dataset(n=4000, seed=42):
    """
    Generate a conservative synthetic dataset for training.
    Features:
      - sunlight_hrs (0-12)
      - temp_c (10-40)
      - soil_ph (4.0-8.0)
      - soil_moisture_pct (5-60)
      - plant_age_days (3-60)
      - base_fertilizer_g (0-10)
      - deficiency_score (0-3): higher => more deficiency
    Target:
      - nano_amount_ml (conservative safe recommendations, clipped)
    """
    rng = np.random.RandomState(seed)
    sunlight = rng.uniform(2, 10, size=n)   # hrs/day
    temp_c = rng.uniform(12, 35, size=n)
    soil_ph = rng.uniform(4.5, 7.8, size=n)
    soil_moisture = rng.uniform(8, 55, size=n)
    plant_age = rng.randint(5, 60, size=n)
    base_fert = rng.uniform(0, 8, size=n)
    deficiency = rng.choice([0,1,2,3], size=n, p=[0.5,0.25,0.15,0.1])

    # Baseline conservative dose (ml)
    # Logic (deliberately conservative and smooth):
    # - more deficiency -> higher nano dose
    # - lower soil_ph (acidic) may increase need slightly
    # - low soil moisture -> increase slightly (plant uptake issues)
    # - younger plants often need less nano; mid-age moderate
    # - higher base_fertilizer reduces nano need
    dose = 0.15 \
        + 0.45 * (deficiency / 3.0) \
        + 0.08 * np.clip((6.5 - soil_ph), 0, 2.5) \
        + 0.06 * np.clip((30 - soil_moisture) / 30, 0, 1.0) \
        + 0.02 * np.clip((30 - temp_c) / 30, -1, 1) * 0.0  # small effect removed (conservative)
    # reduce dose when base fertilizer is higher
    dose = dose * (1 - 0.06 * (base_fert / 8.0))
    # age effect: tiny reduction for very young plants
    dose = dose * (1 - 0.04 * np.clip((20 - plant_age), 0, 15) / 15)

    # Add noise and clip to 0..3.0
    noise = rng.normal(0, 0.08, size=n)
    dose = dose + noise
    dose = np.clip(dose, 0.0, 3.0)

    df = pd.DataFrame({
        "sunlight_hrs": sunlight.round(2),
        "temp_c": temp_c.round(2),
        "soil_ph": soil_ph.round(2),
        "soil_moisture_pct": soil_moisture.round(1),
        "plant_age_days": plant_age,
        "base_fertilizer_g": base_fert.round(2),
        "deficiency_score": deficiency,
        "nano_amount_ml": dose.round(3),
    })
    return df

# ---------------------------
# Train function (developer)
# ---------------------------
def train_and_save_model(df, model_path=MODEL_PATH):
    # features and label
    X = df.drop(columns=["nano_amount_ml"])
    y = df["nano_amount_ml"]

    num_cols = ["sunlight_hrs","temp_c","soil_ph","soil_moisture_pct","plant_age_days","base_fertilizer_g","deficiency_score"]
    # build simple pipeline
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("preproc", preproc), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(X[num_cols], y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    joblib.dump({"pipeline": pipe, "meta": {"n_rows": len(df)}}, model_path)
    return {"rmse": rmse, "mae": mae, "r2": r2, "n_rows": len(df)}

# ---------------------------
# Load existing model if any
# ---------------------------
model_bundle = None
if os.path.exists(MODEL_PATH):
    try:
        model_bundle = joblib.load(MODEL_PATH)
    except Exception:
        model_bundle = None

# ---------------------------
# Sidebar: Developer controls (password protected)
# ---------------------------
st.sidebar.header("Developer (training)")
with st.sidebar.expander("Developer: create & train (password required)", expanded=False):
    pw = st.text_input("Dev password", type="password")
    synth_rows = st.number_input("Synthetic dataset size", min_value=500, max_value=20000, value=4000, step=500)
    regenerate = st.button("Create synthetic dataset & Train model")
    if regenerate:
        if pw != DEV_PASSWORD:
            st.sidebar.error("Wrong developer password. Training disabled.")
        else:
            st.sidebar.info("Generating synthetic dataset (conservative)...")
            df_synth = generate_synthetic_dataset(n=int(synth_rows))
            st.sidebar.write("Sample of generated synthetic data:")
            st.sidebar.dataframe(df_synth.head(6))
            st.sidebar.info("Training model (this may take a few seconds)...")
            metrics = train_and_save_model(df_synth)
            st.sidebar.success("Training complete. Model saved.")
            st.sidebar.write(f"Rows used: {metrics['n_rows']}")
            st.sidebar.write(f"RMSE: {metrics['rmse']:.3f} ml")
            st.sidebar.write(f"MAE: {metrics['mae']:.3f} ml")
            st.sidebar.write(f"R²: {metrics['r2']:.3f}")
            # refresh model_bundle
            model_bundle = joblib.load(MODEL_PATH)

# ---------------------------
# Show info about model status
# ---------------------------
st.subheader("Model status")
# ---------------------------
# REAL EXPERIMENT DATA UPLOAD
# ---------------------------
st.header("Upload Real Experimental Data (Optional)")

real_data_file = st.file_uploader(
    "Upload your REAL experiment CSV (same format as inputs + nano_amount_ml)",
    type=["csv"]
)

real_df = None
if real_data_file:
    real_df = pd.read_csv(real_data_file)
    st.subheader("Preview of Real Experimental Data")
    st.dataframe(real_df.head())
    st.success("Real experiment data loaded successfully ✅")
# ---------------------------
# TEST MODEL ON REAL DATA
# ---------------------------
if real_df is not None and model_bundle:
    st.subheader("Test Model on Real Experimental Data")

    test_btn = st.button("Test Model on Real Data")

    if test_btn:
        pipe = model_bundle["pipeline"]

        X_real = real_df.drop(columns=["nano_amount_ml"])
        y_real = real_df["nano_amount_ml"]

        preds_real = pipe.predict(X_real)

        mse_real = mean_squared_error(y_real, preds_real)
        rmse_real = np.sqrt(mse_real)

        st.success(f"✅ RMSE on REAL data: {rmse_real:.3f} ml")

        real_df["Predicted_nano_ml"] = preds_real
        st.subheader("Predictions vs Real Dosage")
        st.dataframe(real_df.head(20))

        st.download_button(
            "Download Predictions CSV",
            real_df.to_csv(index=False),
            "real_predictions.csv",
            "text/csv"
        )
# ---------------------------
# RETRAIN USING REAL + SYNTHETIC DATA (DEV ONLY)
# ---------------------------
if real_df is not None:
    st.subheader("Retrain Model with REAL + Synthetic Data")

    retrain_btn = st.button("Retrain Model Using Real Data")

    if retrain_btn:
        if pw != DEV_PASSWORD:
            st.error("❌ Developer password required to retrain.")
        else:
            st.info("Combining synthetic + real data and retraining...")

            synthetic_df = generate_synthetic_dataset(n=2000)
            combined_df = pd.concat([synthetic_df, real_df], ignore_index=True)

            metrics = train_and_save_model(combined_df)

            st.success("✅ Model retrained using REAL experimental data")
            st.write(f"New RMSE: {metrics['rmse']:.3f} ml")
            st.write(f"Rows used: {metrics['n_rows']}")

            model_bundle = joblib.load(MODEL_PATH)

if model_bundle:
    st.success(f"Trained model found ({model_bundle.get('meta',{}).get('n_rows','?')} synthetic rows).")
else:
    st.warning("No trained model found. Developer must generate & train the model (sidebar).")

# ---------------------------
# Farmer prediction interface
# ---------------------------
st.header("Farmer: Get a recommended conservative nano dosage")
st.markdown("Enter the plant/plot data below; the recommender will give a conservative suggested nano-fertilizer dose (ml). Results are clamped to a safe maximum and accompanied by safety guidance.")

with st.form("farmer_form"):
    # crop selection optional — allows crop-specific safety cap (generic by default)
    crop = st.selectbox("Crop (optional - affects conservative cap)", options=["generic","fenugreek","mint","coriander","spinach"], index=0)
    sunlight_hrs = st.number_input("Sunlight (hrs/day)", value=6.0, min_value=0.0, step=0.1)
    temp_c = st.number_input("Temperature (°C)", value=25.0, min_value=-10.0, max_value=50.0, step=0.1)
    soil_ph = st.number_input("Soil pH", min_value=3.0, max_value=9.0, value=6.5, step=0.1)
    soil_moisture_pct = st.number_input("Soil moisture (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    plant_age_days = st.number_input("Plant age (days since sowing)", min_value=0, value=20, step=1)
    base_fertilizer_g = st.number_input("Base fertilizer applied (g per plant / per plot)", min_value=0.0, value=2.0, step=0.1)
    deficiency_score = st.selectbox("Visible deficiency score (0 = none, 3 = severe)", [0,1,2,3], index=0)
    submit = st.form_submit_button("Predict safe nano dosage")

if submit:
    if not model_bundle:
        st.error("No trained model available. Ask the developer to create & train the model (sidebar).")
    else:
        pipe = model_bundle["pipeline"]
        input_df = pd.DataFrame([{
            "sunlight_hrs": sunlight_hrs,
            "temp_c": temp_c,
            "soil_ph": soil_ph,
            "soil_moisture_pct": soil_moisture_pct,
            "plant_age_days": int(plant_age_days),
            "base_fertilizer_g": base_fertilizer_g,
            "deficiency_score": int(deficiency_score),
        }])
        # raw prediction
        pred = float(pipe.predict(input_df)[0])
        # determine conservative safe cap for selected crop
        safe_cap = SAFE_MAX_BY_CROP.get(crop, SAFE_MAX_BY_CROP["generic"])
        # Clamp prediction
        recommended = float(np.clip(pred, 0.0, safe_cap))
        # Provide a conservative lower bound (e.g., 30% of recommended) and a small safety margin
        lower_bound = max(0.0, recommended * 0.4)
        upper_bound = recommended
        st.subheader("Recommendation (conservative)")
        num_plants = st.number_input(
    "Number of plants to treat",
    min_value=1,
    value=10,
    step=1
)

total_dosage = recommended * num_plants

st.metric("✅ Nano dosage per plant", f"{recommended:.3f} ml")
st.metric("✅ Total nano dosage for all plants", f"{total_dosage:.3f} ml")
st.write(f"Conservative safe cap for crop: {safe_cap} ml (prototype default)")
st.write(f"Suggested safe range (prototype): {lower_bound:.3f} — {upper_bound:.3f} ml")
st.write(f"Model raw prediction (before conservative clamping): {pred:.3f} ml")
# show explanation and guidance
st.warning("""
**Safety & usage guidance (read carefully):**
- This recommendation is a conservative, AI-assisted suggestion based on a synthetic dataset. It does **not** replace:
  - the product label / manufacturer instructions,
  - or expert agronomist advice.
- Start with a small pilot (potted plants) and test the recommendation on a few plants first.
- Do not exceed the conservative cap shown above.
- If you are a minor: get adult supervision before applying any fertilizer. Wear PPE (gloves, mask) and follow safety procedures.
- If in doubt, consult your local agricultural extension or a certified agronomist.
""")
        # optional: show model confidence proxy (inverse of scaled RMSE if available)
try:
     # load metrics if saved (not guaranteed)
     meta = model_bundle.get("meta", {})
     n_rows = meta.get("n_rows", None)
     if n_rows:
          st.caption(f"Model was trained on {n_rows} synthetic samples (prototype).")
except Exception:
     pass

# ---------------------------
# Footer: dataset download (developer)
# ---------------------------
st.markdown("---")
st.caption("This app is a prototype that uses conservative synthetic data to demonstrate how a nano-fertilizer recommender could work. For production use, collect verified experiment data and retrain the model under expert supervision.")

if st.checkbox("Download synthetic training sample (developer use)"):
    sample = generate_synthetic_dataset(n=500)
    st.download_button("Download synthetic_sample.csv", sample.to_csv(index=False), "synthetic_sample.csv", "text/csv")
