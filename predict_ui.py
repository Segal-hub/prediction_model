# import os
# import numpy as np
# import pandas as pd
# import streamlit as st
# import joblib
# from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# MODEL_FILE = "model_behavior_aware.pkl"
# THRESHOLD_FILE = "threshold_behavior_aware.npy"
# DEFAULT_YESTERDAY = "yesterday.csv"       
# DEFAULT_HISTORY = "one_trainset.csv"  

# st.set_page_config(page_title="IoT Disconnection Predictor — Behavior-Aware", layout="wide")
# st.title("ONE Devices Disconnection Predictor — Behavior-Aware Model")

# def load_csv(uploaded, default_path, name):
#     """
#     If 'uploaded' is provided, load it.
#     Else, if default exists for HISTORY -> use it.
#     Else stop with error.
#     """
#     if uploaded is not None:
#         df = pd.read_csv(uploaded)
#         st.success(f"Loaded uploaded {name}")
#         return df
#     elif os.path.exists(default_path):
#         df = pd.read_csv(default_path)
#         st.info(f"Using default {name}: {default_path}")
#         return df
#     else:
#         st.error(f"Missing {name}. Upload or place {default_path}")
#         st.stop()

# def normalize_id(df):
#     if "DeviceReferenceId" in df.columns:
#         df["DeviceReferenceId"] = df["DeviceReferenceId"].astype(str)
#     elif "DeviceId" in df.columns:
#         df["DeviceReferenceId"] = df["DeviceId"].astype(str)
#     else:
#         st.error("Missing DeviceReferenceId or DeviceId column.")
#         st.stop()
#     return df

# def to_datetime_safe(df, cols):
#     for c in cols:
#         if c in df.columns:
#             df[c] = pd.to_datetime(df[c], errors="coerce")
#     return df

# def ensure_country_code(df):
#     if "CountryCode" not in df.columns:
#         df["CountryCode"] = "<UNK>"
#     df["CountryCode"] = df["CountryCode"].astype(str).fillna("<UNK>")
#     return df

# def add_behavior_features(df_current, df_history_healthy):
#     # Build healthy baselines per device
#     baselines = (
#         df_history_healthy.groupby("DeviceReferenceId")
#         .agg({
#             "AvgBatteryVoltage": ["mean", "std"],
#             "AvgRSSI": ["mean", "std"],
#             "HeartbeatCount": ["mean", "std"]
#         })
#     )
#     baselines.columns = [
#         "BatteryMean", "BatteryStd",
#         "RSSIMean", "RSSIStd",
#         "HeartbeatMean", "HeartbeatStd"
#     ]
#     baselines = baselines.reset_index()

#     df = df_current.merge(baselines, on="DeviceReferenceId", how="left")

#     # Avoid division by zero in std
#     for col in ["BatteryStd", "RSSIStd", "HeartbeatStd"]:
#         df[col] = df[col].replace(0, np.nan)

#     # Z-scores (clip & fill like training)
#     df["BatteryZ"]  = (df["AvgBatteryVoltage"] - df["BatteryMean"])   / df["BatteryStd"]
#     df["RSSIZ"]     = (df["AvgRSSI"]          - df["RSSIMean"])       / df["RSSIStd"]
#     df["HeartbeatZ"]= (df["HeartbeatCount"]   - df["HeartbeatMean"])  / df["HeartbeatStd"]
#     df[["BatteryZ","RSSIZ","HeartbeatZ"]] = df[["BatteryZ","RSSIZ","HeartbeatZ"]].clip(-5,5).fillna(0)

#     return df

# # ---------- Column coloring helpers ----------
# RED_BG = "background-color: #ffd6d6"     # light red
# ORANGE_BG = "background-color: #ffe9d6"  # light orange
# NO_BG = ""

# def style_prob(series: pd.Series):
#     """Prob >90 red, >80 orange (series is %)."""
#     return [
#         RED_BG if v > 90 else ORANGE_BG if v > 80 else NO_BG
#         for v in series
#     ]

# def style_battery(series: pd.Series):
#     """Battery <3.6 red, <3.8 orange."""
#     return [
#         RED_BG if v < 3600 else ORANGE_BG if v < 3800 else NO_BG
#         for v in series
#     ]

# def style_rssi(series: pd.Series):
#     """RSSI >50 red, >40 orange."""
#     return [
#         RED_BG if v > 50 else ORANGE_BG if v > 40 else NO_BG
#         for v in series
#     ]

# # FILE INPUTS
# yesterday_file = st.file_uploader("📂 Upload yesterday's data (CSV) — required", type=["csv"], key="yesterday")
# history_file   = st.file_uploader("📜 Upload historical data (CSV: trainset or healthy subset)", type=["csv"], key="history")
 
# # *** IMPORTANT: require uploading Yesterday CSV***
# if not yesterday_file:
#     st.info("Please upload the Yesterday CSV to run predictions.")
#     st.stop()

# # Yesterday MUST be uploaded; History keeps old behavior (default allowed)
# df_current = pd.read_csv(yesterday_file)
# df_history = load_csv(history_file, DEFAULT_HISTORY, "historical data")

# # Normalize IDs
# df_current = normalize_id(df_current)
# df_history = normalize_id(df_history)

# # Parse times & metadata
# df_current = to_datetime_safe(df_current, ["created_at", "Day"])
# if "created_at" in df_current.columns and "Day" in df_current.columns:
#     df_current["DeviceAgeDays"] = (df_current["Day"] - df_current["created_at"]).dt.days.clip(lower=0)
# else:
#     df_current["DeviceAgeDays"] = 0

# # Use the same extraction as training (expand=False to ensure a Series)
# if "SwVersion" in df_current.columns:
#     df_current["SwVersionGroup"] = df_current["SwVersion"].astype(str).str.extract(r"^(\d+\.\d+)", expand=False).fillna("<UNK>")
# else:
#     df_current["SwVersionGroup"] = "<UNK>"

# df_current = ensure_country_code(df_current)

# # Filter history to healthy only (as in training baseline)
# if "DisconnectNext24h" in df_history.columns:
#     df_history_healthy = df_history[df_history["DisconnectNext24h"] == 0].copy()
# else:
#     # If label is missing in the uploaded history, assume all rows are healthy
#     df_history_healthy = df_history.copy()

# # Add behavior-aware features (z-scores)
# df = add_behavior_features(df_current, df_history_healthy)

# # MODEL & THRESHOLD
# try:
#     model = joblib.load(MODEL_FILE)
# except Exception as e:
#     st.error(f"Failed to load model '{MODEL_FILE}': {e}")
#     st.stop()

# try:
#     thr_arr = np.load(THRESHOLD_FILE, allow_pickle=True)
#     # handle scalar .npy or 1-element array
#     threshold = float(thr_arr.item() if hasattr(thr_arr, "item") and thr_arr.shape == () else (thr_arr[0] if np.size(thr_arr) else thr_arr))
# except Exception as e:
#     st.warning(f"Could not load threshold file '{THRESHOLD_FILE}': {e}. Falling back to 0.5.")
#     threshold = 0.5

# st.info(f"Model loaded — Threshold = {threshold:.4f}")

# # FINAL FEATURE SET (EXACTLY AS TRAINING)
# feature_cols = [
#     "AvgBatteryVoltage", "AvgRSSI", "HeartbeatCount", "DeviceAgeDays",
#     "BatteryZ", "RSSIZ", "HeartbeatZ",
#     "CountryCode", "SwVersionGroup"
# ]
# categorical_cols = ["CountryCode", "SwVersionGroup"]

# # Make sure required columns exist
# missing = [c for c in feature_cols if c not in df.columns]
# if missing:
#     st.error(f"Missing required feature columns: {missing}")
#     st.stop()

# # Categorical dtypes like training
# for c in categorical_cols:
#     df[c] = df[c].astype("category")

# X = df[feature_cols].copy()

# # Match the model's internal feature order if available (CalibratedClassifierCV → LGBMClassifier → booster)
# try:
#     booster = model.calibrated_classifiers_[0].base_estimator_.booster_
#     trained_cols = booster.feature_name()
#     if set(trained_cols) == set(X.columns):
#         X = X[trained_cols]
# except Exception:
#     # If anything goes wrong, we stick to our defined order (which matches training script)
#     pass

# # PREDICT
# df["Prediction Propability"] = model.predict_proba(X)[:, 1]  # keep original misspelling used elsewhere
# df["PredLabel"] = (df["Prediction Propability"] >= threshold).astype(int)

# st.subheader("Predicted Devices at Risk")

# # Percentage for readability & coloring
# df["Prediction Propability"] = (df["Prediction Propability"] * 100).round(1)

# risk_cols = [
#     "DeviceReferenceId",
#     "Prediction Propability",
#     "AvgBatteryVoltage",
#     "AvgRSSI",
#     "HeartbeatCount",
# ]

# risk_df = (
#     df.loc[df["PredLabel"] == 1, risk_cols]
#       .sort_values("Prediction Propability", ascending=False)
#       .reset_index(drop=True)
# )

# st.metric("Predicted count", int(len(risk_df)))

# # Apply per-column conditional formatting
# styled = (
#     risk_df.style
#     .apply(style_prob, subset=["Prediction Propability"])
#     .apply(style_battery, subset=["AvgBatteryVoltage"])
#     .apply(style_rssi, subset=["AvgRSSI"])
#     .format({
#         "Prediction Propability": "{:.1f}%",
#         "AvgBatteryVoltage": "{:.3f}",
#         "AvgRSSI": "{:.0f}",
#         "HeartbeatCount": "{:.0f}",
#     })
# )

# st.dataframe(styled, use_container_width=True)

# with st.expander("🗂️ Color legend", expanded=False):
#     st.markdown(
#         """
# - **Probability**: > **90%** = red, > **80%** = orange  
# - **Battery (mV)**: < **3600** = red, < **3800** = orange  
# - **RSSI (abs)**: > **50** = red, > **40** = orange
#         """
#     )

# # Evaluate With Ground Truth
# st.markdown("---")
# st.subheader("Evaluate With Ground Truth")
# truth_file = st.file_uploader("📂 Upload Ground Truth CSV (DeviceReferenceId of actual disconnects)", type=["csv"], key="truth")
# if truth_file:
#     truth_df = pd.read_csv(truth_file)
#     truth_df = normalize_id(truth_df)

#     pred_ids = set(df.loc[df["PredLabel"] == 1, "DeviceReferenceId"])
#     truth_ids = set(truth_df["DeviceReferenceId"])
#     all_ids = set(df["DeviceReferenceId"])

#     tp = pred_ids & truth_ids
#     fp = pred_ids - truth_ids
#     fn = truth_ids - pred_ids
#     tn = all_ids - (tp | fp | fn)

#     y_true = [1 if i in truth_ids else 0 for i in all_ids]
#     y_pred = [1 if i in pred_ids else 0 for i in all_ids]

#     precision = precision_score(y_true, y_pred) if (sum(y_pred) > 0) else 0.0
#     recall = recall_score(y_true, y_pred) if (sum(y_true) > 0) else 0.0
#     f1 = f1_score(y_true, y_pred) if (sum(y_pred) > 0 and sum(y_true) > 0) else 0.0

#     st.write(f"TP: {len(tp)} | FP: {len(fp)} | FN: {len(fn)} | TN: {len(tn)}")
#     st.write(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

#     # Note: y_scores here are % values (0-100), consistent with "Prediction Propability" after scaling.
#     probs_map = dict(zip(df["DeviceReferenceId"], df["Prediction Propability"]))
#     y_scores = [probs_map[i] for i in all_ids]
#     prec, rec, thr = precision_recall_curve(y_true, y_scores)

#     best_thr, best_f1 = threshold * 100.0, 0.0  # compare in % space
#     for p, r, t in zip(prec, rec, thr):
#         if p >= 0.45:
#             f1_tmp = 2 * (p * r) / (p + r + 1e-9)
#             if f1_tmp > best_f1:
#                 best_thr, best_f1 = t, f1_tmp

#     if best_f1 > 0:
#         st.info(f"Suggested higher precision threshold: {best_thr:.3f} (F1={best_f1:.3f})")

######################################### with additional features
# import os
# import numpy as np
# import pandas as pd
# import streamlit as st
# import joblib
# from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# MODEL_FILE = "model_behavior_aware.pkl"
# THRESHOLD_FILE = "threshold_behavior_aware.npy"

# DEFAULT_YESTERDAY = "yesterday.csv"
# DEFAULT_HISTORY = "one_trainset.csv"

# st.set_page_config(page_title="IoT Disconnection Predictor — Behavior-Aware", layout="wide")
# st.title("ONE Devices Disconnection Predictor — Behavior-Aware Model")


# def load_csv(uploaded, default_path, name):
#     if uploaded is not None:
#         df = pd.read_csv(uploaded)
#         st.success(f"Loaded uploaded {name}")
#         return df
    
#     elif os.path.exists(default_path):
#         df = pd.read_csv(default_path)
#         st.info(f"Using default {name}: {default_path}")
#         return df
    
#     st.error(f"Missing {name}. Upload or place {default_path}")
#     st.stop()


# def normalize_id(df):
#     if "DeviceReferenceId" in df.columns:
#         df["DeviceReferenceId"] = df["DeviceReferenceId"].astype(str)
#     elif "DeviceId" in df.columns:
#         df["DeviceReferenceId"] = df["DeviceId"].astype(str)
#     else:
#         st.error("Missing DeviceReferenceId or DeviceId column.")
#         st.stop()
#     return df


# def to_datetime_safe(df, cols):
#     for c in cols:
#         if c in df.columns:
#             df[c] = pd.to_datetime(df[c], errors="coerce")
#     return df


# def ensure_country(df):
#     if "Country" not in df.columns:
#         df["Country"] = "<UNK>"
#     df["Country"] = df["Country"].astype(str).fillna("<UNK>")
#     return df


# def add_behavior_features(df_current, df_history_healthy):
#     baselines = (
#         df_history_healthy.groupby("DeviceReferenceId")
#         .agg({
#             "AvgBatteryVoltage": ["mean", "std"],
#             "AvgRSSI": ["mean", "std"],
#             "HeartbeatCount": ["mean", "std"],
#         })
#     )
#     baselines.columns = [
#         "BatteryMean", "BatteryStd",
#         "RSSIMean", "RSSIStd",
#         "HeartbeatMean", "HeartbeatStd"
#     ]
#     baselines = baselines.reset_index()

#     df = df_current.merge(baselines, on="DeviceReferenceId", how="left")

#     for c in ["BatteryStd", "RSSIStd", "HeartbeatStd"]:
#         df[c] = df[c].replace(0, np.nan)

#     df["BatteryZ"]  = (df["AvgBatteryVoltage"] - df["BatteryMean"]) / df["BatteryStd"]
#     df["RSSIZ"]     = (df["AvgRSSI"] - df["RSSIMean"]) / df["RSSIStd"]
#     df["HeartbeatZ"]= (df["HeartbeatCount"] - df["HeartbeatMean"]) / df["HeartbeatStd"]

#     df[["BatteryZ", "RSSIZ", "HeartbeatZ"]] = (
#         df[["BatteryZ", "RSSIZ", "HeartbeatZ"]].clip(-5, 5).fillna(0)
#     )

#     return df

# # ========== Coloring Helpers ==========
# RED_BG = "background-color: #ffd6d6"
# ORANGE_BG = "background-color: #ffe9d6"
# NO_BG = ""

# def style_prob(series):
#     return [RED_BG if v > 90 else ORANGE_BG if v > 80 else NO_BG for v in series]

# def style_battery(series):
#     return [RED_BG if v < 3600 else ORANGE_BG if v < 3800 else NO_BG for v in series]

# def style_rssi(series):
#     return [RED_BG if v > 50 else ORANGE_BG if v > 40 else NO_BG for v in series]

# yesterday_file = st.file_uploader("📂 Upload yesterday's data (CSV) — required", type=["csv"], key="yesterday")
# history_file = st.file_uploader("Upload historical data (CSV: trainset or healthy subset)", type=["csv"], key="history")

# if not yesterday_file:
#     st.info("Please upload the Yesterday CSV.")
#     st.stop()

# df_current = pd.read_csv(yesterday_file)
# df_history = load_csv(history_file, DEFAULT_HISTORY, "historical data")

# df_current = normalize_id(df_current)
# df_history = normalize_id(df_history)

# df_current = to_datetime_safe(df_current, ["created_at", "Day"])

# if "created_at" in df_current.columns and "Day" in df_current.columns:
#     df_current["DeviceAgeDays"] = (df_current["Day"] - df_current["created_at"]).dt.days.clip(lower=0)
# else:
#     df_current["DeviceAgeDays"] = 0

# df_current["SwVersionGroup"] = df_current["SwVersion"].astype(str).str.extract(
#     r"^(\d+\.\d+)", expand=False
# ).fillna("<UNK>")

# df_current = ensure_country(df_current)

# if "DisconnectNext24h" in df_history.columns:
#     df_history_healthy = df_history[df_history["DisconnectNext24h"] == 0].copy()
# else:
#     df_history_healthy = df_history.copy()

# df = add_behavior_features(df_current, df_history_healthy)


# try:
#     model = joblib.load(MODEL_FILE)
# except Exception as e:
#     st.error(f"Failed to load model: {e}")
#     st.stop()

# try:
#     thr_arr = np.load(THRESHOLD_FILE, allow_pickle=True)
#     threshold = float(thr_arr[0])
# except Exception:
#     st.warning("Could not load threshold file — using 0.5.")
#     threshold = 0.5

# st.info(f"Model loaded — Threshold = {threshold:.4f}")


# feature_cols = [
#     "AvgBatteryVoltage", "AvgRSSI", "HeartbeatCount", "DeviceAgeDays",
#     "BatteryZ", "RSSIZ", "HeartbeatZ",
#     "Country", "SwVersionGroup"
# ]

# categorical_cols = ["Country", "SwVersionGroup"]

# missing = [c for c in feature_cols if c not in df.columns]
# if missing:
#     st.error(f"Missing required columns: {missing}")
#     st.stop()

# for c in categorical_cols:
#     df[c] = df[c].astype("category")

# X = df[feature_cols]


# df["Prediction Propability"] = model.predict_proba(X)[:, 1]
# df["PredLabel"] = (df["Prediction Propability"] >= threshold).astype(int)
# df["Prediction Propability"] = (df["Prediction Propability"] * 100).round(1)

# st.subheader("Predicted Devices at Risk")

# risk_cols = [
#     "DeviceReferenceId",
#     "Country",
#     "BU",
#     "SwVersion",
#     "Prediction Propability",
#     "AvgBatteryVoltage",
#     "AvgRSSI",
#     "HeartbeatCount",
#     "Is FloLive",
#     "Last transmission"
# ]

# risk_df = (
#     df.loc[df["PredLabel"] == 1, risk_cols]
#       .sort_values("Prediction Propability", ascending=False)
#       .reset_index(drop=True)
# )

# st.metric("Predicted count", len(risk_df))

# styled = (
#     risk_df.style
#     .apply(style_prob, subset=["Prediction Propability"])
#     .apply(style_battery, subset=["AvgBatteryVoltage"])
#     .apply(style_rssi, subset=["AvgRSSI"])
#     .format({
#         "Prediction Propability": "{:.1f}%",
#         "AvgBatteryVoltage": "{:.3f}",
#         "AvgRSSI": "{:.0f}",
#         "HeartbeatCount": "{:.0f}",
#     })
# )

# st.dataframe(styled, use_container_width=True)


# with st.expander("🗂️ Color legend"):
#     st.markdown("""
# - **Probability**: > 90% red, > 80% orange  
# - **Battery**: < 3600mV red, < 3800mV orange  
# - **RSSI**: > 50 red, > 40 orange  
# """)


# st.markdown("---")
# st.subheader("Evaluate With Ground Truth")
# truth_file = st.file_uploader("📂 Upload Ground Truth CSV (DeviceReferenceId)", type=["csv"], key="truth")

# if truth_file:
#     truth_df = pd.read_csv(truth_file)
#     truth_df = normalize_id(truth_df)

#     pred_ids = set(df.loc[df["PredLabel"] == 1, "DeviceReferenceId"])
#     truth_ids = set(truth_df["DeviceReferenceId"])
#     all_ids = set(df["DeviceReferenceId"])

#     tp = pred_ids & truth_ids
#     fp = pred_ids - truth_ids
#     fn = truth_ids - pred_ids
#     tn = all_ids - (tp | fp | fn)

#     y_true = [1 if i in truth_ids else 0 for i in all_ids]
#     y_pred = [1 if i in pred_ids else 0 for i in all_ids]

#     precision = precision_score(y_true, y_pred) if sum(y_pred) else 0.0
#     recall = recall_score(y_true, y_pred) if sum(y_true) else 0.0
#     f1 = f1_score(y_true, y_pred) if sum(y_pred) and sum(y_true) else 0.0

#     st.write(f"TP: {len(tp)} | FP: {len(fp)} | FN: {len(fn)} | TN: {len(tn)}")
#     st.write(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

#     probs_map = dict(zip(df["DeviceReferenceId"], df["Prediction Propability"]))
#     y_scores = [probs_map[i] for i in all_ids]

#     prec, rec, thr = precision_recall_curve(y_true, y_scores)

#     best_thr, best_f1 = threshold * 100, 0.0
#     for p, r, t in zip(prec, rec, thr):
#         if p >= 0.45:
#             f1_tmp = 2*p*r/(p+r+1e-9)
#             if f1_tmp > best_f1:
#                 best_thr, best_f1 = t, f1_tmp

#     if best_f1 > 0:
#         st.info(f"Suggested higher precision threshold: {best_thr:.3f} (F1 = {best_f1:.3f})")


################################# with long period prediction -- trainset2.csv
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_24H = "model_trend_24h.pkl"
MODEL_72H = "model_trend_72h.pkl"
THR_24H   = "threshold_trend_24h.npy"
THR_72H   = "threshold_trend_72h.npy"

st.set_page_config(page_title="Trend-Aware IoT Predictor", layout="wide")
st.title("IoT Device Disconnection Predictor — Trend-Aware Model")


def normalize_id(df):
    if "DeviceReferenceId" in df.columns:
        df["DeviceReferenceId"] = df["DeviceReferenceId"].astype(str)
    elif "DeviceId" in df.columns:
        df["DeviceReferenceId"] = df["DeviceId"].astype(str)
    else:
        st.error("Missing DeviceReferenceId or DeviceId column.")
        st.stop()
    return df


uploaded = st.file_uploader("Upload today's enriched trend CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload the CSV.")
    st.stop()

df = pd.read_csv(uploaded)
df = normalize_id(df)


df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["Day"]        = pd.to_datetime(df["Day"], errors="coerce")

if "Last transmission" in df.columns:
    df["Last transmission"] = pd.to_datetime(df["Last transmission"], errors="coerce")
elif "Last  transmission" in df.columns:
    df["Last transmission"] = pd.to_datetime(df["Last  transmission"], errors="coerce")
else:
    st.error("Missing 'Last transmission' column.")
    st.stop()

df["DeviceAgeDays"] = (df["Day"] - df["created_at"]).dt.days.clip(lower=0)

df["HoursSinceLastTransmission"] = (
    (df["Day"] - df["Last transmission"]).dt.total_seconds() / 3600
).clip(lower=0)


df["SwVersion"] = df["SwVersion"].astype(str)
df["SwVersionGroup"] = (
    df["SwVersion"].str.extract(r"^(\d+\.\d+)", expand=False).fillna("<UNK>")
)


feature_cols = [
    "AvgBatteryVoltage",
    "AvgRSSI",
    "HeartbeatCount",
    "MaxGapMinutes",
    "BatterySlope3d",
    "BatterySlope7d",
    "RSSISlope3d",
    "RSSISlope7d",
    "HeartbeatSlope3d",
    "HeartbeatSlope7d",
    "DeviceAgeDays",
    "HoursSinceLastTransmission",
    "Country",
    "BU",
    "Is FloLive",
    "SwVersionGroup"
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

categorical_cols = ["Country", "BU", "SwVersionGroup"]

df["Country"] = df["Country"].astype(str).fillna("<UNK>")
df["BU"] = df["BU"].astype(str).fillna("<UNK>")
df["Is FloLive"] = df["Is FloLive"].astype(int)

for c in categorical_cols:
    df[c] = df[c].astype("category")

RED_BG = "background-color: #ffd6d6"
ORANGE_BG = "background-color: #ffe9d6"
NO_BG = ""

def style_prob(series):
    return [RED_BG if v > 90 else ORANGE_BG if v > 80 else NO_BG for v in series]

def style_battery(series):
    return [RED_BG if v < 3600 else ORANGE_BG if v < 3800 else NO_BG for v in series]

def style_rssi(series):
    return [RED_BG if v > 50 else ORANGE_BG if v > 40 else NO_BG for v in series]


horizon = st.sidebar.selectbox("Prediction Horizon", ["24h", "72h"])

MODEL_PATH = MODEL_24H if horizon == "24h" else MODEL_72H
THR_PATH   = THR_24H if horizon == "24h" else THR_72H

model = joblib.load(MODEL_PATH)
threshold = float(np.load(THR_PATH)[0])

st.sidebar.success("Model Loaded")
st.sidebar.write(f"Using threshold = **{threshold:.3f}**")

X = df[feature_cols].copy()

try:
    trained_cols = model.feature_name_
    if set(trained_cols) == set(X.columns):
        X = X[trained_cols]
except:
    pass


df["PredProb"] = model.predict_proba(X)[:, 1]
df["PredLabel"] = (df["PredProb"] >= threshold).astype(int)
df["PredProb"] = (df["PredProb"] * 100).round(1)


st.subheader("High-Risk Devices")

risk_cols = [
    "DeviceReferenceId",
    "Country",
    "BU",
    "FarmName",
    "SwVersion",
    "Is FloLive",
    "Last transmission",
    "AvgBatteryVoltage",
    "AvgRSSI",
    "PredProb"
    ]

risky = (
    df[df["PredLabel"] == 1][risk_cols]
    .sort_values("PredProb", ascending=False)
    .reset_index(drop=True)
)

st.metric("Predicted high-risk devices", len(risky))


styled = (
    risky.style
    .apply(style_prob, subset=["PredProb"])
    .apply(style_battery, subset=["AvgBatteryVoltage"])
    .apply(style_rssi, subset=["AvgRSSI"])
    .format({
        "PredProb": "{:.1f}%",
        "AvgBatteryVoltage": "{:.0f}",
        "AvgRSSI": "{:.0f}",
    })
)

st.dataframe(styled, use_container_width=True)

st.subheader(" Evaluate With Ground Truth CSV")

truth_file = st.file_uploader("Upload truth CSV (DeviceReferenceId list)", type=["csv"], key="truth")

if truth_file:
    truth_df = pd.read_csv(truth_file)
    truth_df = normalize_id(truth_df)

    pred_ids  = set(df.loc[df["PredLabel"] == 1, "DeviceReferenceId"])
    truth_ids = set(truth_df["DeviceReferenceId"])
    all_ids   = set(df["DeviceReferenceId"])

    TP = pred_ids & truth_ids
    FP = pred_ids - truth_ids
    FN = truth_ids - pred_ids
    TN = all_ids - (TP | FP | FN)

    st.write(f" TP: {len(TP)}")
    st.write(f" FP: {len(FP)}")
    st.write(f" FN: {len(FN)}")
    st.write(f" TN: {len(TN)}")

    y_true = [1 if i in truth_ids else 0 for i in all_ids]
    y_pred = [1 if i in pred_ids else 0 for i in all_ids]

    st.write(f"Precision: {precision_score(y_true, y_pred):.3f}")
    st.write(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    st.write(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
