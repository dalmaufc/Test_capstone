import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai

# -----------------------------------------------------------
# App settings
# -----------------------------------------------------------
st.set_page_config(page_title="PE Matchmaker Recommender", page_icon="🔎🏢", layout="wide")

# -----------------------------------------------------------
# Load data and model
# -----------------------------------------------------------
df = pd.read_csv("PE_BDD_df_cleaned2.csv")

# Load updated model and feature list
payload = joblib.load("pe_matcher_LGBM_tuned.pkl")
booster = payload["model"]
feature_list = payload["feature_list"]

# Load filtered list of allowed companies
allowed_df = pd.read_csv("2025-07-04T18-13_export.csv")
allowed_companies = allowed_df["target_company"].dropna().unique()

# Sanitizer
_sanitize = lambda c: re.sub(r"[^A-Za-z0-9_]", "_", c)

industry_cols = ["business services_x", "healthcare & life sciences_x", "technology & software_x"]
geo_cols = [
    'africa_x', 'austria_x', 'belgium_x', 'canada_x', 'caribbean_x', 'china_x',
    'france_x', 'germany_x', 'greece_x', 'india_x', 'ireland_x', 'israel_x',
    'italy_x', 'japan and south korea_x', 'latam_x', 'luxembourg_x', 'middle east_x',
    'netherlands_x', 'nordic_x', 'oceania_x', 'poland_x', 'portugal_x',
    'rest of asia_x', 'rest of europe_x', 'rest of world_x', 'russia',
    'spain_x', 'switzerland_x', 'turkey_x', 'united kingdom_x', 'united states_x'
]
tier_cols = ["Tier_1", "Tier_2", "Tier_3", "Tier_4", "Tier_5"]

pe_cols = [c.replace("_x", "_y") for c in industry_cols + geo_cols]
pe_cols += ["min_ebitda", "max_ebitda"] + tier_cols + ["pe_y"]

pe_features_df = df[pe_cols].drop_duplicates("pe_y").set_index("pe_y")
pe_features_df.columns = [_sanitize(c) for c in pe_features_df.columns]
pe_features = pe_features_df.to_dict(orient="index")

if "top5_df" not in st.session_state:
    st.session_state.top5_df = None
if "company_dict" not in st.session_state:
    st.session_state.company_dict = None

# -----------------------------------------------------------
# UI
# -----------------------------------------------------------
st.title("PE Matchmaker Recommender 🔎🏢 (LGBM Final Version)")

mode = st.sidebar.radio("Select Mode", ["Select Existing Company", "Manual Input"])

selected_industries = []
selected_geos = []
ebitda_value = 10

if mode == "Select Existing Company":
    st.sidebar.header("Select Company")

    # Filter companies in df based on allowed list
    company_names = df[df["target_company"].isin(allowed_companies)]["target_company"].dropna().unique()
    selected_company = st.sidebar.selectbox("Choose a company", sorted(company_names))

    company_row = df[df["target_company"] == selected_company].iloc[0]

    selected_industries = [col for col in industry_cols if company_row.get(col, 0) == 1]
    selected_geos = [col for col in geo_cols if company_row.get(col, 0) == 1]
    ebitda_value = int(company_row.get("reported_ebitda_m_y1", 10))
    
    st.sidebar.markdown("✅ Features loaded from company data.")
else:
    st.sidebar.header("Define New Company Profile")
    selected_industries = st.sidebar.multiselect("Select Industries", sorted(industry_cols))
    selected_geos = st.sidebar.multiselect("Select Geographies", sorted(geo_cols))
    ebitda_value = st.sidebar.slider("Estimated EBITDA (€M)", 0, 50, 10)

st.sidebar.header("API Configuration")
api_key_input = st.sidebar.text_input("Enter your Gemini API key", type="password")
if api_key_input:
    st.session_state["GOOGLE_API_KEY"] = api_key_input

def recommend_top5(company_features: dict, k: int = 5):
    base = pd.Series(company_features)
    cand = pd.DataFrame(pe_features).T.reset_index().rename(columns={"index": "pe_y"})

    df_comb = pd.concat(
        [pd.DataFrame([base] * len(cand)).reset_index(drop=True), cand.reset_index(drop=True)],
        axis=1
    )

    x_sector_cols = [_sanitize(c) for c in industry_cols]
    y_sector_cols = [_sanitize(c.replace("_x", "_y")) for c in industry_cols]
    x_region_cols = [_sanitize(c) for c in geo_cols]
    y_region_cols = [_sanitize(c.replace("_x", "_y")) for c in geo_cols]

    for col in x_sector_cols + y_sector_cols + x_region_cols + y_region_cols:
        df_comb[col] = df_comb[col].fillna(0).astype(np.int8)

    df_comb["sector_overlap_cnt"] = (
        (df_comb[x_sector_cols].to_numpy() & df_comb[y_sector_cols].to_numpy()).sum(1)
    ) if x_sector_cols and y_sector_cols else 0

    df_comb["region_overlap_cnt"] = (
        (df_comb[x_region_cols].to_numpy() & df_comb[y_region_cols].to_numpy()).sum(1)
    ) if x_region_cols and y_region_cols else 0

    df_comb["sector_match"] = (df_comb["sector_overlap_cnt"] > 0).astype(np.int8)
    df_comb["region_match"] = (df_comb["region_overlap_cnt"] > 0).astype(np.int8)

    df_comb["ebitda_in_range"] = (
        (df_comb["reported_ebitda_m_y1"].astype(float) >= df_comb["min_ebitda"].astype(float)) &
        (
            (df_comb["max_ebitda"].astype(float) == 0) |
            (df_comb["reported_ebitda_m_y1"].astype(float) <= df_comb["max_ebitda"].astype(float))
        )
    ).astype(np.int8)

    actual_tier_cols = [c for c in tier_cols if c in df_comb.columns]
    tier_weights = pd.Series({c: i for i, c in enumerate(actual_tier_cols, 1)})
    df_comb["tier_numeric"] = (
        df_comb[actual_tier_cols].fillna(0).astype(np.int8)
          .dot(tier_weights)
          .astype(np.int8)
    )

    df_comb.columns = [_sanitize(c) for c in df_comb.columns]

    for feat in feature_list:
        if feat not in df_comb.columns:
            df_comb[feat] = 0
    X = df_comb[feature_list].astype(np.float32)

    scores = booster.predict(X, num_iteration=booster.best_iteration)
    out = (
        pd.DataFrame({"PE": df_comb["pe_y"], "Score": scores})
          .sort_values("Score", ascending=False)
          .head(k)
          .reset_index(drop=True)
    )
    return out, df_comb, X

if st.sidebar.button("Get Recommendations"):
    if not selected_industries or not selected_geos:
        st.warning("Please select at least one industry and one geography.")
    else:
        company_dict = {c: 0 for c in industry_cols + geo_cols}
        for col in selected_industries:
            company_dict[col] = 1
        for col in selected_geos:
            company_dict[col] = 1
        company_dict["reported_ebitda_m_y1"] = ebitda_value
        company_dict["deal_value_eur_m"] = 0

        company_dict = {_sanitize(k): v for k, v in company_dict.items()}

        top5_df, df_comb_full, X_full = recommend_top5(company_dict, k=5)

        st.session_state.top5_df = top5_df
        st.session_state.company_dict = company_dict
        st.session_state.df_comb_full = df_comb_full
        st.session_state.X_full = X_full

if st.session_state.top5_df is not None:
    st.subheader("Top 5 Recommended PEs")

    display_df = st.session_state.top5_df[["PE"]].copy()
    display_df.index = display_df.index + 1
    st.table(display_df)

    if mode == "Select Existing Company" and "pe_y" in df.columns:
        true_pe_y = company_row["pe_y"]
        st.info(f"Actual target PE_y (from data): **{true_pe_y}**")

        recommended_pes = st.session_state.top5_df["PE"].values
        if true_pe_y in recommended_pes:
            st.success("✅ The actual PE_y is in the top 5 recommendations!")
        else:
            st.warning("⚠️ The actual PE_y is NOT in the top 5 recommendations.")

    selected_pe = st.selectbox("Select a PE to see detailed explanation", st.session_state.top5_df["PE"])

    if st.button("Generate Explanation for Selected PE"):
        if "GOOGLE_API_KEY" not in st.session_state or not st.session_state["GOOGLE_API_KEY"]:
            st.warning("Please enter your Gemini API key in the sidebar before generating explanation.")
            st.stop()

        genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])

        df_comb_full = st.session_state.df_comb_full
        X_full = st.session_state.X_full

        selected_index = df_comb_full[df_comb_full["pe_y"] == selected_pe].index[0]

        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_full.iloc[[selected_index]])

        analysis_text = ""
        for i, feat in enumerate(feature_list):
            contribution = shap_values[0][i]
            if abs(contribution) > 0.01:
                analysis_text += f"{feat}: {contribution:+.3f}\n"

        model = genai.GenerativeModel("models/gemini-1.5-flash", generation_config={"temperature": 0.7})
        prompt = f"Given these feature contributions:\n{analysis_text}\nWrite a clear, friendly explanation for why this PE firm is a good match for the given company."

        response = model.generate_content(prompt)

        st.subheader("Human-style explanation")
        st.write(response.text)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values, X_full.iloc[[selected_index]], plot_type="bar", max_display=10, show=False)
        st.pyplot(fig)

st.caption("Powered by your LGBM model 🚀, SHAP ✨ and Gemini 🤖")
