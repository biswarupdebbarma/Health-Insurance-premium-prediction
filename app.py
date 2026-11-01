import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from insurance_model import predict_charges

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "health_insurance.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
ILLUSTRATION_PATH = BASE_DIR / "life_cover_illustration.svg"

st.set_page_config(
    page_title="Evergreen | Insurance Cost Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_theme():
    illustration = base64.b64encode(ILLUSTRATION_PATH.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
        :root {{
            --ink: #17333d;
            --muted: #647d83;
            --teal: #237b73;
            --gold: #c99b52;
        }}
        .stApp {{
            background-color: #f3f8f6;
            background-image:
                linear-gradient(90deg, rgba(243,248,246,.99) 0%, rgba(243,248,246,.96) 42%, rgba(243,248,246,.64) 74%, rgba(243,248,246,.50) 100%),
                url("data:image/svg+xml;base64,{illustration}");
            background-repeat: no-repeat, no-repeat;
            background-size: cover, min(72vw, 940px) auto;
            background-position: center, right 2vw top 4rem;
            color: var(--ink);
            font-family: 'Segoe UI', sans-serif;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"] {{ right: 1rem; }}
        .block-container {{
            max-width: 1180px;
            padding: 1.4rem 2rem 4rem;
        }}
        h1, h2, h3, p, label {{ font-family: 'Segoe UI', sans-serif; }}
        .brand {{
            display: flex;
            align-items: center;
            gap: .65rem;
            color: #355a5d;
            font-size: .79rem;
            font-weight: 700;
            letter-spacing: .15em;
            text-transform: uppercase;
        }}
        .brand-mark {{
            display: grid;
            width: 2rem;
            height: 2rem;
            place-items: center;
            border-radius: 50%;
            background: #e0eeea;
            color: var(--teal);
            font-size: 1.05rem;
        }}
        .hero {{ padding: 3.7rem 0 2.5rem; }}
        .eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            padding: .42rem .8rem;
            border: 1px solid rgba(35,123,115,.16);
            border-radius: 99px;
            background: rgba(255,255,255,.72);
            color: #38746e;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .035em;
        }}
        .hero h1 {{
            max-width: 650px;
            margin: 1.15rem 0 .85rem;
            color: var(--ink);
            font-family: 'Segoe UI', sans-serif;
            font-size: clamp(2.6rem, 5vw, 4.3rem);
            font-weight: 800;
            letter-spacing: -.065em;
            line-height: 1.04;
        }}
        .hero h1 span {{ color: var(--teal); }}
        .hero-copy {{
            max-width: 510px;
            color: #5b7379;
            font-size: 1.08rem;
            line-height: 1.75;
        }}
        .trust-row {{
            display: flex;
            flex-wrap: wrap;
            gap: .65rem;
            margin-top: 1.35rem;
        }}
        .trust-chip {{
            padding: .52rem .78rem;
            border: 1px solid rgba(57,105,104,.14);
            border-radius: 10px;
            background: rgba(255,255,255,.75);
            color: #49666b;
            font-size: .78rem;
            font-weight: 600;
        }}
        .section-heading {{
            margin: .2rem 0 .2rem;
            color: var(--ink);
            font-family: 'Segoe UI', sans-serif;
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -.035em;
        }}
        .section-copy {{ margin-bottom: 1.25rem; color: var(--muted); font-size: .91rem; }}
        div[data-testid="stForm"] {{
            padding: 1.7rem 1.8rem 1.4rem;
            border: 1px solid rgba(53,91,92,.13);
            border-radius: 22px;
            background: rgba(255,255,255,.92);
            box-shadow: 0 22px 65px rgba(27,65,68,.10);
            backdrop-filter: blur(14px);
        }}
        div[data-testid="stForm"] label {{
            color: #36545b;
            font-size: .85rem;
            font-weight: 600;
        }}
        div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input {{
            border-color: #dce8e5;
            border-radius: 10px;
            background-color: #fbfdfc;
        }}
        div[data-testid="stFormSubmitButton"] button {{
            min-height: 3.1rem;
            margin-top: .55rem;
            border: 0;
            border-radius: 11px;
            background: #237b73;
            color: #fff;
            font-size: .96rem;
            font-weight: 700;
            box-shadow: 0 8px 18px rgba(35,123,115,.2);
            transition: transform .18s ease, background .18s ease, box-shadow .18s ease;
        }}
        div[data-testid="stFormSubmitButton"] button:hover {{
            transform: translateY(-1px);
            background: #1b6862;
            box-shadow: 0 11px 22px rgba(35,123,115,.24);
            color: #fff;
        }}
        .result-card {{
            margin-top: 1.25rem;
            padding: 1.5rem 1.7rem;
            border: 1px solid rgba(35,123,115,.18);
            border-radius: 18px;
            background: linear-gradient(120deg, rgba(229,243,237,.97), rgba(255,255,255,.96));
            box-shadow: 0 12px 35px rgba(27,65,68,.08);
        }}
        .result-label {{
            color: #54736e;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }}
        .result-value {{
            margin: .25rem 0;
            color: #1b665f;
            font-family: 'Segoe UI', sans-serif;
            font-size: clamp(2.1rem, 4vw, 3rem);
            font-weight: 800;
            letter-spacing: -.055em;
        }}
        .fine-print {{ color: #6d8584; font-size: .8rem; line-height: 1.55; }}
        [data-testid="stAlert"] {{ border-radius: 12px; }}
        @media (max-width: 760px) {{
            .block-container {{ padding: 1rem 1rem 3rem; }}
            .hero {{ padding: 2.6rem 0 1.6rem; }}
            .stApp {{
                background-image:
                    linear-gradient(180deg, rgba(243,248,246,.95) 0%, rgba(243,248,246,.98) 52%, #f3f8f6 100%),
                    url("data:image/svg+xml;base64,{illustration}");
                background-size: cover, 115vw auto;
                background-position: center, right -20vw top 4rem;
            }}
            div[data-testid="stForm"] {{ padding: 1.25rem 1rem 1rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    if not DATA_PATH.is_file() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError("health_insurance.csv is missing or empty.")
    data = pd.read_csv(DATA_PATH)
    if data.empty or "charges" not in data.columns:
        raise ValueError("CSV must contain data and a 'charges' column.")
    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("model.pkl is missing. Run make_model.py first.")
    model = joblib.load(MODEL_PATH)
    if not isinstance(model, dict) or model.get("format_version") != 1:
        raise ValueError("model.pkl has an unsupported format. Run make_model.py to rebuild it.")
    return model


def main():
    apply_theme()

    try:
        data = load_data()
        model = load_model()
    except Exception as error:
        st.error(str(error))
        st.stop()

    features = data.drop(columns="charges")
    st.markdown(
        """
        <div class="brand"><span class="brand-mark">✦</span> Evergreen <span>·</span> Coverage studio</div>
        <section class="hero">
            <div class="eyebrow">A little clarity for the road ahead</div>
            <h1>Protect what matters.<br><span>Plan what’s next.</span></h1>
            <p class="hero-copy">
                Get a thoughtful estimate of health insurance costs based on your
                profile. A simple first step toward feeling more prepared.
            </p>
            <div class="trust-row">
                <span class="trust-chip">✓ &nbsp;No account needed</span>
                <span class="trust-chip">⌁ &nbsp;Takes less than a minute</span>
                <span class="trust-chip">♡ &nbsp;Built around real data</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Your coverage snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A few details help us tailor your estimate to you.</div>',
        unsafe_allow_html=True,
    )

    category_options = {
        column: sorted(data[column].dropna().astype(str).unique())
        for column in ["sex", "smoker", "region"]
    }
    if any(not options for options in category_options.values()):
        st.error("The dataset is missing values needed for the profile selectors.")
        st.stop()

    with st.form("insurance_profile"):
        age_col, sex_col, bmi_col = st.columns(3)
        children_col, smoker_col, region_col = st.columns(3)
        with age_col:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        with sex_col:
            sex = st.selectbox("Sex", category_options["sex"])
        with bmi_col:
            bmi = st.number_input("Body mass index", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
        with children_col:
            children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)
        with smoker_col:
            smoker = st.selectbox("Tobacco use", category_options["smoker"])
        with region_col:
            region = st.selectbox("Region", category_options["region"])
        submitted = st.form_submit_button("Get my estimate  →", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            [{
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }],
            columns=features.columns,
        )
        try:
            prediction = float(predict_charges(model, input_df)[0])
        except (TypeError, ValueError) as error:
            st.error(f"Could not generate a prediction: {error}")
            st.stop()
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Your estimated annual cost</div>
                <div class="result-value">${prediction:,.2f}</div>
                <div class="fine-print">
                    This is an estimate based on historical sample data, not an insurance quote.
                    Actual premiums vary by provider, plan, location, and underwriting.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()