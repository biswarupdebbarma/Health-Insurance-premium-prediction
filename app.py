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
FALLBACK_ILLUSTRATION = """<svg xmlns="http://www.w3.org/2000/svg" width="920" height="680" viewBox="0 0 920 680" fill="none">
<circle cx="650" cy="331" r="278" fill="#DCEEEF"/><circle cx="650" cy="331" r="226" fill="#E8F3F2"/>
<path d="M653 102 827 168v137c0 119-72 211-174 268-102-57-174-149-174-268V168l174-66Z" fill="#fff" stroke="#C7DFDC" stroke-width="8"/>
<path d="M653 144 787 195v110c0 94-55 168-134 216-79-48-134-122-134-216V195l134-51Z" fill="#F4F9F8"/>
<path d="M651 183c-18-29-62-23-70 11-10 42 70 84 70 84s80-42 70-84c-8-34-52-40-70-11Z" fill="#D9A95E"/>
<path d="M372 561c0-104 59-170 133-170s133 66 133 170H372Z" fill="#2C6D70"/>
<path d="M516 477c0-78 45-128 100-128s100 50 100 128v84H516v-84Z" fill="#91BDB7"/>
<path d="M422 354c0-50 37-87 83-87s83 37 83 87-37 84-83 84-83-34-83-84Z" fill="#E7B991"/>
<path d="M422 347c-6-56 29-99 83-99 42 0 74 28 82 68-19-11-40-18-64-18-37 0-67 16-101 49Z" fill="#344C53"/>
<path d="M567 304c0-51 36-87 82-87 47 0 83 36 83 87s-36 82-83 82c-46 0-82-31-82-82Z" fill="#DDAE87"/>
<path d="M567 296c-2-55 33-98 82-98 49 0 79 31 84 74-23-15-47-21-76-21-34 0-64 14-90 45Z" fill="#6B5144"/>
<path d="M573 534c0-77 31-119 76-119s77 42 77 119v27H573v-27Z" fill="#F0C77D"/>
<path d="M603 421c0-33 21-55 47-55s47 22 47 55-21 51-47 51-47-18-47-51Z" fill="#E9B58D"/>
<path d="M602 416c-2-35 17-64 48-64 28 0 46 18 49 47-13-9-28-13-44-13-20 0-37 10-53 30Z" fill="#50423F"/>
<path d="M290 567h485" stroke="#B7D3D0" stroke-width="8" stroke-linecap="round"/>
<circle cx="786" cy="462" r="34" fill="#fff"/><path d="m772 462 10 10 19-21" stroke="#4E9388" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="429" cy="181" r="18" fill="#E9C379"/><circle cx="799" cy="225" r="11" fill="#7DAFA8"/>
<path d="M288 260c35-30 76-46 122-47" stroke="#BCD9D5" stroke-width="5" stroke-linecap="round" stroke-dasharray="3 15"/>
</svg>"""

st.set_page_config(
    page_title="Biswarup | Insurance Cost Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_theme():
    illustration_bytes = (
        ILLUSTRATION_PATH.read_bytes()
        if ILLUSTRATION_PATH.is_file()
        else FALLBACK_ILLUSTRATION.encode("utf-8")
    )
    illustration = base64.b64encode(illustration_bytes).decode("ascii")
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
        <div class="brand"><span class="brand-mark">✦</span>  <span>·</span> </div>
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
