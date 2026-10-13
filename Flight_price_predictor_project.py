# =============================================================================
# ✈️  FLIGHT PRICE PREDICTION — app.py
# A complete ML dashboard in a single Streamlit file.
# Dataset: Clean_Dataset.csv  (Kaggle – EaseMyTrip flight price dataset)
# Author : Built with ❤️  using Python, Scikit-Learn & Streamlit
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Gradient page background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8eaf6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a4e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #c5cae9 !important; }
[data-testid="stSidebar"] .sidebar-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #7986cb !important;
    letter-spacing: 0.04em;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #90caf9;
    margin-bottom: 0.5rem;
    letter-spacing: 0.03em;
}

/* ── Section headings ── */
h1 { color: #e8eaf6 !important; }
h2 { color: #90caf9 !important; font-weight: 700 !important; }
h3 { color: #b39ddb !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #1a237e 0%, #4a148c 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.15);
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 0.02em;
    margin: 0;
    color: #ffffff !important;
}
.hero p {
    color: #b0bec5;
    font-size: 1.1rem;
    margin-top: 0.6rem;
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.12);
}
.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9e9e9e;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #80deea;
}

/* ── Best model badge ── */
.best-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00c853, #00897b);
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Price result ── */
.price-result {
    background: linear-gradient(135deg, #1b5e20 0%, #004d40 100%);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    border: 1px solid #00e676;
    box-shadow: 0 0 24px rgba(0,230,118,0.25);
    margin-top: 1.5rem;
}
.price-result h2 { color: #b9f6ca !important; font-size: 1.2rem !important; }
.price-result h1 { color: #00e676 !important; font-size: 3.2rem !important; }

/* ── Streamlit default overrides ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.1);
}
div[data-testid="stMetricValue"] { color: #80deea !important; }
div[data-testid="stMetricLabel"] { color: #9e9e9e !important; }

/* Dataframe */
.stDataFrame { background: rgba(255,255,255,0.04); border-radius: 10px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #3949ab, #7b1fa2);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.65rem 2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Select/slider labels */
.stSelectbox label, .stSlider label { color: #b0bec5 !important; }

/* Expander */
details { background: rgba(255,255,255,0.04) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (path or uploader fallback)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None) -> pd.DataFrame:
    """Load dataset from an uploaded file or local path."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    # Try common local names
    for path in ["Clean_Dataset.csv", "clean_dataset.csv", "data/Clean_Dataset.csv"]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["airline", "source_city", "destination_city",
                    "departure_time", "arrival_time", "stops", "class"]

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame):
    """Encode categoricals, return X, y, encoders dict."""
    df = df.copy()

    # Drop unnamed index column if present
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True, errors="ignore")

    # Normalise column names: lowercase + replace spaces with _
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Map 'stops' text → numeric if needed (e.g. "zero" → 0)
    stop_map = {
        "zero": 0, "one": 1, "two_or_more": 2,
        "0": 0, "1": 1, "2": 2, "2+": 2,
    }
    if "stops" in df.columns:
        df["stops_num"] = (
            df["stops"].astype(str).str.lower()
            .map(stop_map)
            .fillna(df["stops"])
        )
        try:
            df["stops_num"] = df["stops_num"].astype(float)
        except Exception:
            df["stops_num"] = 0.0

    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    feature_cols = [f"{c}_enc" for c in CATEGORICAL_COLS if c in df.columns]
    feature_cols += ["duration", "days_left"]
    if "stops_num" in df.columns:
        feature_cols = [c for c in feature_cols if c != "stops_enc"]
        feature_cols.append("stops_num")

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["price"].copy()

    return X, y, encoders, df


# ──────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(_X, _y):
    """Train Linear, Polynomial, and Decision-Tree regressors. Return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42
    )

    results = {}

    # 1. Linear Regression
    lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["Linear Regression"] = {
        "model": lr,
        "R²": r2_score(y_test, y_pred_lr),
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "MSE": mean_squared_error(y_test, y_pred_lr),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    }

    # 2. Polynomial Regression (degree=2)
    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    poly.fit(X_train, y_train)
    y_pred_poly = poly.predict(X_test)
    results["Polynomial Regression"] = {
        "model": poly,
        "R²": r2_score(y_test, y_pred_poly),
        "MAE": mean_absolute_error(y_test, y_pred_poly),
        "MSE": mean_squared_error(y_test, y_pred_poly),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_poly)),
    }

    # 3. Decision Tree Regressor
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results["Decision Tree Regressor"] = {
        "model": dt,
        "R²": r2_score(y_test, y_pred_dt),
        "MAE": mean_absolute_error(y_test, y_pred_dt),
        "MSE": mean_squared_error(y_test, y_pred_dt),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    }

    best_name = max(results, key=lambda k: results[k]["R²"])
    return results, best_name, X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ──────────────────────────────────────────────────────────────────────────────
PLOT_BG   = "rgba(15,12,41,0)"
GRID_CLR  = "rgba(255,255,255,0.07)"
FONT_CLR  = "#c5cae9"

def dark_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=FONT_CLR, size=15)),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR),
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">✈️ Flight Price AI</p>', unsafe_allow_html=True)
    st.divider()

    uploaded_csv = st.file_uploader(
        "📂 Upload Clean_Dataset.csv",
        type=["csv"],
        help="Upload the dataset if it's not in the same folder as app.py",
    )

    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset Analysis", "🤖 Model Training", "🔮 Price Prediction"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; color:#616161; line-height:1.7'>
    <b style='color:#7986cb'>Dataset</b><br>EaseMyTrip · 300k+ rows<br><br>
    <b style='color:#7986cb'>Models</b><br>Linear · Poly · Decision Tree<br><br>
    <b style='color:#7986cb'>Stack</b><br>Python · Scikit-Learn · Streamlit · Plotly
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
df_raw = load_data(uploaded_csv)

if df_raw is None:
    st.error(
        "⚠️  **Dataset not found.**  "
        "Please upload `Clean_Dataset.csv` using the sidebar uploader, "
        "or place the file in the same directory as `app.py` and restart."
    )
    st.stop()

X, y, encoders, df = preprocess(df_raw)


# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE 1 : HOME
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":

    st.markdown("""
    <div class='hero'>
        <h1>✈️ Flight Price Prediction</h1>
        <p>An end-to-end Machine Learning dashboard — from raw data to real-time price prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📋 Total Records", f"{len(df):,}")
    with c2:
        st.metric("📊 Features", str(len(df.columns) - 1))
    with c3:
        st.metric("💰 Avg Price (₹)", f"{df['price'].mean():,.0f}")
    with c4:
        st.metric("✈️ Airlines", str(df["airline"].nunique()) if "airline" in df.columns else "–")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>📖 Project Description</div>
            <p style='color:#b0bec5; font-size:0.93rem; line-height:1.75'>
            This application predicts domestic Indian flight ticket prices using
            real-world booking data scraped from <b>EaseMyTrip</b>.  It trains
            three regression models, automatically picks the best one by R² score,
            and lets you get an instant price estimate for any itinerary.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div class='card-title'>🎯 Target Variable</div>
            <p style='color:#b0bec5; font-size:0.93rem'>
            <b style='color:#80deea'>price</b> — Ticket fare in Indian Rupees (₹).
            Ranges from ~₹1,000 for budget Economy seats to ₹1,00,000+ for
            last-minute Business class bookings.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div class='card-title'>🛠️ Technologies Used</div>
            <p style='color:#b0bec5; font-size:0.92rem; line-height:2'>
            🐍 Python 3.x &nbsp;|&nbsp; 📦 Pandas &nbsp;|&nbsp; 🔢 NumPy<br>
            🤖 Scikit-Learn &nbsp;|&nbsp; 📊 Plotly &nbsp;|&nbsp; 🌐 Streamlit
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        features_html = "".join([
            f"<li style='color:#b0bec5;margin-bottom:4px'>{f}</li>"
            for f in ["✈️ Airline", "🏙️ Source City", "🏙️ Destination City",
                      "🌅 Departure Time", "🌇 Arrival Time", "🛑 No. of Stops",
                      "⏱️ Duration (hrs)", "💺 Class (Economy / Business)",
                      "📅 Days Left to Departure"]
        ])
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>🔑 Input Features</div>
            <ul style='padding-left:1.2rem'>{features_html}</ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div class='card-title'>🤖 ML Models</div>
            <ol style='color:#b0bec5; padding-left:1.2rem; line-height:2'>
            <li>Linear Regression</li>
            <li>Polynomial Regression (degree=2)</li>
            <li>Decision Tree Regressor</li>
            </ol>
            <p style='color:#9e9e9e; font-size:0.85rem'>
            80 / 20 train-test split · Evaluated on R², MAE, MSE, RMSE
            </p>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE 2 : DATASET ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dataset Analysis":
    st.markdown("## 📊 Dataset Analysis")

    # ── Raw data preview
    with st.expander("📋 Preview Raw Dataset", expanded=True):
        n_rows = st.slider("Rows to display", 5, 100, 20, key="preview_slider")
        st.dataframe(df.head(n_rows), use_container_width=True)

    st.markdown("---")

    # ── Basic info row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        with st.expander("🔠 Data Types"):
            dtype_df = df.dtypes.reset_index()
            dtype_df.columns = ["Column", "DType"]
            st.dataframe(dtype_df, use_container_width=True, height=260)

    with col_r:
        with st.expander("❌ Missing Values per Column"):
            mv = df.isnull().sum().reset_index()
            mv.columns = ["Column", "Missing"]
            st.dataframe(mv[mv["Missing"] > 0] if mv["Missing"].sum() > 0
                         else pd.DataFrame({"Column": ["✅ No missing values"], "Missing": [0]}),
                         use_container_width=True, height=260)

    st.markdown("---")

    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"),
                     use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Interactive Visualisations")

    # ── Airline vs Price
    if "airline" in df.columns:
        agg_airline = df.groupby("airline")["price"].median().reset_index().sort_values("price", ascending=False)
        fig = px.bar(agg_airline, x="airline", y="price",
                     color="price", color_continuous_scale="Viridis",
                     labels={"airline": "Airline", "price": "Median Price (₹)"},
                     text_auto=".2s")
        dark_layout(fig, "✈️ Median Ticket Price by Airline")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # ── Class vs Price
    with c1:
        if "class" in df.columns:
            agg_cls = df.groupby("class")["price"].median().reset_index()
            fig2 = px.pie(agg_cls, names="class", values="price",
                          color_discrete_sequence=["#3949ab", "#7b1fa2"],
                          hole=0.45)
            dark_layout(fig2, "💺 Price Share by Class")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Stops vs Price
    with c2:
        if "stops" in df.columns:
            agg_stops = df.groupby("stops")["price"].median().reset_index()
            fig3 = px.bar(agg_stops, x="stops", y="price",
                          color="price", color_continuous_scale="Plasma",
                          labels={"stops": "Stops", "price": "Median Price (₹)"})
            dark_layout(fig3, "🛑 Median Price by Number of Stops")
            st.plotly_chart(fig3, use_container_width=True)

    # ── Duration vs Price (scatter with density)
    if "duration" in df.columns:
        sample = df.sample(min(5000, len(df)), random_state=1)
        fig4 = px.scatter(sample, x="duration", y="price",
                          color="class" if "class" in df.columns else None,
                          opacity=0.45,
                          color_discrete_sequence=px.colors.qualitative.Bold,
                          labels={"duration": "Duration (hrs)", "price": "Price (₹)"},
                          trendline="ols")
        dark_layout(fig4, "⏱️ Duration vs Price")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Days Left vs Price
    if "days_left" in df.columns:
        agg_days = df.groupby("days_left")["price"].median().reset_index()
        fig5 = px.line(agg_days, x="days_left", y="price",
                       markers=False,
                       color_discrete_sequence=["#80deea"],
                       labels={"days_left": "Days Left", "price": "Median Price (₹)"})
        dark_layout(fig5, "📅 Days Left to Departure vs Median Price")
        st.plotly_chart(fig5, use_container_width=True)

    # ── Correlation Heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig_h = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.3f}<extra></extra>",
        ))
        dark_layout(fig_h, "Feature Correlation Matrix")
        fig_h.update_layout(height=480)
        st.plotly_chart(fig_h, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE 3 : MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training & Evaluation")

    with st.spinner("🔧 Training models — this takes a few seconds …"):
        results, best_name, X_train, X_test, y_train, y_test = train_models(X, y)

    st.success(f"✅ All models trained!  Best model: **{best_name}**")

    # ── Training progress bars (cosmetic, show split info)
    st.markdown("### 📐 Dataset Split")
    c1, c2 = st.columns(2)
    c1.metric("🏋️ Training Samples", f"{len(X_train):,}")
    c2.metric("🧪 Testing Samples",  f"{len(X_test):,}")
    st.progress(0.8, text="80% Training  |  20% Testing")

    st.markdown("---")
    st.markdown("### 📊 Model Comparison")

    # Build comparison dataframe
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "R² Score": f"{res['R²']:.4f}",
            "MAE (₹)": f"{res['MAE']:,.0f}",
            "MSE": f"{res['MSE']:,.0f}",
            "RMSE (₹)": f"{res['RMSE']:,.0f}",
            "Best?": "🏆 Yes" if name == best_name else "–",
        })
    compare_df = pd.DataFrame(rows)
    st.dataframe(compare_df.set_index("Model"), use_container_width=True)

    st.markdown("---")

    # ── R² Bar chart
    model_names = list(results.keys())
    r2_vals = [results[m]["R²"] for m in model_names]
    colors = ["#00e676" if m == best_name else "#5c6bc0" for m in model_names]

    fig_r2 = go.Figure(go.Bar(
        x=model_names, y=r2_vals,
        marker_color=colors,
        text=[f"{v:.4f}" for v in r2_vals],
        textposition="outside",
    ))
    dark_layout(fig_r2, "R² Score Comparison (higher = better)")
    fig_r2.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig_r2, use_container_width=True)

    # ── Metrics grouped bar
    metrics = ["MAE (₹)", "RMSE (₹)"]
    fig_err = go.Figure()
    for i, metric in enumerate(["MAE", "RMSE"]):
        fig_err.add_trace(go.Bar(
            name=metric,
            x=model_names,
            y=[results[m][metric] for m in model_names],
            text=[f"₹{results[m][metric]:,.0f}" for m in model_names],
            textposition="outside",
        ))
    dark_layout(fig_err, "Error Metrics Comparison (lower = better)")
    fig_err.update_layout(barmode="group")
    st.plotly_chart(fig_err, use_container_width=True)

    # ── Per-model detail cards
    st.markdown("### 🔍 Detailed Model Reports")
    cols = st.columns(3)
    icons = ["📏", "🔮", "🌳"]
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            badge = "<span class='best-badge'>BEST</span>" if name == best_name else ""
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>{icons[i]} {name} {badge}</div>
                <div class='metric-box' style='margin-bottom:0.6rem'>
                    <div class='metric-label'>R² Score</div>
                    <div class='metric-value'>{res['R²']:.4f}</div>
                </div>
                <div class='metric-box' style='margin-bottom:0.6rem'>
                    <div class='metric-label'>MAE</div>
                    <div class='metric-value'>₹{res['MAE']:,.0f}</div>
                </div>
                <div class='metric-box' style='margin-bottom:0.6rem'>
                    <div class='metric-label'>RMSE</div>
                    <div class='metric-value'>₹{res['RMSE']:,.0f}</div>
                </div>
                <div class='metric-box'>
                    <div class='metric-label'>MSE</div>
                    <div class='metric-value' style='font-size:1.2rem'>{res['MSE']:,.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# ── PAGE 4 : PRICE PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Price Prediction":
    st.markdown("## 🔮 Predict Your Flight Price")

    # Train models silently
    with st.spinner("⚙️ Preparing models …"):
        results, best_name, X_train, X_test, y_train, y_test = train_models(X, y)

    st.info(f"🏆 Using best model: **{best_name}**  (R² = {results[best_name]['R²']:.4f})")

    st.markdown("---")
    st.markdown("### 🎛️ Enter Flight Details")

    def get_options(col):
        """Return unique string options for a categorical column."""
        return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else []

    col1, col2, col3 = st.columns(3)

    with col1:
        airline_opts = get_options("airline")
        inp_airline = st.selectbox("✈️ Airline", airline_opts or ["IndiGo"])

        source_opts = get_options("source_city")
        inp_source = st.selectbox("🛫 Source City", source_opts or ["Delhi"])

        dest_opts = get_options("destination_city")
        inp_dest = st.selectbox("🛬 Destination City", dest_opts or ["Mumbai"])

    with col2:
        dep_opts = get_options("departure_time")
        inp_dep = st.selectbox("🌅 Departure Time", dep_opts or ["Morning"])

        arr_opts = get_options("arrival_time")
        inp_arr = st.selectbox("🌇 Arrival Time", arr_opts or ["Evening"])

        stops_opts = get_options("stops")
        inp_stops = st.selectbox("🛑 Number of Stops", stops_opts or ["zero"])

    with col3:
        cls_opts = get_options("class")
        inp_class = st.selectbox("💺 Class", cls_opts or ["Economy"])

        inp_duration = st.slider(
            "⏱️ Duration (hours)", 0.5, 50.0,
            float(df["duration"].median()) if "duration" in df.columns else 2.5,
            0.5,
        )
        inp_days = st.slider(
            "📅 Days Left to Departure", 1, 50,
            int(df["days_left"].median()) if "days_left" in df.columns else 15,
        )

    st.markdown("---")

    if st.button("🔮 Predict Flight Price", use_container_width=True):

        # ── Build input row
        stop_map_pred = {
            "zero": 0, "one": 1, "two_or_more": 2,
            "0": 0, "1": 1, "2": 2, "2+": 2,
        }
        stops_num_val = stop_map_pred.get(inp_stops.lower(), 0)

        row = {}
        for col, val in [
            ("airline", inp_airline), ("source_city", inp_source),
            ("destination_city", inp_dest), ("departure_time", inp_dep),
            ("arrival_time", inp_arr), ("stops", inp_stops), ("class", inp_class),
        ]:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels gracefully
                if val in le.classes_:
                    row[f"{col}_enc"] = le.transform([val])[0]
                else:
                    row[f"{col}_enc"] = 0

        row["duration"] = inp_duration
        row["days_left"] = inp_days
        # Use stops_num if it was used during training
        if "stops_num" in X.columns:
            row["stops_num"] = stops_num_val
            row.pop("stops_enc", None)

        input_df = pd.DataFrame([row])
        # Align columns to training feature order
        for c in X.columns:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[X.columns]

        best_model = results[best_name]["model"]
        pred_price = best_model.predict(input_df)[0]
        pred_price = max(0, pred_price)

        st.markdown(f"""
        <div class='price-result'>
            <h2>✈️ Predicted Flight Price</h2>
            <h1>₹ {pred_price:,.0f}</h1>
            <p style='color:#80cbc4; margin-top:0.5rem; font-size:0.9rem'>
            Model used: <b>{best_name}</b> &nbsp;·&nbsp; R² = {results[best_name]['R²']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Price context gauge
        st.markdown("<br>", unsafe_allow_html=True)
        avg_price = float(df["price"].mean())
        pct = min(pred_price / (avg_price * 2.5), 1.0)
        st.markdown("**Price relative to dataset average:**")
        st.progress(pct)

        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset Avg Price", f"₹{avg_price:,.0f}")
        c2.metric("Your Predicted Price", f"₹{pred_price:,.0f}")
        delta = pred_price - avg_price
        c3.metric("Difference from Avg", f"₹{abs(delta):,.0f}",
                  delta_color="inverse" if delta < 0 else "normal",
                  delta=f"{'below' if delta < 0 else 'above'} avg")

        # ── Similar flights from dataset
        with st.expander("🔍 Similar Flights in Dataset (top 10 by price proximity)"):
            filters = {}
            if "airline" in df.columns:
                filters["airline"] = inp_airline
            if "class" in df.columns:
                filters["class"] = inp_class
            similar = df.copy()
            for col, val in filters.items():
                similar = similar[similar[col].astype(str) == str(val)]
            if not similar.empty:
                similar["price_diff"] = (similar["price"] - pred_price).abs()
                cols_show = [c for c in ["airline", "source_city", "destination_city",
                                          "stops", "duration", "days_left", "class", "price"]
                             if c in similar.columns]
                st.dataframe(
                    similar.nsmallest(10, "price_diff")[cols_show].reset_index(drop=True),
                    use_container_width=True,
                )
            else:
                st.info("No similar flights found with those exact filters.")
