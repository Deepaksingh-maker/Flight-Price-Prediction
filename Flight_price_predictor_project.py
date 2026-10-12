import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="✈️ Flight Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>

.hero{
    background: linear-gradient(135deg,#1E3A8A,#2563EB);
    padding:30px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-bottom:25px;
}

.card{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
    border:1px solid #374151;
    margin-bottom:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,.25);
}

.card-title{
    font-size:22px;
    font-weight:bold;
    color:#60A5FA;
    margin-bottom:10px;
}

.sidebar-title{
    font-size:28px;
    font-weight:bold;
    color:#60A5FA;
    text-align:center;
}

</style>
            
""", unsafe_allow_html=True)
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
#----------------
# SIDEBAR
#----------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">✈️ Flight Price AI</p>', unsafe_allow_html=True)
    st.divider()

    file = st.file_uploader(
        "📂 Upload Flight_clean_dataset.csv",
        type=["csv"],
        help="Upload the dataset if it's not in the same folder as Flight_price_predictor_project.py",
    )

    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dataset Analysis", "🤖 Model Training", "🔮 Price Prediction"],
        label_visibility="collapsed",
    )
    

if file is None:
    st.error(
        "⚠️  **Dataset not found.**  "
        "Please upload `Flight_clean_dataset.csv` using the sidebar uploader, "
        "or place the file in the same directory as `Flight_price_predictor_project.py` and restart."
    )
    st.stop()

df = pd.read_csv(file)


#----------------
#-PAGE 1 : HOME
#----------------
if page == "🏠 Home":
    st.markdown("""
    <div class='hero'>
        <h1>✈️ Flight Price Prediction</h1>
        <p>An end-to-end Machine Learning dashboard — from raw data to real-time price prediction</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> 📋 Total Records </p>
        <h3>{len(df):,}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

    c2.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> 📊 Features </p>
        <h3>{len(df.columns)-1}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    c3.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> 💰 Avg Price </p>
        <h3>₹ {df['price'].mean():,.0f}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    c4.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> ✈️ Airlines </p>
        <h3>{df['airline'].nunique()}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    
    st.divider()

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
    

#--------------------------
#-PAGE 2 : DATASET ANALYSIS
#--------------------------
elif page == "📊 Dataset Analysis":
    st.markdown("## 📊 Dataset Analysis")
    with st.expander("📋 Preview Raw Dataset", expanded=True):
        n_rows = st.slider("Rows to display", 5, 100, 20, key="preview_slider")
        st.dataframe(df.head(n_rows), use_container_width=True)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> Row </p>
        <h3>{df.shape[0]:,}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    c2.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> Columns </p>
        <h3>{df.shape[1]}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    c3.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> Missing Values </p>
        <h3>{int(df.isnull().sum().sum())}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    c4.markdown(
            f"""
        <div style="background:#1e293b;padding:10px;border-radius:12px;text-align:center;">
        <p> Duplicates Values </p>
        <h3>{int(df.duplicated().sum())}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown("---")
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
    st.markdown("### 🔥 Correlation Heatmap")
    # Sirf numeric columns
    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        title="Correlation Heatmap",
        title_x=0.5,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

#--------------------------
#-PAGE 3 : MODEL TRAINING
#--------------------------

elif page == "🤖 Model Training":

    st.title("🤖 Model Training")
    st.write("Train and compare different Machine Learning Regression models.")

    if st.button("🚀 Train Models"):

        with st.spinner("Training Models..."):

            # -----------------------------
            # Copy Dataset
            # -----------------------------
            df_model = df.copy()

            if "Unnamed: 0" in df_model.columns:
                df_model.drop("Unnamed: 0", axis=1, inplace=True)

            if "flight" in df_model.columns:
                df_model.drop("flight", axis=1, inplace=True)

            # -----------------------------
            # Encode Categorical Columns
            # -----------------------------
            le = LabelEncoder()

            for col in df_model.columns:
                if df_model[col].dtype == "object":
                    df_model[col] = le.fit_transform(df_model[col])

            # -----------------------------
            # Features & Target
            # -----------------------------
            X = df_model.drop("price", axis=1)
            y = df_model["price"]

            # -----------------------------
            # Train Test Split
            # -----------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
            st.session_state["X_columns"] = X.columns
            # ==========================================
            # 1. Linear Regression
            # ==========================================

            linear = LinearRegression()

            linear.fit(X_train, y_train)

            linear_pred = linear.predict(X_test)

            linear_r2 = r2_score(y_test, linear_pred)
            linear_mae = mean_absolute_error(y_test, linear_pred)
            linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))

            # ==========================================
            # 2. Polynomial Regression
            # ==========================================

            poly = PolynomialFeatures(degree=2)

            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            poly_model = LinearRegression()

            poly_model.fit(X_train_poly, y_train)

            poly_pred = poly_model.predict(X_test_poly)

            poly_r2 = r2_score(y_test, poly_pred)
            poly_mae = mean_absolute_error(y_test, poly_pred)
            poly_rmse = np.sqrt(mean_squared_error(y_test, poly_pred))

            # ==========================================
            # 3. Decision Tree
            # ==========================================

            tree = DecisionTreeRegressor(random_state=42)

            tree.fit(X_train, y_train)

            tree_pred = tree.predict(X_test)

            tree_r2 = r2_score(y_test, tree_pred)
            tree_mae = mean_absolute_error(y_test, tree_pred)
            tree_rmse = np.sqrt(mean_squared_error(y_test, tree_pred))
            st.session_state["model"] = tree
            st.session_state["label_encoder"] = le  

        st.success("✅ Training Completed Successfully!")

        # -----------------------------
        # Comparison Table
        # -----------------------------

        result = pd.DataFrame({

            "Model": [
                "Linear Regression",
                "Polynomial Regression",
                "Decision Tree Regressor"
            ],

            "R² Score": [
                linear_r2,
                poly_r2,
                tree_r2
            ],

            "MAE": [
                linear_mae,
                poly_mae,
                tree_mae
            ],

            "RMSE": [
                linear_rmse,
                poly_rmse,
                tree_rmse
            ]

        })

        st.subheader("📊 Model Comparison")

        st.dataframe(
            result.style.format({
                "R² Score": "{:.4f}",
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}"
            }),
            use_container_width=True
        )

        # -----------------------------
        # Best Model
        # -----------------------------

        best_model = result.loc[result["R² Score"].idxmax()]

        st.success(
            f"""
🏆 Best Model : {best_model['Model']}

📈 R² Score : {best_model['R² Score']:.4f}
"""
        )

        # -----------------------------
        # Metrics
        # -----------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric("🏆 Best Model", best_model["Model"])
        c2.metric("📈 Best R² Score", f"{best_model['R² Score']:.4f}")
        c3.metric("📉 RMSE", f"{best_model['RMSE']:.2f}")

#--------------------------
#-PAGE 4 : MODEL PREDICTION
#--------------------------
elif page == "🔮 Price Prediction":

    st.title("🔮 Flight Price Prediction")

    if "model" not in st.session_state:

        st.warning("⚠️ Please train the model first from the 'Model Training' page.")

        st.stop()

    model = st.session_state["model"]

    st.markdown("## 🎛️ Enter Flight Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox(
            "✈️ Airline",
            sorted(df["airline"].unique())
        )

        source = st.selectbox(
            "🛫 Source City",
            sorted(df["source_city"].unique())
        )

        destination = st.selectbox(
            "🛬 Destination City",
            sorted(df["destination_city"].unique())
        )

    with col2:
        departure = st.selectbox(
            "🌅 Departure Time",
            sorted(df["departure_time"].unique())
        )

        arrival = st.selectbox(
            "🌇 Arrival Time",
            sorted(df["arrival_time"].unique())
        )

        stops = st.selectbox(
            "🛑 Number of Stops",
            sorted(df["stops"].unique())
        )

    with col3:
        flight_class = st.selectbox(
            "💺 Class",
            sorted(df["class"].unique())
        )

        duration = st.slider(
            "⏱ Duration (hours)",
            float(df["duration"].min()),
            float(df["duration"].max()),
            float(df["duration"].median())
        )

        days_left = st.slider(
            "📅 Days Left to Departure",
            int(df["days_left"].min()),
            int(df["days_left"].max()),
            int(df["days_left"].median())
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Flight Price", use_container_width=True):

        temp = df.copy()

        # Remove unwanted columns
        if "Unnamed: 0" in temp.columns:
            temp.drop("Unnamed: 0", axis=1, inplace=True)

        if "flight" in temp.columns:
            temp.drop("flight", axis=1, inplace=True)

        # Encode dataset exactly like training
        for col in temp.columns:
            if temp[col].dtype == "object":
                le = LabelEncoder()
                temp[col] = le.fit_transform(temp[col])

        # Encode user inputs
        airline = LabelEncoder().fit(df["airline"]).transform([airline])[0]
        source = LabelEncoder().fit(df["source_city"]).transform([source])[0]
        destination = LabelEncoder().fit(df["destination_city"]).transform([destination])[0]
        departure = LabelEncoder().fit(df["departure_time"]).transform([departure])[0]
        arrival = LabelEncoder().fit(df["arrival_time"]).transform([arrival])[0]
        stops = LabelEncoder().fit(df["stops"]).transform([stops])[0]
        flight_class = LabelEncoder().fit(df["class"]).transform([flight_class])[0]

        input_df = pd.DataFrame({
            "airline": [airline],
            "source_city": [source],
            "departure_time": [departure],
            "stops": [stops],
            "arrival_time": [arrival],
            "destination_city": [destination],
            "class": [flight_class],
            "duration": [duration],
            "days_left": [days_left],
        })

        # Same column order as training
        input_df = input_df.reindex(columns=st.session_state["X_columns"], fill_value=0)

        prediction = model.predict(input_df)

        pred_price = prediction[0]

        # Price Category
        if pred_price < 5000:
            category = "🟢 Budget Flight"
        elif pred_price < 15000:
            category = "🟡 Standard Flight"
        elif pred_price < 30000:
            category = "🟠 Premium Flight"
        else:
            category = "🔴 Luxury Flight"

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                label="💰 Predicted Price",
                value=f"₹ {pred_price:,.2f}"
            )

        with c2:
            st.metric(
                label="🤖 Model Used",
                value="Decision Tree"
            )

        with c3:
            st.metric(
                label="✈️ Category",
                value=category
            )

        st.success("✅ Prediction Completed Successfully!")