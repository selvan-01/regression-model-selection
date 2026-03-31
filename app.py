# ============================================================
# 🚀 Regression Model Selection - Advanced Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score


# ============================================================
# 🎨 Page Configuration
# ============================================================
st.set_page_config(
    page_title="Regression Model Selector",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 🎨 Custom CSS Styling (Modern UI)
# ============================================================
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #1f4037, #99f2c8);
        }
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
        }
        .sub-title {
            font-size: 18px;
            color: #f0f0f0;
            text-align: center;
        }
        .stButton>button {
            background-color: #ff7b00;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #ffffff;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 🏷️ Title Section
# ============================================================
st.markdown('<div class="main-title">📊 Regression Model Selection App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Compare Multiple ML Models & Find the Best One</div>', unsafe_allow_html=True)

st.write("")


# ============================================================
# 📂 File Upload
# ============================================================
st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])


if file is not None:
    dataset = pd.read_csv(file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(dataset.head())


    # ============================================================
    # ⚙️ Feature Selection
    # ============================================================
    st.sidebar.header("⚙️ Configuration")

    target_column = st.sidebar.selectbox("Select Target Column", dataset.columns)

    X = dataset.drop(columns=[target_column]).values
    y = dataset[target_column].values


    # ============================================================
    # 📊 Train Test Split
    # ============================================================
    test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )


    # ============================================================
    # 🤖 Model Training Button
    # ============================================================
    if st.button("🚀 Train & Compare Models"):

        # Linear Regression
        modelLR = LinearRegression()
        modelLR.fit(X_train, y_train)

        # Polynomial Regression
        poly_reg = PolynomialFeatures(degree=4)
        X_poly = poly_reg.fit_transform(X_train)
        modelPLR = LinearRegression()
        modelPLR.fit(X_poly, y_train)

        # Random Forest
        modelRFR = RandomForestRegressor(n_estimators=50, random_state=0)
        modelRFR.fit(X_train, y_train)

        # Decision Tree
        modelDTR = DecisionTreeRegressor(random_state=0)
        modelDTR.fit(X_train, y_train)

        # SVR (with scaling ✅)
        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X_train_scaled = sc_X.fit_transform(X_train)
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1))

        modelSVR = SVR(kernel='rbf')
        modelSVR.fit(X_train_scaled, y_train_scaled.ravel())


        # ============================================================
        # 📈 Predictions
        # ============================================================
        modelLR_pred  = modelLR.predict(X_test)
        modelPLR_pred = modelPLR.predict(poly_reg.transform(X_test))
        modelRFR_pred = modelRFR.predict(X_test)
        modelDTR_pred = modelDTR.predict(X_test)

        X_test_scaled = sc_X.transform(X_test)
        modelSVR_pred = sc_y.inverse_transform(
            modelSVR.predict(X_test_scaled).reshape(-1, 1)
        )


        # ============================================================
        # 📊 Accuracy Results
        # ============================================================
        results = {
            "Linear Regression": r2_score(y_test, modelLR_pred),
            "Polynomial Regression": r2_score(y_test, modelPLR_pred),
            "Random Forest": r2_score(y_test, modelRFR_pred),
            "Decision Tree": r2_score(y_test, modelDTR_pred),
            "Support Vector Regression": r2_score(y_test, modelSVR_pred)
        }

        results_df = pd.DataFrame(list(results.items()), columns=["Model", "R2 Score"])


        # ============================================================
        # 🏆 Best Model Highlight
        # ============================================================
        best_model = max(results, key=results.get)

        st.success(f"🏆 Best Model: {best_model}")


        # ============================================================
        # 📊 Display Results
        # ============================================================
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(results_df)


        # ============================================================
        # 📉 Visualization
        # ============================================================
        st.subheader("📈 Performance Chart")
        st.bar_chart(results_df.set_index("Model"))


else:
    st.info("👈 Upload a dataset to get started")