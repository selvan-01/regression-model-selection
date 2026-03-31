# ============================================================
# 📌 Project: Regression Model Selection
# 📊 Goal: Compare multiple regression algorithms and evaluate performance
# ============================================================


# ============================================================
# 🔹 1. Import Required Libraries
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 🔹 2. Load Dataset from Local Directory
# ============================================================
from google.colab import files
uploaded = files.upload()

# Dataset Description:
# AT  -> Ambient Temperature (°C)
# V   -> Exhaust Vacuum (cm Hg)
# AP  -> Ambient Pressure (millibar)
# RH  -> Relative Humidity (%)
# PE  -> Net Hourly Electrical Energy Output (MW)

dataset = pd.read_csv('dataset.csv')


# ============================================================
# 🔹 3. Split Features (X) and Target (y)
# ============================================================
X = dataset.iloc[:, :-1].values   # Independent variables
y = dataset.iloc[:, -1].values    # Dependent variable

# Reshape for SVR (if needed later)
ysvm = y.reshape(len(y), 1)


# ============================================================
# 🔹 4. Split Dataset into Training and Testing Sets
# ============================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


# ============================================================
# 🔹 5. Import Machine Learning Models
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# ============================================================
# 🔹 6. Initialize Models
# ============================================================

# Linear Regression
modelLR = LinearRegression()

# Polynomial Regression (degree = 4)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
modelPLR = LinearRegression()

# Random Forest Regression
modelRFR = RandomForestRegressor(n_estimators=10, random_state=0)

# Decision Tree Regression
modelDTR = DecisionTreeRegressor(random_state=0)

# Support Vector Regression (RBF Kernel)
modelSVR = SVR(kernel='rbf')


# ============================================================
# 🔹 7. Train Models
# ============================================================
modelLR.fit(X_train, y_train)
modelPLR.fit(X_poly, y_train)
modelRFR.fit(X_train, y_train)
modelDTR.fit(X_train, y_train)
modelSVR.fit(X_train, y_train)


# ============================================================
# 🔹 8. Make Predictions on Test Data
# ============================================================
modelLRy_pred  = modelLR.predict(X_test)
modelPLRy_pred = modelPLR.predict(poly_reg.transform(X_test))
modelRFRy_pred = modelRFR.predict(X_test)
modelDTRy_pred = modelDTR.predict(X_test)
modelSVRy_pred = modelSVR.predict(X_test)


# ============================================================
# 🔹 9. Evaluate Model Performance using R² Score
# ============================================================
from sklearn.metrics import r2_score

print("📊 Model Performance Comparison:\n")

print("Linear Regression Accuracy: {:.4f}".format(r2_score(y_test, modelLRy_pred)))
print("Polynomial Regression Accuracy: {:.4f}".format(r2_score(y_test, modelPLRy_pred)))
print("Random Forest Regression Accuracy: {:.4f}".format(r2_score(y_test, modelRFRy_pred)))
print("Decision Tree Regression Accuracy: {:.4f}".format(r2_score(y_test, modelDTRy_pred)))
print("Support Vector Regression Accuracy: {:.4f}".format(r2_score(y_test, modelSVRy_pred)))


# ============================================================
# ✅ End of Project
# ============================================================