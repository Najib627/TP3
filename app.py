import joblib
import numpy as np
import pandas as pd
import streamlit as st

services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# doit exister pour que joblib puisse recharger le pipeline
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["NbServices"] = (data[services] == "Yes").sum(axis=1) + (data["InternetService"] != "No").astype(int)
    data["AvgCharge"] = np.where(data["tenure"] > 0, data["TotalCharges"] / data["tenure"].replace(0, 1), data["MonthlyCharges"])
    data["TenureGroup"] = pd.cut(data["tenure"], bins=[-1, 12, 24, 48, 72],
                                 labels=["0-1an", "1-2ans", "2-4ans", "4-6ans"]).astype(str)
    data["MonthlyEcheck"] = ((data["Contract"] == "Month-to-month") &
                             (data["PaymentMethod"] == "Electronic check")).map({True: "Yes", False: "No"})
    return data

import __main__
__main__.feature_engineering = feature_engineering

@st.cache_resource
def charger():
    return joblib.load("model/churn_model.joblib")

art = charger()
st.title(" Prédiction du churn client — Telco")
st.caption(f"Modèle : {art['nom']} — seuil de décision : {art['seuil']:.2f}")

yn = ["No", "Yes"]
c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox("Genre", ["Female", "Male"])
    senior = st.selectbox("Senior (65+)", yn)
    partner = st.selectbox("Partenaire", yn)
    dependents = st.selectbox("Personnes à charge", yn)
    tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
    contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
with c2:
    phone = st.selectbox("Téléphone", yn, index=1)
    multiple = st.selectbox("Lignes multiples", yn)
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Sécurité en ligne", yn)
    backup = st.selectbox("Sauvegarde en ligne", yn)
    protection = st.selectbox("Protection appareil", yn)
with c3:
    support = st.selectbox("Support technique", yn)
    tv = st.selectbox("Streaming TV", yn)
    movies = st.selectbox("Streaming films", yn)
    paperless = st.selectbox("Facture dématérialisée", yn, index=1)
    payment = st.selectbox("Paiement", ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Montant mensuel ($)", 18.0, 120.0, 70.0)

if st.button("Prédire", type="primary"):
    client = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": monthly * tenure}])
    proba = art["pipeline"].predict_proba(client)[0, 1]
    st.metric("Probabilité de churn", f"{proba:.1%}")
    if proba >= art["seuil"]:
        st.error(" Client à risque : il risque de résilier.")
    else:
        st.success("Client probablement fidèle.")
