import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _cle(s):
    """Convertit une colonne numérique en texte stable (315.0 -> '315', NaN -> 'nan')."""
    return pd.to_numeric(s, errors="coerce").round(0).astype("Int64").astype(str)


def _texte(s):
    return s.astype("object").where(s.notna(), np.nan)


class FraudFeatures(BaseEstimator, TransformerMixin):
    FREQ_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "R_emaildomain",
                 "DeviceInfo", "id_30", "id_31", "id_33", "card1_addr1", "uid"]
    AGG_GROUPES = ["card1", "card1_addr1", "uid"]
    AGG_COLS = ["TransactionAmt", "D15", "C13", "dist1"]
    EXCLURE = ["TransactionID", "isFraud", "TransactionDT", "day", "D1n", "uid", "card1_addr1"]

    def __init__(self, colonnes=None):
        self.colonnes = colonnes

    # --- variables calculées ligne par ligne (rien n'est appris)
    def _base(self, X):
        X = X.copy()
        for c in X.columns:                       # même nettoyage du texte qu'au prétraitement
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = _texte(X[c]).str.strip().str.lower()
        X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])
        X["Amt_cents"] = (X["TransactionAmt"] - np.floor(X["TransactionAmt"])).round(3)
        X["day"] = X["TransactionDT"] // 86400
        X["hour"] = (X["TransactionDT"] // 3600) % 24
        X["dow"] = X["day"] % 7
        X["D1n"] = X["day"] - X["D1"]
        X["card1_addr1"] = _cle(X["card1"]) + "_" + _cle(X["addr1"])
        X["uid"] = X["card1_addr1"] + "_" + _cle(X["D1n"])
        for c in ["P_emaildomain", "R_emaildomain"]:
            if c in X:
                X[c + "_prov"] = _texte(X[c]).str.split(".").str[0]
        if "P_emaildomain" in X and "R_emaildomain" in X:
            X["email_match"] = np.where(X["R_emaildomain"].isna(), np.nan,
                                        (X["P_emaildomain"] == X["R_emaildomain"]).astype(float))
        if "DeviceInfo" in X:
            X["Device_marque"] = _texte(X["DeviceInfo"]).str.split(r"[ /_\-]").str[0]
        if "id_30" in X:
            X["OS"] = _texte(X["id_30"]).str.split(" ").str[0]
        if "id_31" in X:
            X["navigateur"] = _texte(X["id_31"]).str.split(" ").str[0]
        return X

    # --- application de ce qui a été appris au fit
    def _encoder(self, X):
        nouvelles = {}
        for c, vc in self.freq_maps_.items():
            nouvelles[c + "_freq"] = X[c].map(vc).astype("float32")
        for (g, c), agg in self.agg_maps_.items():
            m, s = X[g].map(agg["mean"]), X[g].map(agg["std"])
            nouvelles[f"{c}_mean_{g}"], nouvelles[f"{c}_std_{g}"] = m, s
            if c == "TransactionAmt":
                nouvelles[f"Amt_ratio_{g}"] = X[c] / m
        X = pd.concat([X, pd.DataFrame(nouvelles, index=X.index)], axis=1)
        for c, mp in self.cat_maps_.items():
            enc = X[c].map(mp).astype("float32")
            enc[X[c].notna() & enc.isna()] = -1          # modalité jamais vue au train
            X[c] = enc
        return X

    def fit(self, X, y=None):
        self.input_cols_ = list(self.colonnes) if self.colonnes is not None else list(X.columns)
        X = self._base(X.reindex(columns=self.input_cols_))
        self.freq_maps_ = {c: X[c].value_counts() for c in self.FREQ_COLS if c in X}
        self.agg_maps_ = {(g, c): X.groupby(g)[c].agg(["mean", "std"])
                          for g in self.AGG_GROUPES for c in self.AGG_COLS if c in X}
        self.cat_cols_ = [c for c in X.columns
                          if not pd.api.types.is_numeric_dtype(X[c]) and c not in self.EXCLURE]
        self.cat_maps_ = {c: {v: i for i, v in enumerate(X[c].dropna().unique())} for c in self.cat_cols_}
        Xt = self._encoder(X)
        self.features_ = [c for c in Xt.columns if c not in self.EXCLURE]
        return self

    def transform(self, X):
        X = self._base(X.reindex(columns=self.input_cols_))
        X = self._encoder(X)
        return X.reindex(columns=self.features_).astype("float32")
