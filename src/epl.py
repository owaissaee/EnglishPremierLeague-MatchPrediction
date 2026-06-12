import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, recall_score

RND = 42
P = "/home/owais-saeed/python/epl_data/Datasets/"
OUT = "/home/owais-saeed/python/ML/EPL/"
os.makedirs(OUT, exist_ok=True)

Datasets = [
    "2000-01.csv", "2001-02.csv", "2002-03.csv", "2003-04.csv", "2004-05.csv",
    "2005-06.csv", "2006-07.csv", "2007-08.csv", "2008-09.csv", "2009-10.csv",
    "2010-11.csv", "2011-12.csv", "2012-13.csv", "2013-14.csv", "2014-15.csv",
    "2015-16.csv", "2016-17.csv", "2017-18.csv", "2018-19.csv", "2019-20.csv"
]

df = pd.read_csv(P + "final_dataset.csv", index_col=0)

raw = pd.concat(
    [pd.read_csv(P + f, encoding="latin-1")[["Date", "HomeTeam", "AwayTeam", "FTR"]]
     for f in Datasets],
    ignore_index=True
)
raw = raw.drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
key = raw.set_index(["Date", "HomeTeam", "AwayTeam"])["FTR"]
df["FTR3"] = df.set_index(["Date", "HomeTeam", "AwayTeam"]).index.map(key)

assert df["FTR3"].isna().sum() == 0, "label recovery incomplete"
assert ((df["FTR3"] == "H") == (df["FTR"] == "H")).all()

form_map = {"W": 3, "D": 1, "L": 0, "M": 0}
form_cols = ["HM1", "HM2", "HM3", "HM4", "HM5", "AM1", "AM2", "AM3", "AM4", "AM5"]
for c in form_cols:
    df[c] = df[c].map(form_map)

for c, nc in [
    ("HTP", "HTP_n"), ("ATP", "ATP_n"), ("HTGD", "HTGD_n"),
    ("ATGD", "ATGD_n"), ("DiffPts", "DiffPts_n"), ("DiffFormPts", "DiffFormPts_n")
]:
    df[nc] = df[c] / df["MW"]

base_feats = [
    "HTGS", "ATGS", "HTGC", "ATGC", "HTP_n", "ATP_n",
    "HM1", "HM2", "HM3", "HM4", "HM5", "AM1", "AM2", "AM3", "AM4", "AM5",
    "HTFormPts", "ATFormPts", "HTWinStreak3", "HTWinStreak5", "HTLossStreak3",
    "HTLossStreak5", "ATWinStreak3", "ATWinStreak5", "ATLossStreak3", "ATLossStreak5",
    "HTGD_n", "ATGD_n", "DiffPts_n", "DiffFormPts_n"
]

y = df["FTR3"]

odds = pd.concat(
    [pd.read_csv(P + f, encoding="latin-1")[["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]]
     for f in ["2015-16.csv", "2016-17.csv", "2017-18.csv"]],
    ignore_index=True
)
inv = 1 / odds[["B365H", "B365D", "B365A"]]
ssum = inv.sum(axis=1)
odds["pH"] = inv.B365H / ssum
odds["pD"] = inv.B365D / ssum
odds["pA"] = inv.B365A / ssum
ok = odds.set_index(["Date", "HomeTeam", "AwayTeam"])[["pH", "pD", "pA"]]
idx = df.set_index(["Date", "HomeTeam", "AwayTeam"]).index
for col in ["pH", "pD", "pA"]:
    df[col] = idx.map(ok[col]).astype(float)


def make_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=RND),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", gamma="scale", random_state=RND),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, alpha=1e-3, random_state=RND),
    }


def run(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    Xtr_scaled = sc.transform(Xtr)
    Xte_scaled = sc.transform(Xte)
    out = {}
    for name, clf in make_models().items():
        clf.fit(Xtr_scaled, ytr)
        pred = clf.predict(Xte_scaled)
        out[name] = {
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            "precision": precision_score(yte, pred, average="macro", zero_division=0),
            "recall": recall_score(yte, pred, average="macro", zero_division=0),
            "pred": pred,
        }
    return out


test_mask = df.index >= 5700
train_mask = ~test_mask

XA = df[base_feats].astype(float)
rA = run(XA[train_mask], y[train_mask], XA[test_mask], y[test_mask])

print("=" * 72)
print("GLOBAL 3-WAY MODEL  (train 2000-2015, test 2015-2018)")
print("=" * 72)
print("Class distribution (test):", y[test_mask].value_counts().to_dict())
print("Majority baseline (predict H):", f"{(y[test_mask] == 'H').mean():.4f}\n")
print(f"{'Classifier':<22}{'Accuracy':>11}{'MacroF1':>10}")
for n in make_models():
    print(f"{n:<22}{rA[n]['accuracy']:>11.4f}{rA[n]['macro_f1']:>10.4f}")

best = max(rA, key=lambda k: rA[k]["accuracy"])

odds_test = df.loc[test_mask, ["pH", "pD", "pA"]]
have_odds = odds_test.notna().all(axis=1)
odds_pred = odds_test[have_odds].idxmax(axis=1).map({"pH": "H", "pD": "D", "pA": "A"})
odds_acc = accuracy_score(y[test_mask][have_odds], odds_pred)
odds_f1 = f1_score(y[test_mask][have_odds], odds_pred, average="macro", zero_division=0)
print(f"\nBet365 odds-implied predictor: acc={odds_acc:.4f}, macroF1={odds_f1:.4f}")
print(f"Best trained classifier ({best}): acc={rA[best]['accuracy']:.4f}, macroF1={rA[best]['macro_f1']:.4f}")

labels = ["A", "D", "H"]
print("\nBest classifier confusion matrix [A,D,H]:")
print(confusion_matrix(y[test_mask], rA[best]["pred"], labels=labels))
print(classification_report(y[test_mask], rA[best]["pred"], labels=labels, zero_division=0))

mm = df.Date.str[3:5].astype(int)
yy = df.Date.str[-2:].astype(int)
df["Season"] = np.where(mm >= 7, 2000 + yy, 2000 + yy - 1)

rows = []
for s in sorted(df.Season.unique()):
    sub = df[df.Season == s]
    if len(sub) < 200:
        continue
    Xs = sub[base_feats].astype(float)
    ys = sub["FTR3"]
    cut = int(len(sub) * 0.8)
    res = run(Xs.iloc[:cut], ys.iloc[:cut], Xs.iloc[cut:], ys.iloc[cut:])
    bm = max(res, key=lambda k: res[k]["accuracy"])
    rows.append((s, bm, res[bm]["accuracy"], np.mean([res[k]["accuracy"] for k in res])))

pr = pd.DataFrame(rows, columns=["Season", "Best model", "Best acc", "Mean acc(5)"])
print("\nPER-SEASON 3-WAY MODELS")
print(pr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

summary_rows = []
per_class_rows = []
for name in make_models():
    pred = rA[name]["pred"]
    summary_rows.append({
        "Classifier": name,
        "Accuracy": rA[name]["accuracy"],
        "MacroF1": rA[name]["macro_f1"],
        "Precision": rA[name]["precision"],
        "Recall": rA[name]["recall"],
    })
    rep = classification_report(y[test_mask], pred, labels=labels, output_dict=True, zero_division=0)
    per_class_rows.append({
        "Classifier": name,
        "Away Win": rep["A"]["f1-score"],
        "Draw": rep["D"]["f1-score"],
        "Home Win": rep["H"]["f1-score"],
    })
    cm = confusion_matrix(y[test_mask], pred, labels=labels)
    pd.DataFrame(cm, index=["Actual Away", "Actual Draw", "Actual Home"], columns=["Pred Away", "Pred Draw", "Pred Home"]).to_csv(
        os.path.join(OUT, f"confusion_matrix_{name.lower().replace(' ', '_')}.csv")
    )

pd.DataFrame(summary_rows + [{"Classifier": "Bet365 odds (test-only)", "Accuracy": odds_acc, "MacroF1": odds_f1}]).to_csv(
    os.path.join(OUT, "results_global_3way.csv"), index=False
)
pd.DataFrame(per_class_rows).to_csv(os.path.join(OUT, "results_per_class_f1.csv"), index=False)
pr.to_csv(os.path.join(OUT, "results_per_season_3way.csv"), index=False)
print("\nSaved CSV files in", OUT)
