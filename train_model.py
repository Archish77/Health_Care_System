import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart.csv")

print("Shape:", df.shape)
print("\nTarget Distribution:\n", df["target"].value_counts())

df["chol"] = pd.cut(
    df["chol"],
    bins=[0, 200, 239, 1000],
    labels=[0, 1, 2]
).astype(int)

df["trestbps"] = pd.cut(
    df["trestbps"],
    bins=[0, 120, 139, 300],
    labels=[0, 1, 2]
).astype(int)

df["oldpeak"] = pd.cut(
    df["oldpeak"],
    bins=[-1, 1.0, 2.5, 10],
    labels=[0, 1, 2]
).astype(int)

df["thalach"] = pd.cut(
    df["thalach"],
    bins=[0, 119, 149, 300],
    labels=[2, 1, 0]
).astype(int)

print("\nBinning Applied Successfully")

X = df.drop("target", axis=1)
y = df["target"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\nModel Saved")