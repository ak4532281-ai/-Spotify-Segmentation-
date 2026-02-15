
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading Spotify Dataset...\n")

df = pd.read_csv("spotify dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())


print("\nChecking Missing Values...\n")
print(df.isnull().sum())

df = df.dropna()

print("\nAfter Cleaning Shape:", df.shape)

target_col = "playlist_genre"

drop_cols = ["track_id", "artists", "album_name", "track_name"]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)


X = df.drop(target_col, axis=1)

X = X.select_dtypes(include=["int64", "float64"])
y = df[target_col]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nGenres Encoded Successfully!")
print("Total Genres:", len(encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nTraining Random Forest Model...\n")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("====================================")
print("Model Accuracy:", accuracy * 100, "%")
print("====================================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

def predict_genre(sample_data):
    """
    Predict genre for a new song input
    """
    sample_scaled = scaler.transform([sample_data])
    pred = model.predict(sample_scaled)
    genre_name = encoder.inverse_transform(pred)
    return genre_name[0]


sample_song = X.iloc[0].values
predicted_genre = predict_genre(sample_song)

print("\nExample Prediction:")
print("Predicted Genre:", predicted_genre)

importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,6))
plt.barh(feature_names, importances)
plt.title("Feature Importance for Genre Classification")
plt.xlabel("Importance Score")
plt.show()
