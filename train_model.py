import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load processed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Pastikan kolom target sudah benar (boolean) setelah encoding
y_train = y_train.astype(bool)
y_test = y_test.astype(bool)

# Periksa apakah kolom Course sudah dimodifikasi dengan kategori 'Others'
# Jika sudah, lanjutkan untuk training model

# Train models for each condition
conditions = ['Treatment_Seeked']
models = {}

for condition in conditions:
    print(f"Training model for {condition}...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train[condition])
    models[condition] = model
    
    # Evaluate model
    predictions = model.predict(X_test)
    print(f"Classification Report for {condition}:\n")
    print(classification_report(y_test[condition], predictions))

    # Save the model with valid filename (replace ? with _)
    valid_condition = condition.replace('?', '').replace(' ', '_')
    joblib.dump(model, f"{valid_condition}.pkl")

print("Model training completed and saved.")
