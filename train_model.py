from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data.data, data.target)

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
