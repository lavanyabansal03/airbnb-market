import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURATION ---
# construct paths relative to this script so running from anywhere works
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "combined_airbnb.csv")
# if you want to train on a specific city dataset change the filename above
TARGET_COLUMN = "price"                # Change if your target column has a different name
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "airbnb_price_model.pkl")
COLUMNS_FILE = os.path.join(MODEL_DIR, "model_columns.pkl")

# --- MAKE SURE MODEL DIR EXISTS ---
os.makedirs(MODEL_DIR, exist_ok=True)

# --- LOAD DATA ---
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file not found: {DATA_FILE}\n" \
        "make sure you're running the script from the repo or adjust the path")
df = pd.read_csv(DATA_FILE)

# --- PREPROCESSING ---
# drop rows with missing target
df = df.dropna(subset=[TARGET_COLUMN])

# convert categorical fields to dummy variables (one‑hot) so that
# the RandomForestRegressor can consume them directly
categorical = df.select_dtypes(include=["object"]).columns.to_list()
if TARGET_COLUMN in categorical:
    categorical.remove(TARGET_COLUMN)

# drop any free‑text columns that are unlikely to help (e.g. name)
for col in ["name"]:
    if col in df.columns:
        df = df.drop(columns=[col])
        if col in categorical:
            categorical.remove(col)

if categorical:
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

# --- PREPARE FEATURES AND TARGET ---
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Optional: split into train/test if needed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- TRAIN MODEL ---
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

rf.fit(X_train, y_train)

# --- SAVE MODEL ---
with open(MODEL_FILE, "wb") as f:
    pickle.dump(rf, f)

# --- SAVE FEATURE COLUMNS ---
model_columns = list(X_train.columns)
with open(COLUMNS_FILE, "wb") as f:
    pickle.dump(model_columns, f)

print(f"Model saved to: {MODEL_FILE}")
print(f"Model columns saved to: {COLUMNS_FILE}")
