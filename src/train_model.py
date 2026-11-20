"""Train the Random Forest IDS model from the included NSL-KDD CSV."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nsl_kdd_train.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'class','difficulty_level'
]
FEATURES = ['protocol_type', 'src_bytes', 'dst_bytes', 'count']

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    df = pd.read_csv(DATA, header=None, names=COLUMNS)
    X = df[FEATURES].copy()
    X['protocol_type'] = X['protocol_type'].astype(str).str.lower()
    for col in ['src_bytes', 'dst_bytes', 'count']:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    y = (df['class'].astype(str).str.strip().str.lower() != 'normal').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ('protocol', OneHotEncoder(handle_unknown='ignore'), ['protocol_type']),
        ('numeric', 'passthrough', ['src_bytes', 'dst_bytes', 'count'])
    ])
    model = Pipeline([
        ('preprocess', prep),
        ('classifier', RandomForestClassifier(
            n_estimators=150, random_state=42,
            class_weight='balanced_subsample', n_jobs=-1
        ))
    ])
    print(f"Training on {len(X_train):,} rows; testing on {len(X_test):,} rows...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Holdout accuracy: {accuracy:.4%}")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Attack'], zero_division=0))
    joblib.dump(model, MODEL_DIR / 'ids_random_forest.joblib')

    report = {
        'records': int(len(df)), 'features': FEATURES,
        'accuracy': float(accuracy),
        'classification_report': classification_report(
            y_test, predictions, target_names=['Normal', 'Attack'],
            output_dict=True, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'note': 'Random holdout from supplied training data, not an independent NSL-KDD test-set result.'
    }
    (REPORT_DIR / 'evaluation.json').write_text(json.dumps(report, indent=2))
    names = model.named_steps['preprocess'].get_feature_names_out()
    values = model.named_steps['classifier'].feature_importances_
    order = np.argsort(values)[::-1]
    plt.figure(figsize=(9, 5))
    plt.bar([names[i] for i in order], values[order])
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Importance')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'feature_importance.png', dpi=160)
    print('Saved model and evaluation reports.')

if __name__ == '__main__':
    main()
