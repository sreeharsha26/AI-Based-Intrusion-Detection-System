# AI-Based Network Intrusion Detection System

A learning project that trains a Random Forest classifier on the supplied NSL-KDD
connection-record dataset and includes an experimental Scapy packet monitor.

## Features
- Binary classification: `normal` versus any attack label.
- Preprocessing with a scikit-learn pipeline and one-hot encoding for protocol type.
- Random Forest training and holdout evaluation.
- Feature-importance chart and JSON evaluation report.
- Experimental live packet-flow aggregation and console alerts.

## Project structure
```text
AI-Based-Intrusion-Detection-System/
├── data/nsl_kdd_train.csv
├── models/ids_random_forest.joblib
├── reports/evaluation.json
├── reports/feature_importance.png
├── src/train_model.py
├── src/realtime_ids.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements
- Python 3.10 or later recommended
- Windows, Linux, or macOS for training
- Npcap on Windows for Scapy packet capture

## Setup (Windows CMD)
```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model
```bat
python src\train_model.py
```
The script reads `data/nsl_kdd_train.csv`, trains the model, and updates
`models/ids_random_forest.joblib` and the files in `reports/`.

## Run the experimental live monitor
Open CMD as Administrator if required for packet capture:
```bat
python src\realtime_ids.py
```
To select a network interface:
```bat
python src\realtime_ids.py --iface "Wi-Fi"
```
Find interface names with `python -c "from scapy.all import show_interfaces; show_interfaces()"`.

## Dataset and labels
The included CSV is the NSL-KDD training data supplied with this project.
It contains 41 connection attributes, an attack/normal label, and a difficulty
field. The model uses four selected fields: `protocol_type`, `src_bytes`,
`dst_bytes`, and `count`. The target is binary: `normal` is class 0; all
non-normal attack names are class 1.

## Important limitations
- The reported metrics are from a random holdout split of the supplied training
  file. They are not an independent evaluation on `KDDTest+`.
- NSL-KDD features describe network connections and traffic context. Live packet
  values are approximations; packet length and flow aggregation do not reproduce
  every NSL-KDD feature exactly. The live monitor is therefore experimental and
  must not be represented as a validated production IDS.
- The model performs binary normal/attack classification, not identification of
  individual attack families.
- Use packet capture only on networks and devices you own or are authorized to monitor.

## Reproducibility
Random seed: 42. Test split: 25%. Classifier: Random Forest with 150 trees.
Current generated holdout accuracy: 97.8917%.

## License / dataset note
This repository includes the dataset file supplied by the project owner.
Before redistributing it publicly, verify the dataset's original terms and
attribution requirements. The source dataset is commonly distributed as NSL-KDD.
