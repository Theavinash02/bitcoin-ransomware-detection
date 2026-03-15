# Bitcoin Ransomware Detection

A machine learning web application that detects Bitcoin ransomware activity by classifying Bitcoin addresses based on transaction graph features.

Built as a Final Year Project (B.Tech Information Technology, Anna University, 2024).

---

## Problem Statement

Bitcoin has been widely used by ransomware operators to collect payments anonymously. This project uses the [BitcoinHeist dataset](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) to train a Random Forest classifier that identifies whether a Bitcoin address is associated with known ransomware families or is benign (`white`).

## Dataset

**BitcoinHeist Ransomware Address Dataset** — UCI Machine Learning Repository

| Feature | Description |
|---|---|
| `year` | Year of transaction |
| `day` | Day of year (1–365) |
| `length` | Length of the transaction chain |
| `weight` | Fraction of BTC received from a single source |
| `count` | Number of transactions at the address |
| `looped` | Whether coins were sent back to the source |
| `neighbors` | Number of distinct addresses transacted with |
| `income` | Total BTC income received |

**Target:** Ransomware family label (e.g. `montrealgang`, `paduacoin`, `white` for benign)

## Model

- Algorithm: Random Forest Classifier (scikit-learn)
- Train/test split: 80/20
- Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix

> To regenerate the model, run `bitcoin_ransomwere.ipynb` end-to-end.  
> The `.pkl` file is excluded from this repository via `.gitignore`.

## Project Structure

```
├── main.py                    # Flask web application
├── bitcoin_ransomwere.ipynb   # Model training notebook
├── templates/
│   ├── admin.html             # Login page
│   ├── index1.html            # Prediction form
│   └── result.html            # Prediction result
├── static/
│   └── css/style.css
├── requirements.txt
└── .gitignore
```

## Running Locally

```bash
pip install -r requirements.txt

# First: run the notebook to generate model.pkl
jupyter notebook bitcoin_ransomwere.ipynb

# Then: start the Flask app
python main.py
```

Visit `http://localhost:2000`

## Requirements

```
flask
scikit-learn
pandas
numpy
joblib
```

---

## Note

This is v1 of the project - the original final year submission.  
A research-grade v2 is in progress, adding XGBoost comparison, SHAP explainability, and a Bayesian Network model using [pgmpy](https://github.com/pgmpy/pgmpy).
