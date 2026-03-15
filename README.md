# Bitcoin Ransomware Detection

A machine learning web application that detects Bitcoin ransomware activity by classifying Bitcoin addresses based on transaction graph features.

Built as a Final Year Project (B.Tech Information Technology, Anna University, 2024).

🌐 **Live Demo:** [theavinash02.github.io/bitcoin-ransomware-detection](https://theavinash02.github.io/bitcoin-ransomware-detection/)

---

## Problem Statement

Bitcoin has been widely used by ransomware operators to collect payments anonymously. This project uses the [BitcoinHeist dataset](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) to train a Random Forest classifier that identifies whether a Bitcoin address is associated with known ransomware families or is benign (`white`).

---

## Results

Trained on 2,916,697 Bitcoin addresses. Evaluated on a held-out 20% test split.

| Metric | Score |
|---|---|
| Accuracy | **98.87%** |
| Precision (weighted avg) | **0.99** |
| Recall (weighted avg) | **0.99** |
| F1-score (weighted avg) | **0.99** |
| F1-score (macro avg) | 0.13 |
| Training time | ~12 min on CPU |

> **Note on macro F1:** The low macro F1 (0.13) reflects a severe class imbalance - 98.6% of addresses are benign (`white`). Rare ransomware families with very few samples (e.g. `montrealAPT`: 2 test samples) score 0.00. This is addressed in v2 with SMOTE oversampling and a Bayesian Network comparison.

---

## Dataset

**BitcoinHeist Ransomware Address Dataset** - UCI Machine Learning Repository  
*Farrukhjon Masudov et al. - [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*

| Feature | Type | Description |
|---|---|---|
| `year` | int | Year of transaction |
| `day` | int | Day of year (1–365) |
| `length` | int | Length of the transaction chain |
| `weight` | float | Fraction of BTC received from a single source |
| `count` | int | Number of transactions at the address |
| `looped` | binary | Whether coins were sent back to the source |
| `neighbors` | int | Number of distinct addresses transacted with |
| `income` | float | Total BTC income received |

**Target:** 29 ransomware family labels + `white` (benign)  
**Dataset size:** 2,916,697 addresses

---

## Model

- **Algorithm:** Random Forest Classifier (scikit-learn, 100 estimators)
- **Train/test split:** 80/20, random_state=42
- **Evaluation:** Accuracy, Precision, Recall, F1-score (weighted + macro), Confusion Matrix

> To regenerate the model, run `bitcoin_ransomware.ipynb` end-to-end.  
> The `.pkl` file is excluded from this repository via `.gitignore`.  
> The dataset CSV is also excluded - download it from the UCI link above.

---

## Project Structure

```
├── main.py                     # Flask web application
├── bitcoin_ransomware.ipynb    # Model training + evaluation notebook
├── index.html                  # GitHub Pages landing page
├── templates/
│   ├── admin.html              # Login page
│   ├── index1.html             # Prediction form
│   └── result.html             # Prediction result
├── requirements.txt
└── .gitignore
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/theavinash02/bitcoin-ransomware-detection.git
cd bitcoin-ransomware-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Get BitcoinHeistData.csv from:
# https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset
# Place it in the project root folder

# 4. Run the notebook to generate model.pkl
jupyter notebook bitcoin_ransomware.ipynb

# 5. Start the Flask app
python main.py
```

Visit `http://localhost:2000` - login with the credentials set in `main.py`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | scikit-learn RandomForestClassifier |
| Data processing | pandas, numpy |
| Visualisation | seaborn, matplotlib |
| Web framework | Flask |
| Frontend | HTML, CSS (standalone, no framework) |

---

## Roadmap - v2 (in progress)

- [ ] XGBoost and LightGBM comparison against Random Forest
- [ ] SHAP explainability - per-prediction feature importance
- [ ] Bayesian Network model using [pgmpy](https://github.com/pgmpy/pgmpy)
- [ ] SMOTE oversampling to address class imbalance
- [ ] REST API with JSON endpoints

---

## Dataset Credit

> Farrukhjon Masudov, Taha Belkhouja, Yan Shi, Janardhan Rao Doppa, Haifeng Chen, Srikanth V. Krishnamurthy.  
> *BitcoinHeist: Topological Data Analysis for Ransomware Detection on the Bitcoin Blockchain.*  
> IJCAI 2020. [UCI Repository](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) - CC BY 4.0.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
