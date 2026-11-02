# IT Support Ticket Priority Prediction Engine

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to automate IT support ticket triage, classify ticket priorities (`LOW`, `MEDIUM`, `HIGH`, `URGENT`), and route critical incidents in real-time through an interactive Streamlit executive dashboard.

---

## System Architecture

[ Unstructured Support Ticket ] + [ Metadata Parameters ]
│
▼
[ Regex & NLTK Text Cleaning ]
│
▼
[ TF-IDF Vectorizer + One-Hot Encoder ]
│
▼
[ Stacked Sparse Feature Matrix Construction ]
│
▼
[ Balanced Subsample Random Forest Classifier ]
│
▼
[ Domain Business Rules / Heuristic Layer ]
│
▼
[ Executive Streamlit Dashboard & Analytics ]

---

## Project Structure

it-ticket-priority-predictor/
├── datasets/
│   ├── synthetic_it_support_tickets.csv    # Raw ticket ingestion data
│   └── cleaned_tickets.csv                 # Preprocessed dataset
├── models/
│   ├── tfidf_vectorizer.pkl                # Fitted TF-IDF vectorizer (1,500 features)
│   ├── category_encoder.pkl                # Fitted OneHotEncoder for metadata
│   └── priority_classifier_model.pkl       # Trained Random Forest Classifier
├── notebooks/
│   ├── 01_data_ingestion.ipynb             # Ingestion & initial exploratory analysis
│   ├── 02_data_preprocessing.ipynb         # Missing value handling & schema validation
│   ├── 03_text_cleaning_nlp.ipynb          # NLTK lemmatization, regex noise removal
│   ├── 04_feature_engineering.ipynb        # Sublinear TF-IDF & One-Hot Encoding
│   ├── 05_model_training.ipynb             # Random Forest training with balanced subsampling
│   └── 06_model_evaluation.ipynb           # Model evaluation metrics & confusion matrix
├── dashboard/
│   └── app.py                              # Streamlit executive dashboard application
├── sample_test_tickets.csv                 # Sample dataset for bulk prediction testing
├── requirements.txt                        # Required Python packages
└── README.md                               # Project documentation

---

## Key Features

1. **Hybrid NLP & Metadata Pipeline:** Combines ticket narratives (unstructured text) with categorical metadata (`product_area`, `issue_type`, `customer_segment`, `channel`, `platform`).
2. **Balanced Minority Class Detection:** Employs `class_weight='balanced_subsample'` inside Random Forest decision trees to handle severe class imbalance (`URGENT` tickets comprise ~5% of raw data).
3. **Domain Heuristic Layer:** Adjusts raw model probability outputs with business rules to reliably flag critical security breaches, database outages, and system crashes as `URGENT` or `HIGH`.
4. **Interactive Analytics:** Features Plotly charts for distribution exploration across priorities, product areas, channels, and issue categories.
5. **Single & Batch Prediction:** Supports real-time individual ticket triage alongside CSV batch file uploads with downloadable results.

---

## Tech Stack

* **Language:** Python 3.9+
* **Data Manipulation:** Pandas, NumPy
* **NLP & Vectorization:** NLTK, Scikit-Learn (`TfidfVectorizer`)
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `OneHotEncoder`)
* **Visualization:** Plotly Express, Matplotlib
* **Dashboard:** Streamlit
* **Model Serialization:** Joblib

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment
```powershell
# Clone repository
git clone [https://github.com/your-username/it-ticket-priority-predictor.git](https://github.com/your-username/it-ticket-priority-predictor.git)
cd it-ticket-priority-predictor

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate


Install Required Dependencies
Create a requirements.txt file with the following content:

pandas
numpy
scikit-learn
nltk
joblib
streamlit
plotly
scipy

Install packages via pip:

pip install -r requirements.txt


Execution Guide
Step 1: Run Machine Learning Pipelines (Notebooks 01–06)
Execute the notebooks sequentially to build, clean, engineer features, train, and save all pipeline artifacts:

PowerShell
# Execute feature engineering & model training scripts
jupyter notebook
Run 01_data_ingestion.ipynb — Load raw dataset.

Run 02_data_preprocessing.ipynb — Clean schema and missing values.

Run 03_text_cleaning_nlp.ipynb — Apply regex noise removal and NLTK lemmatization.

Run 04_feature_engineering.ipynb — Fit and export tfidf_vectorizer.pkl and category_encoder.pkl.

Run 05_model_training.ipynb — Fit and export priority_classifier_model.pkl.

Run 06_model_evaluation.ipynb — Evaluate precision, recall, and classification metrics.



Step 2: Launch Executive Dashboard
Run the Streamlit application from the project root directory:

PowerShell
python -m streamlit run dashboard/app.py
Open your browser and navigate to http://localhost:8501.


Dashboard Viewports & Features
Dataset Overview: Key KPI metrics (Total Tickets, Product Areas, Issue Categories) and raw record inspection.

Charts & Interactive Analytics: Plotly horizontal/vertical bar charts and donut pie charts for distribution analytics with hover tooltips and interactive zooming.

Model Configuration: Technical specifications detailing estimator parameters, vectorizer n-gram bounds, and feature stacking architectures.

Single Ticket Prediction: Form inputs for real-time priority classification, confidence scoring, and class probability distribution tables.

Bulk Prediction: Batch file processor supporting CSV uploads and classified result downloads.

About Pipeline: Architectural walkthrough of processing stages from text cleaning to business heuristic adjustments.