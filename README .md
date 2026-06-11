# 🎬 IMDb Movie Rating Prediction Project

This project delivers an end-to-end Machine Learning pipeline designed to clean, process, and predict the average IMDb rating (`averageRating`) of movies. The project implements and evaluates two primary models: **Random Forest Regressor** and **Elastic Net Regression**, while conducting a deep dive into Fairness Analysis, Feature Importance, and Error Analysis.

---

## 📌 Key Features & Project Pipeline

- **Rigorous Data Cleaning:** Handles missing values, enforces proper data types, and resolves duplicates (leaving a clean dataset of ~115,560 rows for training).
- **Advanced Feature Engineering:**
  - Computes a 10-year rolling weighted reputation for actors and directors (`weighted_actor_reputation_10yrs`, `weighted_director_reputation_10yrs`) extracted from a core crew lookup reference table (`crew_lookup_df`).
  - Implements categorical encoding (One-Hot Encoding) for movie genres.
  - Formulates language features (classifying movies into English vs. Foreign).
- **Strict Evaluation Protocol:** Utilizes 10-Fold Cross-Validation during validation to prevent overfitting, alongside a final evaluation on a held-out static test set (20%).
- **Fairness & Slice Analysis:** Examines model performance across sub-populations (by top genres and language categories) to detect potential algorithmic bias.

---

## 📊 Model Performance (Final Static Test Set)

Following model training and testing, the **Random Forest** regressor achieved the following official evaluation metrics:

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **R² Score** | **0.3170** | The model explains approximately 31.7% of the variance in movie ratings. |
| **RMSE** | **1.0634** | The standard deviation of the residuals (prediction errors). |
| **MAE** | **0.8022** | On average, the model's predictions miss the true rating by just ~0.8 points. |

### 🔍 Feature Importance & Coefficients
Feature impact analysis revealed the primary drivers behind the model's predictions:
1. **Weighted Actor Reputation (10 Yrs):** Ranked as the top predictor (~36.8% importance in Random Forest).
2. **Weighted Director Reputation (10 Yrs):** Heavily influenced predictions (~19.6% importance in Random Forest).
3. **Documentary Genre:** Exhibited a strong positive linear relationship (Standardized Coefficient of +1.25 in Elastic Net).
4. **Horror Genre:** Showed a statistically significant negative correlation with movie ratings across both models.

---

## ⚖️ Fairness & Bias Analysis

To ensure algorithmic accountability, fairness was evaluated across distinct data slices:
- **Language Bias:** The model is notably more accurate when predicting ratings for **English** movies ($MAE = 0.69$) compared to **Foreign** language films ($MAE = 0.83$).
- **Genre Bias:** The model demonstrates outstanding predictive capabilities on *Documentaries* and *Romance* films ($MAE \approx 0.73$), while *Action* films present a higher learning challenge ($MAE = 0.86$).

---

## 📦 Saved Artifacts

To support seamless production deployment or evaluation on new data (such as `NEW.csv`), the pipeline serializes state objects using `joblib`:

1. `random_forest.pkl`: A dictionary architecture housing the trained Random Forest estimator and the final categorical mapping parameters (`final_trained_mappings`).
2. `elastic_net.pkl`: A serialized bundle containing the trained Elastic Net estimator, mapping dictionary, and the `StandardScaler` state utilized for continuous feature normalization.
3. `best_movie_rating_model.pkl`: The top-performing model artifact selected for immediate evaluation workflows.

---

## 🛠️ Prerequisites & Usage

### Dependencies
Ensure the following Python packages are installed before execution:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib