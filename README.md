# Mechine_Learning_IMDB
# IMDb Movie Rating Prediction Pipeline

An end-to-end data science pipeline built to predict movie ratings (`averageRating`) using advanced feature engineering, target encoding, and machine learning models (Elastic Net and Random Forest). This pipeline is designed with a strict zero-data-leakage architecture, making it ready to be evaluated on completely unseen test data via serialized artifacts.

## 📌 Project Overview
Predicting movie ratings on IMDb is a challenging regression task due to high variance and complex feature interactions. This project establishes a robust workflow that handles historical reputation data (cast and directors), standardizes film attributes, and trains optimized models. 

The **Random Forest Regressor** emerged as the champion model, outperforming the baseline linear approach by capturing complex, non-linear feature interactions without introducing chronological data leakage.

---

## 🛠️ Pipeline Architecture & Components

The framework is cleanly modularized into two distinct stages to guarantee strict validation sanitization:

### 1. Data Cleaning (`clean_data`)
Performs structural integrity checks and baseline filters on the raw datasets:
* Drops records missing primary identifiers (`tconst`), release metrics (`startYear`), or target labels (`averageRating`).
* Dynamically type-casts relational columns to robust numeric types.
* Restricts analysis to feature-length cinematic content (runtimes between 60 and 300 minutes), while preserving missing fields (`NaN`) for downstream mathematical imputation.
* Drops potential target-leakage markers (`BoxOffice`, `budget`, `numVotes`).

### 2. Isolated Processing & Feature Engineering (`prepare_data`)
This function handles all data transformations and acts as the gatekeeper against **Data Leakage**:
* **Train Mode (`is_train=True`)**: Learns statistical parameters (medians, categorical frequencies, and historical score dictionaries) from the training slice and populates a stateful `mappings` dictionary.
* **Test Mode (`is_train=False`)**: Evaluates incoming unseen rows using *only* the fixed historical mappings extracted during training.

---

## 📊 Feature Engineering Highlights

| Feature Name | Type | Description / Implementation | Engineering Design Choice |
| :--- | :--- | :--- | :--- |
| `weighted_actor_reputation_10yrs` | Continuous | Historical target encoding of up to 4 top cast members over a rolling 10-year window prior to the film's release. Evaluated via an inverse rank-weighted average: $Weights = [N, N-1, ..., 1]$. | **Lead Actor Precedence:** A simple mean degrades the signal. Linear sequence weighting emphasizes the lead actor, who historically exerts the largest impact on audience ratings. |
| `weighted_director_reputation_10yrs` | Continuous | Average rating of the director's catalog within a tight 10-year lookback window before the target release year. Uses a trained `GLOBAL_DEFAULT` if the director is unrepresented in the historical matrix. | Handles the "Cold Start" problem elegantly for debut directors without introducing null propagation or looking ahead into test records. |
| `runtime_short` / `medium` / `long` | Binary | Bins film durations into explicit categories based on empirical training medians:<br>- **Short**: 60 - 90 mins<br>- **Medium**: 90 - 140 mins<br>- **Long**: 140 - 300 mins | Discretizing continuous runtime eases the classification splitting logic for non-linear tree partitions. |
| `genre_[11_top_genres]` | Binary | One-Hot Encoding of the top 11 most frequent genres in the training distribution. | Optimized via experimental hyperparameter tuning; tracking more than 11 genres introduced sparse cardinality noise. |
| `genre_other` / `genre_missing` | Binary | Separates residual sparse genres (`other`) from entirely unclassified metadata rows (`missing`). | Prevents structural confounding between unpopular genres and unindexed records. |
| `is_english_language` | Binary | Flag indicating if English is flagged in the native language metadata fields. | Replaces geographic metrics (`Country`), which showed higher validation loss during feature selection runs. |
| `is_language_unknown` | Binary | Explicit indicator capturing missing, null, or unknown raw language entries. | Prevents misclassifying blank fields as foreign cinema, preserving data reliability patterns. |

---

## 📈 Experimental Results & Performance

Evaluation metrics captured over the completely isolated test split:

| Model | MAE | RMSE | $R^2$ (Variance Explained) |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **0.690** | **0.932** | **28.5%** |
| **Elastic Net (Linear Baseline)** | 0.741 | 0.985 | 20.1% |

### Core Data Science Insights:
1. **Non-Linear Dominance:** The Random Forest significantly outperforms Elastic Net. This confirms that movie ratings rely heavily on feature intersections (e.g., a specific genre *combined* with a high-reputation lead actor) rather than isolated linear slopes.
2. **The Outlier Effect:** Across both models, RMSE remains consistently higher than MAE. This behavior reflects the inherent volatility (High Variance) of genres like **Action** and **Comedy**, which contain extreme outliers (critically panned direct-to-streaming releases vs. massive box-office blockbusters) that penalize squared-error metrics.

---

## 💾 Model Serialization & Delivery

To support seamless validation on an independent external test set, the complete state can be exported into a single pickle artifact. This file preserves both the trained estimator weights and the strict historical feature encodings.

### Exporting the Pipeline
Run the following block inside your notebook environment to write the localized binaries:
```python
import joblib

# Bundle the model weights and data-leakage guards together
artifacts_to_save = {
    'model': rf_model,
    'mappings': trained_mappings
}

# Serialize physically to disk
joblib.dump(artifacts_to_save, 'best_movie_rating_model.pkl')
print("🥇 Artifact successfully exported to best_movie_rating_model.pkl")
