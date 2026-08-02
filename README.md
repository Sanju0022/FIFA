# ⚽ FIFA World Cup 2026 — Player & Squad Analytics

A full data science project on a 1,248-player, 48-squad FIFA World Cup 2026 dataset:
cleaning, exploratory analysis, unsupervised learning, association rule mining,
classification modeling with hyperparameter tuning, and a dynamic, sectioned
Streamlit dashboard.

**Live components**
- `notebook/FIFA_WC2026_Analysis.ipynb` — the full analysis, built to run top-to-bottom in Google Colab
- `app.py` — the Streamlit dashboard
- `data/` — the single cleaned dataset the deployed app reads (everything else — model metrics, ROC points, association rules — is computed live from it, cached)

---

## 1. Project structure

```
.
├── app.py                                  # Streamlit dashboard (13 sections)
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── fifa_wc2026_cleaned_enriched.csv    # the ONLY file the deployed app needs
└── notebook/
    └── FIFA_WC2026_Analysis.ipynb          # full offline analysis (raw import → cleaning → all artifacts)
```

**Single-file architecture.** Every chart, KPI, and ML comparison in `app.py` is
derived from the one CSV in `data/` — there is nothing else for a deployment to
go missing. Clustering, PCA, and all engineered KPI columns are precomputed and
stored as columns in that file. Classification (8 algorithms, class-weighting,
GridSearchCV tuning, threshold tuning, the overfitting diagnostic, the team-vs-
individual comparison) and association-rule mining are trained **live, in the
app, on first load** — cached with `st.cache_data` so it only runs once per
session (~30–40 seconds cold start, instant afterward). The notebook remains
the source of truth for the deeper, more exhaustively-tuned offline analysis
and also re-derives the raw import → cleaning steps from scratch.

---

## 2. Run the dashboard locally

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Dashboard sections (sidebar navigation — each section has its own **local** filters only)

| Section | Chart types used |
|---|---|
| 🏠 Overview | KPI cards, bar chart, sunburst |
| 👤 Player Demographics | bar chart, box plot, sunburst |
| 📈 Performance Analytics | line chart, bar chart, scatter plot |
| 📊 Advanced KPIs | bar chart, scatter plot, line chart, data table |
| 💰 Market Value Insights | bar chart, line chart, **Sankey diagram** |
| 🌍 Team & Squad Analysis | scatter plots, sunburst |
| 🏟️ Club & Group Insights | bar chart, **treemap** |
| 🔎 Player Explorer & Comparison | searchable data table, **radar chart** |
| 🔥 Correlations & Deep Dive | interactive heatmap, **treemap**, scatter with trendline |
| 🧩 Player Archetypes (Clustering) | scatter (PCA), grouped bar, stacked bar |
| 🔗 Association Rules | horizontal bar, data table |
| 🤖 Classification Model Results | grouped bar, **line chart (ROC)**, heatmap (confusion matrix), bar (feature importance) |
| 🎯 Model Improvements | bar (recall lift), grouped bar (threshold & team-effect comparisons), bar (feature importance, side by side), bar (overfitting diagnostic) |

**New in this version**: three additional sections built from columns that existed in the
data but weren't visualized before (`club_team`, `group_letter`, and a full
player-search/radar-comparison tool) — plus two new chart types (treemap, radar).

No donut charts are used anywhere, and there is no dashboard-wide/global filter — every
filter widget is scoped to the section it appears in.

---

## 3. Run the notebook (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/) → File → Upload notebook → select `notebook/FIFA_WC2026_Analysis.ipynb`
2. Upload `notebook/fifa_wc2026_raw.csv` to the Colab session (left sidebar → Files → upload), or mount Google Drive and update `RAW_PATH` in the first code cell
3. Runtime → Run all

The notebook:
1. Imports & inspects the raw data
2. Cleans it **without deleting a single row** (only 3 columns that were 100% empty are dropped; goalkeeper-only stats are filled with 0 because that missingness is structural, not an error)
3. Flags outliers via IQR (kept, not removed) and runs full descriptive/EDA analysis
4. Runs **K-Means clustering + PCA** to find player archetypes
5. Runs **Apriori association rule mining** on binned player traits
6. Trains **6 classifiers** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (GBRT), KNN, SVM — on an 80/20 stratified split
7. Reports Train/Test Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, and one combined ROC chart for all 6 models
8. Re-tunes all 6 with **GridSearchCV, 5-fold cross-validation**, and reprints the same comparison table + ROC chart
9. Saves every result table, chart, and the best model as artifacts for the dashboard

---

## 4. Advanced KPIs

Computed on **tournament** goals/assists (career totals only for Goals per Cap).
Ratios are left as `NaN` — not fabricated as 0 — when the denominator is 0 (e.g. a
player with 0 minutes has an undefined per-90 rate, not a zero rate).

| KPI | Formula |
|---|---|
| Goals per 90 | (Goals ÷ Minutes Played) × 90 |
| Assists per 90 | (Assists ÷ Minutes Played) × 90 |
| Goal Contribution | Goals + Assists |
| Goal Contribution per 90 | ((Goals + Assists) ÷ Minutes Played) × 90 |
| Minutes per Goal | Minutes Played ÷ Goals |
| Save Percentage | Saves ÷ (Saves + Goals Conceded) × 100 |
| Clean Sheet Percentage | Clean Sheets ÷ Matches Played × 100 |
| Start Percentage | Matches Started ÷ Matches Played × 100 |
| Market Value per Cap | Market Value ÷ (Caps + 1) |
| Goals per Cap | Career Goals ÷ Caps |
| Average Minutes per Match | Minutes Played ÷ Matches Played |

See the 📊 Advanced KPIs dashboard section and notebook §4 ("Advanced KPIs — quick look").

## 5. Modeling approach & why

**Target:** `is_high_value` — is a player in the top quartile (≥ €18M) of market value?
This was chosen over predicting `position` because it is a genuinely useful business
question (talent scouting / valuation) and, being binary, supports one clean combined
ROC/AUC comparison across all algorithms.

**Leakage control:** the feature set excludes `market_value_eur` and its direct
derivatives; the model only sees performance, physical, and squad-context features.

**Class balance:** ~26% positive class. This is reported transparently, `stratify=y`
is used in the split, and — because a naive model under this imbalance predicts
"not high value" too often and misses real high-value players — three fixes are
applied and each one's effect is measured separately, not just bundled into a final number:

1. **Class-weighting** (`class_weight='balanced'` / `scale_pos_weight` / manual
   `sample_weight` for sklearn's GBRT, which has no native `class_weight` param) —
   measured alone, before any other tuning (see the 🎯 Model Improvements section,
   "Fix 1")
2. **GridSearchCV, 5-fold CV, scored on F1** (not accuracy/ROC-AUC) with the
   weighting option itself included as a tunable hyperparameter — 8 algorithms:
   Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (GBRT),
   KNN, SVM, **XGBoost**, **LightGBM**
3. **Decision-threshold tuning** — the default 0.5 cutoff is rarely optimal under
   imbalance. The F1-optimal threshold is found via `cross_val_predict` on the
   **training set only** (5-fold), then applied once to the untouched test set

**Result:** recall on the best model improved substantially over the unweighted
baseline, while ROC-AUC held steady — see the live numbers in the 🎯 Model
Improvements dashboard section (they're computed fresh from the single CSV on
each app deployment, so exact figures can shift slightly run to run with the
smaller, faster hyperparameter grid used in the app vs. the notebook's fuller grid).

**Overfitting check.** An earlier tuning pass let Random Forest, XGBoost, LightGBM and
KNN reach ~100% *resubstitution* train accuracy (evaluated on the exact rows they were
fit on) — a red flag on ~1,000 rows. Two things were done: (1) regularization params
(`min_samples_leaf`, `max_features`, `subsample`, `reg_lambda`, `min_child_samples`,
etc.) were added to the grids, and (2) train performance is reported as **out-of-fold
(OOF) accuracy** via cross-validation instead of resubstitution accuracy — because for
KNN with distance-weighting in particular, resubstitution accuracy is a known artifact
(a training point's nearest neighbor during self-prediction is itself, at distance 0).
The honest OOF gaps come back near zero or negative for every model — the "100%
train accuracy" was inflated by memorization capacity, not genuine overfitting. See
the 🎯 Model Improvements dashboard section.

**A separate, explicit test isolates the "playing for a strong team" effect**: the
winning model is retrained live on a feature set with `elo_rating` and
`fifa_ranking_pre_tournament` removed. Team context turns out to be a real and
substantial driver of predicted market value, but individual stats alone still
retain meaningful signal. See the 🎯 Model Improvements dashboard section.

---

## 6. Key insights (see notebook §16 for full writeup)

- Zero rows were removed during cleaning; only 3 fully-empty columns were dropped.
- Ensemble/boosted tree models (Random Forest, Gradient Boosting, XGBoost, LightGBM)
  beat linear/distance models on both ROC-AUC and F1 — market value is driven by
  **non-linear interactions** between age, minutes played, and career output, not a
  simple linear combination.
- **Class imbalance was fixed, not just reported**: class-weighting alone lifted
  recall the most for SVM (+0.35) and Logistic Regression (+0.19); F1-scored
  GridSearchCV and threshold tuning pushed the best model's recall from ~0.68–0.71
  up to **0.815**, with ROC-AUC essentially unchanged (0.94).
- **Team context matters a lot, but isn't everything**: removing Elo rating and FIFA
  ranking from the feature set drops ROC-AUC from 0.943 to 0.759 and recall from 0.815
  to 0.354 — squad strength is the single strongest signal, but individual output
  still carries real, usable predictive power on its own.
- Association rules show **high squad Elo rating and high individual market value
  co-occur** with lift > 1.5 — the same pattern the classification comparison confirms
  from a different angle.
- K-Means finds 4 player archetypes, separated mainly by market value, minutes played,
  and goal involvement rather than discipline stats.

---

## 7. Pushing this repo to GitHub

This folder is ready to push as-is. From inside this folder:

```bash
git init
git add .
git commit -m "Initial commit: FIFA WC2026 analytics — cleaning, ML notebook, Streamlit dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

(Create the empty repo on GitHub first via github.com/new, or with `gh repo create` if
you have the GitHub CLI installed and authenticated.)

---

## 8. Data source

Player and squad data snapshot from sofascore.com, last verified 19-07-2026 (as recorded
in the `data_source` / `last_verified` columns of the raw dataset).
