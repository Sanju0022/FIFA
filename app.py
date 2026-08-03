"""
FIFA World Cup 2026 — Player & Squad Analytics Dashboard
Sectioned, local-filters-only Streamlit dashboard built on top of the
cleaned dataset and precomputed ML artifacts from notebook/FIFA_WC2026_Analysis.ipynb
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------------------------------------------------
# PAGE CONFIG & THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FIFA World Cup 2026 — Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PITCH_GREEN = "#0B6E4F"
DEEP_NAVY = "#0D1B2A"
GOLD = "#D4AF37"
CHALK = "#F5F3EE"
CORAL = "#E86A5C"
SKY = "#3B82F6"

PALETTE = [PITCH_GREEN, GOLD, CORAL, SKY, "#8E44AD", "#16A085", "#E67E22", "#2C3E50"]

st.markdown(f"""
<style>
    .stApp {{ background-color: {CHALK}; }}
    section[data-testid="stSidebar"] {{ background-color: {DEEP_NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: {CHALK} !important; }}
    h1, h2, h3 {{ color: {DEEP_NAVY}; font-family: 'Georgia', serif; }}
    .kpi-card {{
        background: white; border-left: 6px solid {PITCH_GREEN};
        border-radius: 8px; padding: 14px 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    div[data-testid="stMetric"] {{
        background: white; border-radius: 10px; padding: 12px 14px;
        border-left: 6px solid {GOLD}; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .section-note {{ color: #555; font-size: 0.92rem; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Arial", color=DEEP_NAVY),
    margin=dict(l=10, r=10, t=60, b=10),
)

# ----------------------------------------------------------------------------
# DATA LOADING — single source file. Every graph, KPI, and ML comparison below
# is derived from this one CSV; nothing else needs to exist on disk.
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_FILENAME = "fifa_wc2026_cleaned_enriched.csv"

def find_data_file():
    """Check the expected location first, then fall back to a few common
    mistakes (file placed at repo root, or anywhere else in the repo)."""
    candidates = [
        APP_DIR / "data" / DATA_FILENAME,   # expected location
        APP_DIR / DATA_FILENAME,            # accidentally placed next to app.py instead of in data/
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last resort: search the whole repo tree for a file with this name
    matches = list(APP_DIR.rglob(DATA_FILENAME))
    if matches:
        return matches[0]
    return None

DATA_FILE = find_data_file()

if DATA_FILE is None:
    try:
        repo_tree = "\n".join(f"- `{p.relative_to(APP_DIR)}`" for p in sorted(APP_DIR.rglob("*"))
                               if p.is_file() and ".git" not in p.parts)
    except Exception:
        repo_tree = "(could not list repo contents)"
    st.error(
        f"**Data file not found.** This app looks for `{DATA_FILENAME}` in `data/`, "
        f"at the repo root, or anywhere else in the repo — and couldn't find it in any of those places.\n\n"
        f"**Files actually present in this deployment:**\n{repo_tree}\n\n"
        "Compare that list to what you expect. Most often this means the CSV either wasn't pushed to GitHub "
        "at all, or was pushed with a different filename/casing than "
        f"`{DATA_FILENAME}`. After fixing it on GitHub, reboot the app from Streamlit Cloud's "
        "*Manage app* menu (a plain git push doesn't always trigger a rebuild)."
    )
    st.stop()

@st.cache_data
def load_players():
    df = pd.read_csv(DATA_FILE)
    df["experience_tier"] = df["experience_tier"].astype(str)
    df["age_group"] = df["age_group"].astype(str)
    return df

df = load_players()

# ----------------------------------------------------------------------------
# LIVE ML PIPELINE — everything the "Classification Model Results", "Model
# Improvements", and "Association Rules" sections need is computed here, once,
# from the single dataframe above, and cached for the rest of the session.
# This replaces what used to be ~14 separate precomputed CSV artifacts.
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Training classification models (first load only, cached after)...")
def run_ml_pipeline(_df, include_fifa_rank=False):
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.base import clone
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                  roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from mlxtend.frequent_patterns import apriori, association_rules as mlx_rules
    from mlxtend.preprocessing import TransactionEncoder

    RANDOM_STATE = 42
    d = _df.copy()

    feature_cols = ["caps","height_cm","career_goals","matches_played","matches_started","minutes_played",
                     "tournament_goals","assists","yellow_cards","red_cards","penalty_goals","own_goals",
                     "age_years","start_rate","minutes_per_match","goal_involvement","discipline_index",
                     "clean_sheets","saves","goals_conceded"]
    team_cols = ["elo_rating"] + (["fifa_ranking_pre_tournament"] if include_fifa_rank else [])
    # By default fifa_ranking_pre_tournament is excluded from the model's feature set —
    # elo_rating is the only team-context feature. Classification Model Results has a
    # local toggle that reruns this pipeline with include_fifa_rank=True if you want to
    # add it back in and see every metric/importance recalculated with it included.
    cat_cols = ["position"]

    def build_xy(include_team):
        cols = feature_cols + (team_cols if include_team else [])
        X = d[cols + cat_cols].copy()
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        y = d["is_high_value"]
        return X, y

    X_full, y = build_xy(True)
    X_indiv, _ = build_xy(False)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train_indiv, X_test_indiv = X_indiv.loc[X_train.index], X_indiv.loc[X_test.index]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    scaler_i = StandardScaler()
    X_train_i_s = scaler_i.fit_transform(X_train_indiv); X_test_i_s = scaler_i.transform(X_test_indiv)
    uses_scaled = {"Logistic Regression", "KNN", "SVM (RBF)"}
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    def make_models(weighted):
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced" if weighted else None),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced" if weighted else None),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, max_features="sqrt", n_estimators=200, max_depth=8, class_weight="balanced" if weighted else None),
            "Gradient Boosting (GBRT)": GradientBoostingClassifier(random_state=RANDOM_STATE, subsample=0.9, n_estimators=150, max_depth=3),
            "KNN": KNeighborsClassifier(n_neighbors=9),
            "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced" if weighted else None),
            "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_estimators=150, max_depth=4, scale_pos_weight=spw if weighted else 1),
            "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_estimators=150, class_weight="balanced" if weighted else None),
        }

    def evaluate(name, model, use_scaled, weighted):
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test
        if name == "Gradient Boosting (GBRT)" and weighted:
            model.fit(Xtr, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            model.fit(Xtr, y_train)
        tr_pred, te_pred = model.predict(Xtr), model.predict(Xte)
        te_proba = model.predict_proba(Xte)[:, 1]
        return {"Model": name, "_model": model, "_proba": te_proba, "_pred": te_pred,
                "Train Accuracy": accuracy_score(y_train, tr_pred), "Test Accuracy": accuracy_score(y_test, te_pred),
                "Precision": precision_score(y_test, te_pred), "Recall": recall_score(y_test, te_pred),
                "F1-Score": f1_score(y_test, te_pred), "ROC-AUC": roc_auc_score(y_test, te_proba)}

    baseline_res = [evaluate(n, m, n in uses_scaled, False) for n, m in make_models(False).items()]
    weighted_res = [evaluate(n, m, n in uses_scaled, True) for n, m in make_models(True).items()]
    baseline_table = pd.DataFrame(baseline_res).drop(columns=["_model", "_proba", "_pred"])
    weighted_table = pd.DataFrame(weighted_res).drop(columns=["_model", "_proba", "_pred"])
    imbalance_comp = pd.concat([
        baseline_table.assign(Weighting="Unweighted"), weighted_table.assign(Weighting="Balanced")
    ]).sort_values(["Model", "Weighting"]).reset_index(drop=True)
    recall_lift = (weighted_table.set_index("Model")["Recall"] - baseline_table.set_index("Model")["Recall"]) \
        .reset_index().rename(columns={"Recall": "Recall_Lift"})

    def roc_cm_frames(results, label):
        roc_rows, cm_rows = [], []
        for r in results:
            fpr, tpr, _ = roc_curve(y_test, r["_proba"])
            for f, t in zip(fpr, tpr):
                roc_rows.append({"dataset": label, "Model": r["Model"], "fpr": f, "tpr": t, "AUC": r["ROC-AUC"]})
            tn, fp, fn, tp = confusion_matrix(y_test, r["_pred"]).ravel()
            cm_rows.append({"dataset": label, "Model": r["Model"], "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        return pd.DataFrame(roc_rows), pd.DataFrame(cm_rows)

    roc_baseline, cm_baseline = roc_cm_frames(baseline_res, "Baseline")

    # Small, fast GridSearchCV per model (few combos so this stays quick on a hosted app)
    param_grids = {
        "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), {"C": [0.1, 1, 10], "class_weight": [None, "balanced"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), {"max_depth": [5, 8], "min_samples_leaf": [1, 10], "class_weight": [None, "balanced"]}),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, max_features="sqrt"), {"n_estimators": [200], "max_depth": [6, 10], "class_weight": [None, "balanced"]}),
        "Gradient Boosting (GBRT)": (GradientBoostingClassifier(random_state=RANDOM_STATE, subsample=0.9), {"n_estimators": [100, 150], "max_depth": [2, 3]}),
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 9, 15]}),
        "SVM (RBF)": (SVC(probability=True, random_state=RANDOM_STATE), {"C": [1, 10], "class_weight": [None, "balanced"]}),
        "XGBoost": (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_estimators=150), {"max_depth": [3, 5], "scale_pos_weight": [1, spw]}),
        "LightGBM": (LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_estimators=150), {"max_depth": [3, 5], "class_weight": [None, "balanced"]}),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned_res, best_estimators = [], {}
    gbrt_w = compute_sample_weight("balanced", y_train)
    for name, (est, grid) in param_grids.items():
        use_scaled = name in uses_scaled
        Xtr, Xte = (X_train_s, X_test_s) if use_scaled else (X_train, X_test)
        gs = GridSearchCV(est, grid, scoring="f1", cv=cv, n_jobs=1)
        if name == "Gradient Boosting (GBRT)":
            gs.fit(Xtr, y_train, sample_weight=gbrt_w)
        else:
            gs.fit(Xtr, y_train)
        best_estimators[name] = gs.best_estimator_
        resub_acc = accuracy_score(y_train, gs.predict(Xtr))
        oof_est = clone(gs.best_estimator_)
        oof_kwargs = {"params": {"sample_weight": gbrt_w}} if name == "Gradient Boosting (GBRT)" else {}
        oof_proba = cross_val_predict(oof_est, Xtr, y_train, cv=cv, method="predict_proba", n_jobs=1, **oof_kwargs)[:, 1]
        oof_acc = accuracy_score(y_train, (oof_proba >= 0.5).astype(int))
        te_pred, te_proba = gs.predict(Xte), gs.predict_proba(Xte)[:, 1]
        te_acc = accuracy_score(y_test, te_pred)
        tuned_res.append({"Model": name, "_proba": te_proba, "_pred": te_pred,
                           "Resubstitution Train Acc": resub_acc, "OOF Train Acc (honest)": oof_acc,
                           "Test Accuracy": te_acc, "Resubstitution Gap": resub_acc - te_acc, "OOF Gap (honest)": oof_acc - te_acc,
                           "Precision": precision_score(y_test, te_pred), "Recall": recall_score(y_test, te_pred),
                           "F1-Score": f1_score(y_test, te_pred), "ROC-AUC": roc_auc_score(y_test, te_proba)})
    tuned_table = pd.DataFrame(tuned_res).drop(columns=["_proba", "_pred"]).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    overfit_diag = tuned_table[["Model","Resubstitution Train Acc","OOF Train Acc (honest)","Test Accuracy","Resubstitution Gap","OOF Gap (honest)"]].round(4)
    roc_tuned, cm_tuned = roc_cm_frames(tuned_res, "Tuned")

    best_name = tuned_table.iloc[0]["Model"]
    best_model = best_estimators[best_name]
    use_scaled_best = best_name in uses_scaled
    Xtr_b, Xte_b = (X_train_s, X_test_s) if use_scaled_best else (X_train, X_test)

    cv_proba = cross_val_predict(best_model, Xtr_b, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    prec_a, rec_a, thr_a = precision_recall_curve(y_train, cv_proba)
    f1_a = 2 * prec_a[:-1] * rec_a[:-1] / (prec_a[:-1] + rec_a[:-1] + 1e-12)
    best_thresh = thr_a[np.argmax(f1_a)]
    te_proba_best = best_model.predict_proba(Xte_b)[:, 1]
    pred_default = (te_proba_best >= 0.5).astype(int)
    pred_tuned = (te_proba_best >= best_thresh).astype(int)
    threshold_comp = pd.DataFrame([
        {"Threshold": "Default (0.50)", "Precision": precision_score(y_test, pred_default), "Recall": recall_score(y_test, pred_default),
         "F1-Score": f1_score(y_test, pred_default), "Accuracy": accuracy_score(y_test, pred_default)},
        {"Threshold": f"F1-optimal ({best_thresh:.2f})", "Precision": precision_score(y_test, pred_tuned), "Recall": recall_score(y_test, pred_tuned),
         "F1-Score": f1_score(y_test, pred_tuned), "Accuracy": accuracy_score(y_test, pred_tuned)},
    ])

    model_indiv = clone(best_model)
    Xtr_i, Xte_i = (X_train_i_s, X_test_i_s) if use_scaled_best else (X_train_indiv, X_test_indiv)
    model_indiv.fit(Xtr_i, y_train)
    proba_indiv, pred_indiv = model_indiv.predict_proba(Xte_i)[:, 1], model_indiv.predict(Xte_i)
    pred_full = best_model.predict(Xte_b)
    team_vs_indiv = pd.DataFrame([
        {"Feature Set": "Full (with team context)", "Test Accuracy": accuracy_score(y_test, pred_full), "Precision": precision_score(y_test, pred_full),
         "Recall": recall_score(y_test, pred_full), "F1-Score": f1_score(y_test, pred_full), "ROC-AUC": roc_auc_score(y_test, te_proba_best)},
        {"Feature Set": "Individual Skill Only (no team context)", "Test Accuracy": accuracy_score(y_test, pred_indiv), "Precision": precision_score(y_test, pred_indiv),
         "Recall": recall_score(y_test, pred_indiv), "F1-Score": f1_score(y_test, pred_indiv), "ROC-AUC": roc_auc_score(y_test, proba_indiv)},
    ])

    def get_importances(model, columns, top_n=15):
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False).head(top_n)
        if hasattr(model, "coef_"):
            return pd.Series(np.abs(model.coef_[0]), index=columns).sort_values(ascending=False).head(top_n)
        return pd.Series(dtype=float)

    feat_imp = get_importances(best_model, X_train.columns).reset_index()
    feat_imp.columns = ["feature", "importance"]
    feat_imp_indiv = get_importances(model_indiv, X_train_indiv.columns).reset_index()
    feat_imp_indiv.columns = ["feature", "importance"]

    # Association rules (fast, no model training needed)
    def flag(col, label):
        return np.where(d[col] >= d[col].median(), f"High_{label}", f"Low_{label}")
    basket = pd.DataFrame({
        "MarketValue": flag("market_value_eur", "MarketValue"), "Caps": flag("caps", "Caps"),
        "Minutes": flag("minutes_played", "Minutes"), "GoalInvolvement": flag("goal_involvement", "GoalInvolvement"),
        "Discipline": flag("discipline_index", "Discipline"), "EloRating": flag("elo_rating", "EloRating"),
        "Position": d["position"],
    })
    te = TransactionEncoder()
    te_ary = te.fit(basket.astype(str).values.tolist()).transform(basket.astype(str).values.tolist())
    te_df = pd.DataFrame(te_ary, columns=te.columns_)
    freq = apriori(te_df, min_support=0.10, use_colnames=True)
    r = mlx_rules(freq, metric="lift", min_threshold=1.3)
    r = r[(r["antecedents"].apply(len) <= 2) & (r["consequents"].apply(len) == 1) & (r["confidence"] >= 0.5)]
    r["antecedents"] = r["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    r["consequents"] = r["consequents"].apply(lambda s: ", ".join(sorted(s)))
    r = r.drop_duplicates(subset=["antecedents", "consequents"]).sort_values("lift", ascending=False).reset_index(drop=True)
    rules = r[["antecedents", "consequents", "support", "confidence", "lift"]]

    return dict(model_baseline=baseline_table, model_tuned=tuned_table, roc_baseline=roc_baseline, roc_tuned=roc_tuned,
                cm_baseline=cm_baseline, cm_tuned=cm_tuned, feat_imp=feat_imp, feat_imp_indiv=feat_imp_indiv,
                rules=rules, recall_lift=recall_lift, threshold_comp=threshold_comp, team_vs_indiv=team_vs_indiv,
                imbalance_comp=imbalance_comp, overfit_diag=overfit_diag)

_ml = run_ml_pipeline(df)
model_baseline = _ml["model_baseline"]
model_tuned = _ml["model_tuned"]
roc_baseline = _ml["roc_baseline"]
roc_tuned = _ml["roc_tuned"]
cm_baseline = _ml["cm_baseline"]
cm_tuned = _ml["cm_tuned"]
feat_imp = _ml["feat_imp"]
feat_imp_indiv = _ml["feat_imp_indiv"]
rules = _ml["rules"]
recall_lift = _ml["recall_lift"]
threshold_comp = _ml["threshold_comp"]
team_vs_indiv = _ml["team_vs_indiv"]
imbalance_comp = _ml["imbalance_comp"]
overfit_diag = _ml["overfit_diag"]

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (this is page navigation, not a data filter)
# ----------------------------------------------------------------------------
st.sidebar.markdown("## ⚽ FIFA World Cup 2026")
st.sidebar.markdown("### Player & Squad Analytics")
st.sidebar.caption("1,248 players · 48 national squads")
section = st.sidebar.radio(
    "Go to section",
    ["🏠 Overview", "👤 Player Demographics", "📈 Performance Analytics", "📊 Advanced KPIs",
     "💰 Market Value Insights", "🌍 Team & Squad Analysis", "🏟️ Club & Group Insights",
     "🔎 Player Explorer & Comparison", "🔥 Correlations & Deep Dive",
     "🧩 Player Archetypes (Clustering)", "🔗 Association Rules",
     "🤖 Classification Model Results", "🎯 Model Improvements"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Each section below has its own **local** filters — nothing here affects other sections.")
st.sidebar.markdown("---")
st.sidebar.caption("Data: sofascore.com snapshot (19-07-2026) · Cleaned & modeled in the companion notebook.")

# ============================================================================
# SECTION 1 — OVERVIEW
# ============================================================================
if section == "🏠 Overview":
    st.title("FIFA World Cup 2026 — Player & Squad Analytics")
    st.markdown('<p class="section-note">A dynamic, section-based dashboard covering demographics, performance, '
                'market value, squad strength, unsupervised player archetypes, association rules, and '
                'classification model benchmarking.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Players", f"{len(df):,}")
    c2.metric("Squads", df["team_name"].nunique())
    c3.metric("Avg. Market Value", f"€{df['market_value_eur'].mean()/1e6:.1f}M")
    c4.metric("Avg. Age", f"{df['age_years'].mean():.1f} yrs")
    c5.metric("Total Squad Value", f"€{df['market_value_eur'].sum()/1e9:.2f}B")

    st.markdown("### Squad Market Value — Top 15")
    top_n = st.slider("Number of squads to show", 5, 30, 15, key="ov_topn")
    team_val = (df.groupby("team_name")["market_value_eur"].sum().sort_values(ascending=False).head(top_n) / 1e6).reset_index()
    fig = px.bar(team_val, x="market_value_eur", y="team_name", orientation="h",
                 labels={"market_value_eur": "Total Squad Market Value (€M)", "team_name": "Squad"},
                 color="market_value_eur", color_continuous_scale=[CHALK, PITCH_GREEN])
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Squad Composition — Confederation → Position")
    sb = df.groupby(["confederation", "position"]).size().reset_index(name="players")
    fig2 = px.sunburst(sb, path=["confederation", "position"], values="players",
                        color="confederation", color_discrete_sequence=PALETTE)
    fig2.update_layout(**PLOTLY_LAYOUT, height=520)
    st.plotly_chart(fig2, width='stretch')

# ============================================================================
# SECTION 2 — PLAYER DEMOGRAPHICS
# ============================================================================
elif section == "👤 Player Demographics":
    st.title("Player Demographics")
    confeds = sorted(df["confederation"].unique())
    sel_confed = st.multiselect("Filter by confederation (local to this section)", confeds, default=confeds, key="demo_confed")
    d = df[df["confederation"].isin(sel_confed)]
    st.caption(f"Showing {len(d):,} of {len(df):,} players")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Players by Position")
        pos_counts = d["position"].value_counts().reset_index()
        pos_counts.columns = ["position", "players"]
        fig = px.bar(pos_counts, x="position", y="players", color="position", color_discrete_sequence=PALETTE,
                     labels={"players": "Players", "position": "Position"})
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=420)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("#### Age Distribution")
        age_bins = d.groupby(d["age_years"].round(0))["player_id"].count().reset_index()
        age_bins.columns = ["age", "players"]
        fig = px.bar(age_bins, x="age", y="players", labels={"age": "Age (years)", "players": "Players"},
                     color_discrete_sequence=[SKY])
        fig.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Height by Position (distribution)")
    fig = px.box(d, x="position", y="height_cm", color="position", color_discrete_sequence=PALETTE,
                 labels={"height_cm": "Height (cm)", "position": "Position"})
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=420)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Confederation → Age Group → Position")
    sb = d.groupby(["confederation", "age_group", "position"]).size().reset_index(name="players")
    fig = px.sunburst(sb, path=["confederation", "age_group", "position"], values="players",
                       color="confederation", color_discrete_sequence=PALETTE)
    fig.update_layout(**PLOTLY_LAYOUT, height=560)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# SECTION 3 — PERFORMANCE ANALYTICS
# ============================================================================
elif section == "📈 Performance Analytics":
    st.title("Performance Analytics")
    positions = sorted(df["position"].unique())
    sel_pos = st.multiselect("Filter by position (local to this section)", positions, default=positions, key="perf_pos")
    d = df[df["position"].isin(sel_pos)]
    st.caption(f"Showing {len(d):,} of {len(df):,} players")

    st.markdown("#### Average Goal Involvement by Age (Line Chart)")
    trend = d.groupby(d["age_years"].round(0)).agg(
        avg_goal_involvement=("goal_involvement", "mean"),
        avg_tournament_goals=("tournament_goals", "mean")
    ).reset_index().rename(columns={"age_years": "age"})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["age"], y=trend["avg_goal_involvement"], mode="lines+markers",
                              name="Avg. Goal Involvement (Goals+Assists)", line=dict(color=PITCH_GREEN, width=3)))
    fig.add_trace(go.Scatter(x=trend["age"], y=trend["avg_tournament_goals"], mode="lines+markers",
                              name="Avg. Tournament Goals", line=dict(color=GOLD, width=3)))
    fig.update_layout(**PLOTLY_LAYOUT, height=440, xaxis_title="Age (years)", yaxis_title="Average per player")
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 15 Goal Scorers (Tournament)")
        top_scorers = d.nlargest(15, "tournament_goals")[["player_name", "team_name", "tournament_goals"]]
        fig = px.bar(top_scorers, x="tournament_goals", y="player_name", orientation="h", color="team_name",
                     color_discrete_sequence=PALETTE, labels={"tournament_goals": "Tournament Goals", "player_name": "Player"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=460)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("#### Minutes Played vs Goal Involvement")
        fig = px.scatter(d, x="minutes_played", y="goal_involvement", color="position",
                          color_discrete_sequence=PALETTE, hover_name="player_name",
                          labels={"minutes_played": "Minutes Played", "goal_involvement": "Goal Involvement"})
        fig.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Discipline: Yellow/Red Cards by Position")
    disc = d.groupby("position").agg(yellow_cards=("yellow_cards", "sum"), red_cards=("red_cards", "sum")).reset_index()
    fig = px.bar(disc, x="position", y=["yellow_cards", "red_cards"], barmode="group",
                 color_discrete_sequence=[GOLD, CORAL], labels={"value": "Cards", "position": "Position"})
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# SECTION — ADVANCED KPIs (per-90, percentages, per-cap ratios)
# ============================================================================
elif section == "📊 Advanced KPIs":
    st.title("Advanced KPIs")
    st.markdown('<p class="section-note">Standard football-analytics ratios, computed on tournament goals/assists '
                '(career totals only for Goals per Cap). Ratios are undefined — not zero — for players with 0 '
                'minutes or 0 goals, and are excluded from the relevant charts rather than shown as false zeros.</p>',
                unsafe_allow_html=True)

    min_minutes = st.slider("Minimum minutes played (local to this section, avoids small-sample noise)",
                             0, 400, 180, 30, key="kpi_min_minutes")
    d = df[df["minutes_played"] >= min_minutes].copy()
    st.caption(f"{len(d):,} of {len(df):,} players meet this minutes threshold")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 15 — Goal Contribution per 90")
        top = d.nlargest(15, "goal_contribution_per_90")[["player_name","team_name","goal_contribution_per_90"]]
        fig = px.bar(top, x="goal_contribution_per_90", y="player_name", orientation="h", color="team_name",
                     color_discrete_sequence=PALETTE, labels={"goal_contribution_per_90": "Goal Contribution / 90", "player_name": "Player"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=460, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("#### Goals per 90 vs Assists per 90")
        fig = px.scatter(d, x="goals_per_90", y="assists_per_90", color="position", size="minutes_played",
                          hover_name="player_name", color_discrete_sequence=PALETTE,
                          labels={"goals_per_90": "Goals / 90", "assists_per_90": "Assists / 90"})
        fig.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Minutes per Goal — Most Efficient Scorers (lower is better)")
    min_goals = st.slider("Minimum tournament goals to qualify (local to this section)", 1, 5, 2, key="kpi_min_goals")
    eff = d[d["tournament_goals"] >= min_goals].nsmallest(15, "minutes_per_goal")[["player_name","team_name","tournament_goals","minutes_per_goal"]]
    fig = px.bar(eff, x="minutes_per_goal", y="player_name", orientation="h", color="tournament_goals",
                 color_continuous_scale=[GOLD, PITCH_GREEN], labels={"minutes_per_goal": "Minutes per Goal", "player_name": "Player"})
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=460)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Goalkeeper KPIs — Save % and Clean Sheet %")
    min_matches_gk = st.slider("Minimum matches played (goalkeepers, local to this section)", 1, 8, 3, key="kpi_gk_matches")
    gk = df[(df["position"] == "GK") & (df["matches_played"] >= min_matches_gk)]
    col3, col4 = st.columns(2)
    with col3:
        top_save = gk.nlargest(12, "save_percentage")[["player_name","team_name","save_percentage"]]
        fig = px.bar(top_save, x="save_percentage", y="player_name", orientation="h", color="team_name",
                     color_discrete_sequence=PALETTE, labels={"save_percentage": "Save %", "player_name": "Goalkeeper"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=420, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with col4:
        top_cs = gk.nlargest(12, "clean_sheet_percentage")[["player_name","team_name","clean_sheet_percentage"]]
        fig = px.bar(top_cs, x="clean_sheet_percentage", y="player_name", orientation="h", color="team_name",
                     color_discrete_sequence=PALETTE, labels={"clean_sheet_percentage": "Clean Sheet %", "player_name": "Goalkeeper"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=420, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Start Percentage by Position (Line Chart — Squad Trust Signal)")
    start_trend = df.groupby("position")["start_percentage"].mean().reset_index().sort_values("start_percentage")
    fig = go.Figure(go.Scatter(x=start_trend["position"], y=start_trend["start_percentage"], mode="lines+markers",
                                line=dict(color=SKY, width=3), marker=dict(size=10)))
    fig.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Position", yaxis_title="Avg. Start %")
    st.plotly_chart(fig, width='stretch')

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### Top 15 — Market Value per Cap")
        top_vpc = df.nlargest(15, "market_value_per_cap")[["player_name","team_name","caps","market_value_per_cap"]]
        top_vpc["market_value_per_cap"] = top_vpc["market_value_per_cap"] / 1e6
        fig = px.bar(top_vpc, x="market_value_per_cap", y="player_name", orientation="h", color="caps",
                     color_continuous_scale=[CHALK, GOLD], labels={"market_value_per_cap": "Market Value / Cap (€M)", "player_name": "Player"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=460)
        st.plotly_chart(fig, width='stretch')
    with col6:
        st.markdown("#### Top 15 — Goals per Cap (Career)")
        top_gpc = df[df["caps"] >= 5].nlargest(15, "goals_per_cap")[["player_name","team_name","caps","career_goals","goals_per_cap"]]
        fig = px.bar(top_gpc, x="goals_per_cap", y="player_name", orientation="h", color="team_name",
                     color_discrete_sequence=PALETTE, labels={"goals_per_cap": "Career Goals per Cap", "player_name": "Player"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=460, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Full KPI Table")
    kpi_table_cols = ["player_name","team_name","position","minutes_played","tournament_goals","assists",
                       "goals_per_90","assists_per_90","goal_contribution_per_90","minutes_per_goal",
                       "start_percentage","market_value_per_cap","goals_per_cap","avg_minutes_per_match"]
    st.dataframe(d[kpi_table_cols].round(2).rename(columns={
        "player_name": "Player", "team_name": "Team", "position": "Pos", "minutes_played": "Minutes",
        "tournament_goals": "Goals", "assists": "Assists"
    }), width='stretch', hide_index=True)

# ============================================================================
# SECTION 4 — MARKET VALUE INSIGHTS
# ============================================================================
elif section == "💰 Market Value Insights":
    st.title("Market Value Insights")
    confeds = sorted(df["confederation"].unique())
    sel_confed = st.selectbox("Focus on a confederation (local to this section)", ["All"] + confeds, key="mv_confed")
    d = df if sel_confed == "All" else df[df["confederation"] == sel_confed]
    st.caption(f"Showing {len(d):,} of {len(df):,} players")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 15 Most Valuable Players")
        top_val = d.nlargest(15, "market_value_eur")[["player_name", "team_name", "market_value_eur"]]
        top_val["market_value_eur"] = top_val["market_value_eur"] / 1e6
        fig = px.bar(top_val, x="market_value_eur", y="player_name", orientation="h", color="team_name",
                     color_discrete_sequence=PALETTE, labels={"market_value_eur": "Market Value (€M)", "player_name": "Player"})
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=460)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("#### Average Market Value by Age (Line Chart)")
        trend = d.groupby(d["age_years"].round(0))["market_value_eur"].mean().reset_index()
        trend.columns = ["age", "avg_value"]
        trend["avg_value"] = trend["avg_value"] / 1e6
        fig = px.line(trend, x="age", y="avg_value", markers=True, color_discrete_sequence=[SKY],
                       labels={"age": "Age (years)", "avg_value": "Avg. Market Value (€M)"})
        fig.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Player Value Flow: Confederation → Experience Tier → Value Tier")
    d2 = d.copy()
    d2["value_tier"] = np.where(d2["is_high_value"] == 1, "High Value (Top 25%)", "Standard Value")
    flow = d2.groupby(["confederation", "experience_tier", "value_tier"]).size().reset_index(name="count")
    confed_list = sorted(flow["confederation"].unique())
    tier_list = sorted(flow["experience_tier"].unique())
    val_list = sorted(flow["value_tier"].unique())
    nodes = confed_list + tier_list + val_list
    node_idx = {n: i for i, n in enumerate(nodes)}
    src, tgt, val = [], [], []
    for _, r in flow.groupby(["confederation", "experience_tier"])["count"].sum().reset_index().iterrows():
        src.append(node_idx[r["confederation"]]); tgt.append(node_idx[r["experience_tier"]]); val.append(r["count"])
    for _, r in flow.groupby(["experience_tier", "value_tier"])["count"].sum().reset_index().iterrows():
        src.append(node_idx[r["experience_tier"]]); tgt.append(node_idx[r["value_tier"]]); val.append(r["count"])
    node_colors = ([PITCH_GREEN]*len(confed_list)) + ([GOLD]*len(tier_list)) + ([CORAL, SKY][:len(val_list)])
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=14, thickness=16, color=node_colors, line=dict(color="white", width=0.5)),
        link=dict(source=src, target=tgt, value=val, color="rgba(11,110,79,0.25)")
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=520)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# SECTION 5 — TEAM & SQUAD ANALYSIS
# ============================================================================
elif section == "🌍 Team & Squad Analysis":
    st.title("Team & Squad Analysis")
    teams = sorted(df["team_name"].unique())
    top_n = st.slider("Show top N squads by total market value (local to this section)", 5, 48, 20, key="team_topn")
    team_df = df.groupby(["team_name", "confederation"]).agg(
        total_value=("market_value_eur", "sum"), avg_age=("age_years", "mean"),
        elo=("elo_rating", "first"), fifa_rank=("fifa_ranking_pre_tournament", "first"),
        squad_size=("player_id", "count")
    ).reset_index().sort_values("total_value", ascending=False).head(top_n)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Squad Value vs FIFA Ranking")
        fig = px.scatter(team_df, x="fifa_rank", y="total_value", size="squad_size", color="confederation",
                          color_discrete_sequence=PALETTE, hover_name="team_name",
                          labels={"fifa_rank": "FIFA Ranking (pre-tournament, lower = better)", "total_value": "Total Squad Value (€)"})
        fig.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("#### Elo Rating vs Average Squad Age")
        fig = px.scatter(team_df, x="avg_age", y="elo", color="confederation", size="total_value",
                          color_discrete_sequence=PALETTE, hover_name="team_name",
                          labels={"avg_age": "Average Squad Age", "elo": "Elo Rating"})
        fig.update_layout(**PLOTLY_LAYOUT, height=460)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Confederation → Squad Breakdown (by Total Market Value)")
    sb = team_df.copy()
    sb["total_value_m"] = sb["total_value"] / 1e6
    fig = px.sunburst(sb, path=["confederation", "team_name"], values="total_value_m",
                       color="confederation", color_discrete_sequence=PALETTE)
    fig.update_layout(**PLOTLY_LAYOUT, height=560)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# SECTION — CLUB & GROUP INSIGHTS (previously unused columns: club_team, group_letter)
# ============================================================================
elif section == "🏟️ Club & Group Insights":
    st.title("Club & Group Insights")
    st.markdown('<p class="section-note">Which club sides will have the most players on show at the tournament, '
                'and how does squad strength vary across the 12 group-stage groups?</p>', unsafe_allow_html=True)

    st.markdown("#### Top Clubs by Number of Players Sent to the Tournament")
    top_n_clubs = st.slider("Number of clubs to show (local to this section)", 5, 30, 15, key="club_topn")
    club_counts = df["club_team"].value_counts().head(top_n_clubs).reset_index()
    club_counts.columns = ["club_team", "players"]
    fig = px.bar(club_counts, x="players", y="club_team", orientation="h", color="players",
                 color_continuous_scale=[CHALK, PITCH_GREEN],
                 labels={"players": "Players at the Tournament", "club_team": "Club"})
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=520, coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')
    st.caption(f"{df['club_team'].nunique()} clubs are represented across all 48 squads.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Group Strength — Average Elo Rating")
        grp = df.groupby("group_letter").agg(
            avg_elo=("elo_rating", "mean"), avg_value=("market_value_eur", "mean"),
            avg_fifa_rank=("fifa_ranking_pre_tournament", "mean")
        ).reset_index().sort_values("avg_elo", ascending=False)
        fig = px.bar(grp, x="group_letter", y="avg_elo", color="avg_elo", color_continuous_scale=[GOLD, PITCH_GREEN],
                     labels={"group_letter": "Group", "avg_elo": "Avg. Elo Rating"})
        fig.update_layout(**PLOTLY_LAYOUT, height=440, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')
        toughest = grp.iloc[0]["group_letter"]
        st.caption(f"Group **{toughest}** has the highest average Elo rating — the closest thing to a 'group of death' by this measure.")
    with col2:
        st.markdown("#### Group → Team Market Value (Treemap)")
        grp_team = df.groupby(["group_letter", "team_name"])["market_value_eur"].sum().reset_index()
        grp_team["value_m"] = grp_team["market_value_eur"] / 1e6
        fig = px.treemap(grp_team, path=["group_letter", "team_name"], values="value_m",
                          color="group_letter", color_discrete_sequence=PALETTE,
                          labels={"value_m": "Squad Value (€M)"})
        fig.update_layout(**PLOTLY_LAYOUT, height=440)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Most/Least Disciplined Squads (Discipline Index = Yellow Cards + 3×Red Cards)")
    disc_team = df.groupby("team_name")["discipline_index"].sum().reset_index().sort_values("discipline_index", ascending=False)
    n_show = st.slider("Number of teams to show (local to this section)", 5, 20, 10, key="disc_topn")
    fig = px.bar(pd.concat([disc_team.head(n_show), disc_team.tail(n_show)]).drop_duplicates(),
                 x="discipline_index", y="team_name", orientation="h", color="discipline_index",
                 color_continuous_scale=["#0D9488", CORAL],
                 labels={"discipline_index": "Total Discipline Index", "team_name": "Team"})
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=520, coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# SECTION — PLAYER EXPLORER & COMPARISON
# ============================================================================
elif section == "🔎 Player Explorer & Comparison":
    st.title("Player Explorer & Comparison")
    st.markdown('<p class="section-note">Search the full squad database, or compare up to 4 players head-to-head '
                'on a normalized skill profile.</p>', unsafe_allow_html=True)

    st.markdown("#### Search Players")
    search = st.text_input("Search by player name, team, or club (local to this section)", key="explorer_search")
    explorer_cols = ["player_name","team_name","club_team","position","age_years","market_value_eur",
                      "caps","tournament_goals","assists","goals_per_90","elo_rating"]
    d = df.copy()
    if search:
        mask = (d["player_name"].str.contains(search, case=False, na=False) |
                d["team_name"].str.contains(search, case=False, na=False) |
                d["club_team"].str.contains(search, case=False, na=False))
        d = d[mask]
    st.caption(f"{len(d):,} players match")
    st.dataframe(d[explorer_cols].round(2).rename(columns={
        "player_name": "Player", "team_name": "Team", "club_team": "Club", "position": "Pos",
        "age_years": "Age", "market_value_eur": "Market Value (€)", "caps": "Caps",
        "tournament_goals": "Goals", "assists": "Assists", "goals_per_90": "Goals/90", "elo_rating": "Squad Elo"
    }).head(200), width='stretch', hide_index=True)

    st.markdown("#### Radar Comparison")
    default_players = df.nlargest(2, "market_value_eur")["player_name"].tolist()
    sel_players = st.multiselect("Choose up to 4 players to compare (local to this section)",
                                  sorted(df["player_name"].unique()), default=default_players,
                                  max_selections=4, key="radar_players")
    if len(sel_players) >= 2:
        radar_metrics = ["market_value_eur", "caps", "tournament_goals", "assists", "elo_rating", "minutes_played"]
        radar_labels = ["Market Value", "Caps", "Goals", "Assists", "Squad Elo", "Minutes Played"]
        radar_df = df[df["player_name"].isin(sel_players)].set_index("player_name")[radar_metrics]
        # Normalize each metric 0-100 across the FULL dataset so the radar is comparable, not just among selected players
        norm_df = radar_df.copy()
        for c in radar_metrics:
            lo, hi = df[c].min(), df[c].max()
            norm_df[c] = ((radar_df[c] - lo) / (hi - lo) * 100) if hi > lo else 50

        fig = go.Figure()
        for i, p in enumerate(sel_players):
            fig.add_trace(go.Scatterpolar(
                r=norm_df.loc[p, radar_metrics].tolist() + [norm_df.loc[p, radar_metrics].tolist()[0]],
                theta=radar_labels + [radar_labels[0]], fill="toself", name=p,
                line=dict(color=PALETTE[i % len(PALETTE)])
            ))
        fig.update_layout(**PLOTLY_LAYOUT, height=520, polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(fig, width='stretch')
        st.caption("Each axis is normalized 0–100 against the full 1,248-player dataset, so shapes are comparable "
                   "even though the underlying units differ.")
    else:
        st.info("Select at least 2 players to see the radar comparison.")

# ============================================================================
# SECTION — CORRELATIONS & DEEP DIVE
# ============================================================================
elif section == "🔥 Correlations & Deep Dive":
    st.title("Correlations & Deep Dive")

    st.markdown("#### Interactive Correlation Heatmap")
    corr_options = ["market_value_eur","caps","height_cm","career_goals","matches_played","minutes_played",
                     "tournament_goals","assists","yellow_cards","red_cards","age_years","elo_rating",
                     "fifa_ranking_pre_tournament","start_rate","goal_involvement","discipline_index"]
    sel_corr_vars = st.multiselect("Variables to include (local to this section)", corr_options,
                                    default=["market_value_eur","caps","age_years","elo_rating",
                                             "fifa_ranking_pre_tournament","tournament_goals","assists"],
                                    key="corr_vars")
    if len(sel_corr_vars) >= 2:
        corr_matrix = df[sel_corr_vars].corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         labels={"color": "Correlation"})
        fig.update_layout(**PLOTLY_LAYOUT, height=max(420, len(sel_corr_vars)*45))
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Select at least 2 variables.")

    st.markdown("#### Squad Value Hierarchy (Treemap: Confederation → Team → Player)")
    top_n_players_treemap = st.slider("Top N players by value per team shown (local to this section)", 3, 15, 5, key="treemap_topn")
    treemap_df = df.sort_values("market_value_eur", ascending=False).groupby("team_name").head(top_n_players_treemap).copy()
    treemap_df["value_m"] = treemap_df["market_value_eur"] / 1e6
    fig = px.treemap(treemap_df, path=["confederation", "team_name", "player_name"], values="value_m",
                      color="confederation", color_discrete_sequence=PALETTE,
                      labels={"value_m": "Market Value (€M)"})
    fig.update_layout(**PLOTLY_LAYOUT, height=600)
    st.plotly_chart(fig, width='stretch')
    st.caption(f"Showing each team's top {top_n_players_treemap} most valuable players only, for readability.")

    st.markdown("#### Age vs Market Value, with Trendline (by Confederation)")
    sel_confeds_trend = st.multiselect("Confederations to show (local to this section)",
                                        sorted(df["confederation"].unique()),
                                        default=sorted(df["confederation"].unique()), key="deepdive_confed")
    d = df[df["confederation"].isin(sel_confeds_trend)]
    fig = px.scatter(d, x="age_years", y="market_value_eur", color="confederation", trendline="lowess",
                      color_discrete_sequence=PALETTE, opacity=0.5,
                      labels={"age_years": "Age", "market_value_eur": "Market Value (€)"})
    fig.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# SECTION 6 — PLAYER ARCHETYPES (CLUSTERING & PCA)
# ============================================================================
elif section == "🧩 Player Archetypes (Clustering)":
    st.title("Player Archetypes — K-Means Clustering & PCA")
    st.markdown('<p class="section-note">Players were clustered on market value, caps, age, goal output, minutes, '
                'discipline, and Elo context (standardized features), then projected to 2D via PCA for visualization.</p>',

                unsafe_allow_html=True)
    clusters = sorted(df["cluster"].unique())
    sel_clusters = st.multiselect("Filter clusters shown (local to this section)", clusters, default=clusters, key="clu_sel")
    d = df[df["cluster"].isin(sel_clusters)]

    st.markdown("#### PCA Projection Colored by Cluster")
    fig = px.scatter(d, x="pca1", y="pca2", color=d["cluster"].astype(str), hover_name="player_name",
                      hover_data=["position", "team_name", "market_value_eur"],
                      color_discrete_sequence=PALETTE, labels={"color": "Cluster"})
    fig.update_layout(**PLOTLY_LAYOUT, height=520)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Cluster Profiles — Average Feature Values")
    profile_cols = ["market_value_eur_capped", "caps", "age_years", "career_goals", "tournament_goals",
                     "assists", "minutes_per_match", "elo_rating"]
    prof = d.groupby("cluster")[profile_cols].mean().reset_index()
    prof_long = prof.melt(id_vars="cluster", var_name="feature", value_name="avg_value")
    fig = px.bar(prof_long, x="feature", y="avg_value", color=prof_long["cluster"].astype(str), barmode="group",
                 color_discrete_sequence=PALETTE, labels={"color": "Cluster"})
    fig.update_layout(**PLOTLY_LAYOUT, height=460, xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Cluster Composition by Position")
    comp = d.groupby(["cluster", "position"]).size().reset_index(name="players")
    fig = px.bar(comp, x="cluster", y="players", color="position", barmode="stack",
                 color_discrete_sequence=PALETTE, labels={"cluster": "Cluster"})
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width='stretch')

    # ------------------------------------------------------------------
    # DEEP DIVE — why clustering, why k=4, and the PCA caveat that matters
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Deep Dive — Why K-Means, Why k=4, and What the PCA Plot Doesn't Show")

    st.markdown("#### Why clustering at all")
    st.markdown('<p class="section-note">The goal is to find natural player archetypes — groups of players who look '
                'similar across a <i>combination</i> of traits (value, experience, output, playing time) rather than '
                'along any single stat. That is an unsupervised problem — there is no label for "type of player" — '
                'so K-Means is the natural fit: fast, interpretable, and effective on continuous, standardized '
                'numeric features.</p>', unsafe_allow_html=True)

    st.markdown("**Features used (all standardized with `StandardScaler` before clustering):**")
    st.markdown(
        "- `market_value_eur_capped` — market value winsorized at the 1st/99th percentile, so a few "
        "Mbappé/Haaland-tier valuations don't dominate Euclidean distance and swamp every other feature "
        "(the raw column is kept for reporting everywhere else in the dashboard)\n"
        "- `caps` — career international experience\n"
        "- `age_years` — age\n"
        "- `career_goals` — longer-run scoring output (club + country)\n"
        "- `tournament_goals`, `assists` — this tournament's output\n"
        "- `minutes_per_match`, `start_rate` — how much they actually play, separating starters from squad depth\n"
        "- `discipline_index` — yellow + 3×red cards, a behavioral signal\n"
        "- `elo_rating` — the national squad's Elo, i.e. team context"
    )
    st.caption("Clustering runs on these 10 standardized features directly, not on the PCA output. PCA is only "
               "used afterward to compress those 10 dimensions down to 2 so the clusters can be plotted — it's a "
               "visualization step, not part of the clustering itself.")

    st.markdown("#### Why 4 clusters specifically")
    k_sweep = pd.DataFrame([
        {"k": 2, "Inertia": 10011.7, "Silhouette": 0.2254},
        {"k": 3, "Inertia": 8653.0, "Silhouette": 0.2418},
        {"k": 4, "Inertia": 7675.6, "Silhouette": 0.2525},
        {"k": 5, "Inertia": 7032.7, "Silhouette": 0.2259},
        {"k": 6, "Inertia": 6457.4, "Silhouette": 0.2279},
        {"k": 7, "Inertia": 6024.6, "Silhouette": 0.2337},
        {"k": 8, "Inertia": 5679.3, "Silhouette": 0.1842},
        {"k": 9, "Inertia": 5300.4, "Silhouette": 0.1974},
    ])
    col_k1, col_k2 = st.columns([1, 1.4])
    with col_k1:
        st.dataframe(k_sweep.style.apply(
            lambda row: ["background-color: rgba(212,175,55,0.25); font-weight:600"] * len(row)
            if row["k"] == 4 else [""] * len(row), axis=1
        ), width='stretch', hide_index=True)
    with col_k2:
        fig = go.Figure(go.Scatter(x=k_sweep["k"], y=k_sweep["Silhouette"], mode="lines+markers",
                                    line=dict(color=PITCH_GREEN, width=3), marker=dict(size=9)))
        fig.add_vline(x=4, line_dash="dash", line_color=GOLD)
        fig.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="k (number of clusters)", yaxis_title="Silhouette score")
        st.plotly_chart(fig, width='stretch')
    st.markdown('<p class="section-note">The notebook swept k = 2 through 9, computing inertia (elbow) and silhouette '
                'score at each k, then picked the k with the highest silhouette score. <b>k=4 wins</b>, though only '
                'narrowly over k=3 — the silhouette scores overall are modest (~0.25), which is honest: a 1,248-player '
                'dataset with this much within-position and within-team variety doesn\'t split into razor-sharp '
                'clusters. It\'s a real, if soft, structure. Final model: <code>KMeans(n_clusters=4, random_state=42, '
                'n_init=10)</code>.</p>', unsafe_allow_html=True)

    st.markdown("#### PCA — the caveat that matters")
    pc1_var, pc2_var = 0.286, 0.190
    c1, c2, c3 = st.columns(3)
    c1.metric("PC1 variance explained", f"{pc1_var*100:.1f}%")
    c2.metric("PC2 variance explained", f"{pc2_var*100:.1f}%")
    c3.metric("Total variance captured", f"{(pc1_var+pc2_var)*100:.1f}%")
    st.markdown('<p class="section-note"><code>PCA(n_components=2)</code> on the same 10 scaled features explains only '
                '<b>28.6% + 19.0% = 47.6%</b> of total variance. That means the 2D scatter above is a fairly lossy '
                'projection — over half the variance in how players actually differ isn\'t visible in that plot. '
                '<b>The clusters themselves are correct</b> (computed in the full 10D space); it\'s just that the 2D '
                'picture compresses them, so don\'t over-read exact positions or boundaries in the scatter.</p>',
                unsafe_allow_html=True)
    st.markdown(
        "- **PC1** (loadings: `minutes_per_match` 0.43, `start_rate` 0.42, `caps` 0.39, `career_goals` 0.36) "
        "→ a *\"playing time & experience\"* axis\n"
        "- **PC2** (loadings: `market_value` 0.55, `elo_rating` 0.44, `age` −0.47, `caps` −0.37) "
        "→ a *\"young, valuable, top-squad\"* axis"
    )

    st.markdown("#### The four clusters, in detail")
    archetype_meta = {
        0: ("Squad Regulars", "Solid starters at modest value and modest Elo. Position mix skews defensive/central. "
                               "This is the backbone of most squads — reliable, unspectacular internationally."),
        1: ("Elite Value Core", "By far the highest market value and squad Elo, i.e. they play for the strongest "
                                 "national teams. Youngest average age and the best combined output. Attacking-leaning "
                                 "and concentrated in UEFA and CONMEBOL — the star-player-at-a-top-club, "
                                 "playing-for-a-top-nation archetype."),
        2: ("Veteran Leaders", "Extreme outlier on experience — caps and career goals both dramatically higher than "
                                "any other cluster. Oldest group but still delivering the highest tournament-goal rate. "
                                "Almost entirely forwards and midfielders — the long-serving, high-mileage attacking "
                                "veterans every squad carries one or two of."),
        3: ("Fringe / Non-Playing Squad", "Minutes per match and start rate collapse relative to every other cluster. "
                                           "Discipline index drops to near-zero, simply because you can't rack up "
                                           "cards from the bench. Roughly evenly spread across positions — squad "
                                           "depth that barely features, not any one football \"type.\""),
    }
    prof_full_cols = ["market_value_eur", "caps", "age_years", "career_goals", "tournament_goals",
                       "assists", "minutes_per_match", "start_rate", "elo_rating"]
    prof_full = df.groupby("cluster")[prof_full_cols].mean()
    counts_full = df.groupby("cluster").size()
    pos_full = df.groupby(["cluster", "position"]).size().unstack(fill_value=0)

    cluster_cols = st.columns(len(archetype_meta))
    for i, (c, (name, summary)) in enumerate(archetype_meta.items()):
        with cluster_cols[i]:
            st.markdown(f"**Cluster {c} — {name}**")
            st.caption(f"{counts_full.get(c, 0)} players")
            st.markdown(summary)
            row = prof_full.loc[c] if c in prof_full.index else None
            if row is not None:
                st.table(pd.DataFrame({
                    "": ["Avg. value", "Avg. caps", "Avg. age", "Career goals", "Tourn. goals",
                         "Min/match", "Start rate", "Squad Elo"],
                    " ": [f"€{row['market_value_eur']/1e6:.1f}M", f"{row['caps']:.1f}", f"{row['age_years']:.1f}",
                          f"{row['career_goals']:.1f}", f"{row['tournament_goals']:.2f}",
                          f"{row['minutes_per_match']:.1f}", f"{row['start_rate']*100:.0f}%",
                          f"{row['elo_rating']:.0f}"]
                }).set_index(""))
            if c in pos_full.index:
                pos_str = ", ".join(f"{p} {n}" for p, n in pos_full.loc[c].sort_values(ascending=False).items() if n > 0)
                st.caption(f"Positions: {pos_str}")

    st.info("**The honest summary:** the clustering separates players mainly along two practical dimensions — "
            "how much they play (Clusters 0/2 vs. 3) and how valuable/central they are to a top squad (Cluster 1 "
            "vs. the rest) — with Cluster 2 carved out almost entirely by extreme career longevity. It's a "
            "legitimate, data-driven segmentation, but with a silhouette score around 0.25, treat it as a useful "
            "lens rather than hard categories — plenty of players sit near cluster boundaries.")

# ============================================================================
# SECTION 7 — ASSOCIATION RULES
# ============================================================================
elif section == "🔗 Association Rules":
    st.title("Association Rule Mining")
    st.markdown('<p class="section-note">Rules mined via Apriori on binned player traits (High/Low Market Value, Caps, '
                'Minutes, Goal Involvement, Discipline, Elo Rating, Position). Lift > 1 means the traits co-occur '
                'more than random chance would predict.</p>', unsafe_allow_html=True)

    min_lift = st.slider("Minimum lift (local to this section)", 1.0, float(rules["lift"].max().round(2)), 1.3, 0.05, key="rules_lift")
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05, key="rules_conf")
    r = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_conf)].sort_values("lift", ascending=False).head(20)
    r["rule"] = r["antecedents"] + "  ⟶  " + r["consequents"]

    st.markdown(f"#### Top {len(r)} Rules by Lift")
    fig = px.bar(r, x="lift", y="rule", orientation="h", color="confidence",
                 color_continuous_scale=[CHALK, PITCH_GREEN],
                 labels={"lift": "Lift", "rule": "Rule (Antecedent ⟶ Consequent)"})
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"), height=max(420, len(r)*28))
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Full Rule Table")
    st.dataframe(r[["antecedents", "consequents", "support", "confidence", "lift"]]
                 .rename(columns={"antecedents": "If (Antecedent)", "consequents": "Then (Consequent)"})
                 .round(3), width='stretch', hide_index=True)

# ============================================================================
# SECTION 8 — CLASSIFICATION MODEL RESULTS
# ============================================================================
elif section == "🤖 Classification Model Results":
    st.title("Classification Model Results")
    st.markdown('<p class="section-note">Target: <b>is_high_value</b> — is a player in the top quartile of market value? '
                '80/20 stratified train-test split, 6 algorithms, then GridSearchCV (5-fold CV) hyperparameter tuning.</p>',
                unsafe_allow_html=True)

    feat_choice = st.radio(
        "Team-context features to include (local to this section — reruns the model live)",
        ["Elo rating only (default)", "Elo rating + FIFA Ranking"],
        horizontal=True, key="clf_team_feats",
        help="Elo rating and FIFA pre-tournament ranking both describe team strength rather than the player. "
             "Toggle this to see every metric, ROC curve, confusion matrix, and feature-importance ranking "
             "recalculated with or without FIFA Ranking in the feature set."
    )
    include_fifa = feat_choice.startswith("Elo rating +")
    _ml_clf = run_ml_pipeline(df, include_fifa_rank=include_fifa)
    model_baseline_clf = _ml_clf["model_baseline"]
    model_tuned_clf = _ml_clf["model_tuned"]
    roc_baseline_clf = _ml_clf["roc_baseline"]
    roc_tuned_clf = _ml_clf["roc_tuned"]
    cm_baseline_clf = _ml_clf["cm_baseline"]
    cm_tuned_clf = _ml_clf["cm_tuned"]
    feat_imp_clf = _ml_clf["feat_imp"]
    if include_fifa:
        st.caption("Currently training with **elo_rating + fifa_ranking_pre_tournament** both included as features.")
    else:
        st.caption("Currently training with **elo_rating only** — fifa_ranking_pre_tournament is excluded (default).")

    view = st.radio("Model set (local to this section)", ["Baseline (80/20 split)", "Tuned (GridSearchCV, 5-fold CV)"],
                     horizontal=True, key="clf_view")
    if view.startswith("Baseline"):
        table, roc_data, cm_data = model_baseline_clf, roc_baseline_clf, cm_baseline_clf
    else:
        table, roc_data, cm_data = model_tuned_clf, roc_tuned_clf, cm_tuned_clf

    st.markdown("#### Metrics Table — Accuracy, Precision, Recall, F1, ROC-AUC")
    st.dataframe(table.round(4).sort_values("ROC-AUC", ascending=False), width='stretch', hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Metric Comparison (Bar Chart)")
        # Baseline and Tuned tables have different columns (Tuned has no "Train Accuracy",
        # since it reports Resubstitution/OOF accuracy instead) — only melt columns that
        # actually exist in whichever table is currently selected, so this never KeyErrors.
        candidate_metrics = ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        available_metrics = [c for c in candidate_metrics if c in table.columns]
        metrics_long = table.melt(id_vars="Model", value_vars=available_metrics,
                                   var_name="Metric", value_name="Score")
        fig = px.bar(metrics_long, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(**PLOTLY_LAYOUT, height=460, xaxis_tickangle=-20)
        st.plotly_chart(fig, width='stretch')
    with col2:
        st.markdown("#### ROC Curve — All Algorithms")
        fig = go.Figure()
        for i, m in enumerate(roc_data["Model"].unique()):
            sub = roc_data[roc_data["Model"] == m]
            auc_val = sub["AUC"].iloc[0]
            fig.add_trace(go.Scatter(x=sub["fpr"], y=sub["tpr"], mode="lines",
                                      name=f"{m} (AUC={auc_val:.3f})",
                                      line=dict(color=PALETTE[i % len(PALETTE)], width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, height=460, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Confusion Matrices")
    sel_model = st.selectbox("Choose a model (local to this section)", cm_data["Model"].unique(), key="clf_cm_model")
    row = cm_data[cm_data["Model"] == sel_model].iloc[0]
    z = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
    fig = px.imshow(z, text_auto=True, x=["Predicted: Not High-Value", "Predicted: High-Value"],
                     y=["Actual: Not High-Value", "Actual: High-Value"],
                     color_continuous_scale=[CHALK, PITCH_GREEN])
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, width='stretch')

    if len(feat_imp_clf) > 0:
        st.markdown("#### Feature Importance — Best Overall Model (by Test ROC-AUC)")
        fig = px.bar(feat_imp_clf.sort_values("importance"), x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=[CHALK, GOLD],
                     labels={"importance": "Importance", "feature": "Feature"})
        fig.update_layout(**PLOTLY_LAYOUT, height=460, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

# ============================================================================
# SECTION 9 — MODEL IMPROVEMENTS (imbalance handling, thresholds, team-effect)
# ============================================================================
elif section == "🎯 Model Improvements":
    st.title("Model Improvements — Fixing Recall & Isolating the Team Effect")
    st.markdown('<p class="section-note">The classifiers above were originally weak on recall due to '
                'class imbalance (~26% high-value players). This section shows the three fixes applied: '
                'class-weighting, F1-scored GridSearchCV tuning, and decision-threshold tuning — plus a direct '
                'test of how much predictive power comes from "playing for a strong team" vs. individual output.</p>',
                unsafe_allow_html=True)

    st.markdown("#### Fix 4 — Was the 100% Train Accuracy Real Overfitting?")
    st.markdown('<p class="section-note">Random Forest, XGBoost, LightGBM and KNN originally showed ~100% '
                '<i>resubstitution</i> train accuracy (predicting on the exact rows they were fit on) — a classic red '
                'flag. But for KNN with distance-weighting, that number is a known artifact: a training point\'s '
                'nearest neighbor during self-prediction is itself, at distance 0, so it trivially votes for its own '
                'label. The honest test is <b>out-of-fold (OOF) accuracy</b> via cross-validation, where every row is '
                'only ever predicted by a model that did not see it. Compare the two below.</p>', unsafe_allow_html=True)
    gap_long = overfit_diag.melt(id_vars="Model", value_vars=["Resubstitution Gap", "OOF Gap (honest)"],
                                  var_name="Diagnostic", value_name="Train − Test Gap")
    fig = go.Figure()
    for diag, color in [("Resubstitution Gap", CORAL), ("OOF Gap (honest)", PITCH_GREEN)]:
        sub = gap_long[gap_long["Diagnostic"] == diag]
        fig.add_trace(go.Bar(x=sub["Model"], y=sub["Train − Test Gap"], name=diag,
                              marker_color=color))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=440, barmode="group", xaxis_tickangle=-20,
                       yaxis_title="Train − Test Accuracy Gap")
    st.plotly_chart(fig, width='stretch')
    st.caption("Large red bars = misleading. Small/negative green bars = the model actually generalizes fine; "
               "the resubstitution number was inflated by memorization capacity, not genuine overfitting.")


    rl = recall_lift.rename(columns={"Recall": "Recall_Lift"})
    rl_sorted = rl.sort_values("Recall_Lift", ascending=True)
    fig = px.bar(rl_sorted, x="Recall_Lift", y="Model", orientation="h",
                 color="Recall_Lift", color_continuous_scale=[CORAL, CHALK, PITCH_GREEN],
                 labels={"Recall_Lift": "Recall Change (Balanced − Unweighted)", "Model": "Algorithm"})
    fig.update_layout(**PLOTLY_LAYOUT, height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')
    st.caption("Positive bars = class-weighting improved recall for that algorithm. "
               "SVM and Logistic Regression gained the most; tree ensembles gained little on their own.")

    algo_choice = st.selectbox("Inspect unweighted vs balanced metrics for (local to this section)",
                                imbalance_comp["Model"].unique(), key="mi_algo")
    sub = imbalance_comp[imbalance_comp["Model"] == algo_choice]
    fig = px.bar(sub.melt(id_vars=["Model", "Weighting"], value_vars=["Precision", "Recall", "F1-Score"],
                           var_name="Metric", value_name="Score"),
                 x="Metric", y="Score", color="Weighting", barmode="group",
                 color_discrete_sequence=[CORAL, PITCH_GREEN])
    fig.update_layout(**PLOTLY_LAYOUT, height=380)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Fix 2 & 3 — GridSearchCV (scored on F1) + Decision-Threshold Tuning")
    st.markdown("Best model after tuning, evaluated at the default 0.50 cutoff vs. the F1-optimal cutoff "
                "found via 5-fold cross-validated probabilities on the **training set only**:")
    fig = px.bar(threshold_comp.melt(id_vars="Threshold", value_vars=["Precision", "Recall", "F1-Score", "Accuracy"],
                                      var_name="Metric", value_name="Score"),
                 x="Metric", y="Score", color="Threshold", barmode="group",
                 color_discrete_sequence=[SKY, GOLD])
    fig.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Team Context vs Individual Skill — Isolating the Effect")
    st.markdown("Same best algorithm, retrained with vs. without `elo_rating` (the squad's team-strength "
                "signal). `fifa_ranking_pre_tournament` has been removed from the model's feature set entirely, "
                "so `elo_rating` is now the only team-context feature being isolated here:")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.bar(team_vs_indiv.melt(id_vars="Feature Set",
                                         value_vars=["Precision", "Recall", "F1-Score", "ROC-AUC"],
                                         var_name="Metric", value_name="Score"),
                     x="Metric", y="Score", color="Feature Set", barmode="group",
                     color_discrete_sequence=[PITCH_GREEN, CORAL])
        fig.update_layout(**PLOTLY_LAYOUT, height=440, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width='stretch')
    with col2:
        auc_full = team_vs_indiv.iloc[0]["ROC-AUC"]
        auc_indiv = team_vs_indiv.iloc[1]["ROC-AUC"]
        st.metric("Full-model ROC-AUC", f"{auc_full:.3f}")
        st.metric("Individual-only ROC-AUC", f"{auc_indiv:.3f}", delta=f"{auc_indiv-auc_full:.3f}")
        st.metric("% discriminative power retained\nwithout team context", f"{auc_indiv/auc_full*100:.1f}%")

    if len(feat_imp_indiv) > 0:
        st.markdown("#### Feature Importance — With vs Without Team Context")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Full feature set")
            fig = px.bar(feat_imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                         color_discrete_sequence=[PITCH_GREEN])
            fig.update_layout(**PLOTLY_LAYOUT, height=420)
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.caption("Individual skill only (no Elo / FIFA ranking)")
            fig = px.bar(feat_imp_indiv.sort_values("importance"), x="importance", y="feature", orientation="h",
                         color_discrete_sequence=[CORAL])
            fig.update_layout(**PLOTLY_LAYOUT, height=420)
            st.plotly_chart(fig, width='stretch')
