"""Diabetes screening dashboard (Team B, DIRISA Datathon Week 1).

Rebuilds the final model of the team notebook (Shaylin's sensible LightGBM: 13 features,
monotonic rules, re-tuned settings, threshold for 78% recall) and lets you check one person.

Run from the task1 folder:  streamlit run dashboard/app.py
"""
from pathlib import Path

import altair as alt
import lime.lime_tabular
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"

# same choices as the team notebook (sections 4 and 7)
DROP_COLUMNS = ["Income", "Education", "CholCheck", "Fruits", "Veggies",
                "AnyHealthcare", "NoDocbcCost", "HvyAlcoholConsump"]
TARGET_RECALL = 0.78
MONOTONE_RULES = {"HighBP": 1, "HighChol": 1, "BMI": 1, "Stroke": 1, "HeartDiseaseorAttack": 1,
                  "GenHlth": 1, "PhysHlth": 1, "DiffWalk": 1, "Age": 1, "PhysActivity": -1}
# settings found by Shaylin's search for the sensible model (notebook section 7)
BEST_PARAMS = {"colsample_bytree": 0.8909650062296923, "learning_rate": 0.015771499561198804,
               "min_child_samples": 125, "n_estimators": 357, "num_leaves": 47,
               "subsample": 0.6424675409453354, "subsample_freq": 1}

# Team B palette (same hex codes as the slides), LIGHT theme.
# Roles: light page + white cards; dark greens for structure and text; gold for accents
# (bright Gold only for fills/borders/bars; Dark gold for gold-coloured text, as in the palette).
FOREST_DARK, FOREST_2, FOREST_3 = "#183A2A", "#22503A", "#2E6347"
SAGE, PALE_SAGE = "#7FA88A", "#9DBFA6"
GOLD, DEEP_GOLD, DARK_GOLD = "#D4AF37", "#B8921F", "#8A6D12"
OFF_WHITE = "#F1F5F0"
# tints = palette colour mixed with white (keep the light theme inside the palette's family)
PAGE_TINT, SAGE_TINT, GOLD_TINT, LINE_TINT = "#F3F7F4", "#E4EDE6", "#F9F3E1", "#D5E0DA"

PAGE_CSS = f"""
<style>
/* section headings: Forest 2 text with a gold accent bar */
[data-testid="stHeading"] h3 {{ color: {FOREST_2}; border-left: 5px solid {GOLD};
                               padding-left: 0.6rem; }}
/* metric tiles as white cards with a Forest 3 top edge */
[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE_TINT};
                           border-top: 4px solid {FOREST_3}; border-radius: 10px;
                           padding: 0.8rem 1rem; box-shadow: 0 1px 3px rgba(24, 58, 42, 0.08); }}
[data-testid="stMetricLabel"] {{ color: {FOREST_3}; }}
[data-testid="stMetricValue"] {{ color: {FOREST_DARK}; }}
/* tab labels a little larger */
[data-baseweb="tab"] p {{ font-size: 1.05rem; font-weight: 600; }}
</style>
"""

AGE_LABELS = ["18–24", "25–29", "30–34", "35–39", "40–44", "45–49", "50–54",
              "55–59", "60–64", "65–69", "70–74", "75–79", "80+"]
HEALTH_LABELS = ["Excellent", "Very good", "Good", "Fair", "Poor"]
FEATURE_NAMES = {"HighBP": "High blood pressure", "HighChol": "High cholesterol", "BMI": "BMI",
                 "Smoker": "Smoker (100+ cigarettes in life)", "Stroke": "Ever had a stroke",
                 "HeartDiseaseorAttack": "Heart disease or heart attack",
                 "PhysActivity": "Physically active (past 30 days)", "GenHlth": "General health",
                 "MentHlth": "Days of poor mental health (past 30)",
                 "PhysHlth": "Days of poor physical health (past 30)",
                 "DiffWalk": "Serious difficulty walking or climbing stairs", "Sex": "Sex", "Age": "Age"}

# the four people explained with LIME in notebook section 7 (sensible model)
EXAMPLES = {
    "Correct catch (true positive)": dict(HighBP=1, HighChol=1, BMI=31, Smoker=1, Stroke=0, HeartDiseaseorAttack=0,
                                          PhysActivity=0, GenHlth=3, MentHlth=0, PhysHlth=7, DiffWalk=1, Sex=0, Age=12),
    "Missed diabetic (false negative)": dict(HighBP=0, HighChol=1, BMI=32, Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
                                             PhysActivity=1, GenHlth=2, MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=1, Age=11),
    "False alarm (false positive)": dict(HighBP=0, HighChol=0, BMI=37, Smoker=1, Stroke=0, HeartDiseaseorAttack=0,
                                         PhysActivity=1, GenHlth=3, MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=0, Age=10),
    "Borderline case": dict(HighBP=1, HighChol=1, BMI=27, Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
                            PhysActivity=1, GenHlth=3, MentHlth=21, PhysHlth=0, DiffWalk=0, Sex=0, Age=5),
}


@st.cache_resource(show_spinner="Training the final model (first start only, about a minute)...")
def build_model():
    """Rebuilds the notebook's final model: same data, split, settings and threshold."""
    df = pd.read_csv(DATA_PATH).astype(int)
    X = df.drop(columns=["Diabetes_binary"] + DROP_COLUMNS)
    y = df["Diabetes_binary"]

    # grouped + stratified 80/20 split on the 13 features, random_state=17 (notebook section 4)
    groups = X.groupby(list(X.columns)).ngroup()
    train_idx, test_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
                               .split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    monotone = [MONOTONE_RULES.get(col, 0) for col in X_train.columns]
    model = LGBMClassifier(**BEST_PARAMS, monotone_constraints=monotone, scale_pos_weight=ratio,
                           random_state=17, verbose=-1)

    # threshold for 78% recall from grouped 3-fold out-of-fold training predictions (as in section 7)
    oof = cross_val_predict(model, X_train, y_train, groups=groups.iloc[train_idx],
                            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=17),
                            method="predict_proba")[:, 1]
    _, recall, thresholds = precision_recall_curve(y_train, oof)
    threshold = thresholds[np.where(recall[:-1] >= TARGET_RECALL)[0][-1]]

    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = test_proba >= threshold
    yt = y_test.values
    test_scores = {
        "Recall": recall_score(yt, test_pred), "Precision": precision_score(yt, test_pred),
        "ROC-AUC": roc_auc_score(yt, test_proba), "PR-AUC": average_precision_score(yt, test_proba),
        "Caught": int((test_pred & (yt == 1)).sum()), "Missed": int((~test_pred & (yt == 1)).sum()),
        "False alarms": int((test_pred & (yt == 0)).sum()), "Test people": len(yt),
        "Test diabetics": int(yt.sum()),
    }

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values, feature_names=list(X_train.columns),
        class_names=["No diabetes", "Diabetes"], mode="classification", random_state=17)
    return model, threshold, explainer, list(X_train.columns), test_scores, y.mean()


def answer_text(feature, value):
    """The person's answer in words."""
    if feature in ("BMI", "MentHlth", "PhysHlth"):
        return str(value)
    if feature == "GenHlth":
        return HEALTH_LABELS[value - 1]
    if feature == "Age":
        return AGE_LABELS[value - 1]
    if feature == "Sex":
        return "Male" if value == 1 else "Female"
    return "Yes" if value == 1 else "No"


def callout(text, accent, background):
    """A message box in the team palette (replaces Streamlit's default blue/yellow/green boxes)."""
    st.markdown(f'''<div style="background:{background}; border:1px solid {LINE_TINT};
                border-left:6px solid {accent}; padding:0.8rem 1rem; border-radius:8px;
                margin:0.4rem 0; color:{FOREST_DARK};">{text}</div>''', unsafe_allow_html=True)


def explain(model, explainer, columns, person):
    """LIME explanation for one person, as a table in plain words."""
    explainer.random_state = np.random.RandomState(17)   # same person → same explanation every time
    predict_fn = lambda rows: model.predict_proba(pd.DataFrame(rows, columns=columns))
    exp = explainer.explain_instance(person.values[0], predict_fn, num_features=8)
    rows = [{"Answer": FEATURE_NAMES[columns[i]], "Value": answer_text(columns[i], int(person.iloc[0, i])),
             "Effect on risk": weight} for i, weight in exp.as_map()[1]]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Diabetes screening", page_icon="🩺", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)
# header banner: Forest dark with off-white title and a gold accent line
st.markdown(f'''<div style="background:{FOREST_DARK}; border-bottom:5px solid {GOLD}; border-radius:12px;
            padding:1.2rem 1.5rem; margin-bottom:1rem;">
  <div style="color:{OFF_WHITE}; font-size:2rem; font-weight:700;">🩺 Diabetes screening from survey answers</div>
  <div style="color:{GOLD}; font-weight:600; margin-top:0.2rem;">Team B · DIRISA Datathon Week 1</div>
  <div style="color:{OFF_WHITE}; opacity:0.85; margin-top:0.2rem;">Final model: LightGBM without alcohol,
  with monotonic rules (notebook section 7). For learning only, not a medical diagnosis.</div>
</div>''', unsafe_allow_html=True)

model, threshold, explainer, columns, test_scores, base_rate = build_model()

tab_person, tab_model, tab_about = st.tabs(["Check a person", "How good is the model?", "About"])

with tab_person:
    example = st.selectbox("Start from", ["Custom"] + list(EXAMPLES),
                           help="The four people explained with LIME in the notebook.")
    start = EXAMPLES.get(example, dict(HighBP=0, HighChol=0, BMI=27, Smoker=0, Stroke=0, HeartDiseaseorAttack=0,
                                       PhysActivity=1, GenHlth=2, MentHlth=0, PhysHlth=0, DiffWalk=0, Sex=0, Age=7))
    yes_no = ["No", "Yes"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("About the person")
        sex = st.radio("Sex", ["Female", "Male"], index=start["Sex"], horizontal=True, key=f"sex_{example}")
        age = st.selectbox("Age", AGE_LABELS, index=start["Age"] - 1, key=f"age_{example}")
        bmi = st.number_input("BMI", min_value=12, max_value=98, value=start["BMI"], key=f"bmi_{example}")
        gen = st.select_slider("General health", HEALTH_LABELS, value=HEALTH_LABELS[start["GenHlth"] - 1],
                               key=f"gen_{example}")
    with c2:
        st.subheader("Conditions")
        high_bp = st.radio("High blood pressure", yes_no, index=start["HighBP"], horizontal=True, key=f"bp_{example}")
        high_chol = st.radio("High cholesterol", yes_no, index=start["HighChol"], horizontal=True, key=f"chol_{example}")
        stroke = st.radio("Ever had a stroke", yes_no, index=start["Stroke"], horizontal=True, key=f"stroke_{example}")
        heart = st.radio("Heart disease or heart attack", yes_no, index=start["HeartDiseaseorAttack"],
                         horizontal=True, key=f"heart_{example}")
        diff_walk = st.radio("Serious difficulty walking or climbing stairs", yes_no, index=start["DiffWalk"],
                             horizontal=True, key=f"walk_{example}")
    with c3:
        st.subheader("Lifestyle and recent health")
        smoker = st.radio("Smoked at least 100 cigarettes in life", yes_no, index=start["Smoker"],
                          horizontal=True, key=f"smoke_{example}")
        active = st.radio("Physical activity in the past 30 days", yes_no, index=start["PhysActivity"],
                          horizontal=True, key=f"active_{example}")
        ment = st.slider("Days of poor mental health (past 30)", 0, 30, start["MentHlth"], key=f"ment_{example}")
        phys = st.slider("Days of poor physical health (past 30)", 0, 30, start["PhysHlth"], key=f"phys_{example}")

    person = pd.DataFrame([{
        "HighBP": yes_no.index(high_bp), "HighChol": yes_no.index(high_chol), "BMI": int(bmi),
        "Smoker": yes_no.index(smoker), "Stroke": yes_no.index(stroke), "HeartDiseaseorAttack": yes_no.index(heart),
        "PhysActivity": yes_no.index(active), "GenHlth": HEALTH_LABELS.index(gen) + 1, "MentHlth": int(ment),
        "PhysHlth": int(phys), "DiffWalk": yes_no.index(diff_walk), "Sex": int(sex == "Male"),
        "Age": AGE_LABELS.index(age) + 1,
    }])[columns]

    risk = float(model.predict_proba(person)[0, 1])
    flagged = risk >= threshold

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Model risk score", f"{risk:.2f}", help="Higher = more likely to have diabetes/prediabetes. "
              "Because of class weighting this is a ranking score, not a calibrated probability.")
    m2.metric("Screening threshold (78% recall)", f"{threshold:.3f}")
    m3.metric("Result", "Flag for a blood test" if flagged else "Not flagged")
    if flagged:
        callout(f"<span style='color:{DARK_GOLD}; font-weight:800; font-size:1.05rem'>Flagged:</span> this person would be "
                "invited for a blood test.", GOLD, GOLD_TINT)
    else:
        callout(f"<span style='color:{FOREST_3}; font-weight:800; font-size:1.05rem'>Not flagged:</span> this person's risk is "
                "below the screening threshold.", FOREST_3, SAGE_TINT)

    st.subheader("Why? (LIME)")
    st.caption("LIME changes this person's answers thousands of times, watches how the risk score moves, "
               "and fits a simple model to that. Positive = pushes the risk up, negative = pushes it down.")
    with st.spinner("Explaining..."):
        lime_table = explain(model, explainer, columns, person)
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(lime_table.style.format({"Effect on risk": "{:+.3f}"}), hide_index=True,
                     width="stretch")
    with right:
        chart_data = lime_table.assign(
            Direction=np.where(lime_table["Effect on risk"] > 0, "Raises risk", "Lowers risk"),
            Label=lime_table["Effect on risk"].map(lambda w: f"{w:+.2f}"))
        order = list(chart_data.sort_values("Effect on risk", ascending=False)["Answer"])   # biggest push up on top
        pad = 0.07                                                     # room for the value labels at both ends
        x_domain = [min(chart_data["Effect on risk"].min(), 0) - pad, max(chart_data["Effect on risk"].max(), 0) + pad]
        base = alt.Chart(chart_data).encode(
            y=alt.Y("Answer:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)))
        bars = base.mark_bar(cornerRadius=3, stroke=DARK_GOLD, strokeOpacity=0.0).encode(
            x=alt.X("Effect on risk:Q", title="← lowers risk      |      raises risk →",
                    scale=alt.Scale(domain=x_domain, nice=False)),
            color=alt.Color("Direction:N", legend=None,
                            scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[GOLD, FOREST_3])),
            strokeOpacity=alt.condition("datum['Effect on risk'] > 0", alt.value(1.0), alt.value(0.0)),
            tooltip=["Answer", "Value", alt.Tooltip("Effect on risk:Q", format="+.3f")])
        labels = base.mark_text(dx=alt.ExprRef("datum['Effect on risk'] > 0 ? 18 : -18"),
                                color=FOREST_DARK, fontSize=12).encode(
            x="Effect on risk:Q", text="Label:N")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=FOREST_DARK).encode(x="x:Q")
        chart = ((bars + labels + zero)
                 .properties(height=320, background="transparent")
                 .configure_axis(labelColor=FOREST_DARK, titleColor=FOREST_3, gridColor=LINE_TINT,
                                 domainColor=LINE_TINT, tickColor=LINE_TINT, labelFontSize=12)
                 .configure_view(stroke=None))
        st.altair_chart(chart, width="stretch")

with tab_model:
    s = test_scores
    st.subheader(f"Final model on {s['Test people']:,} unseen test people")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall (diabetics caught)", f"{s['Recall']:.1%}")
    k2.metric("Precision (flagged who have it)", f"{s['Precision']:.1%}")
    k3.metric("ROC-AUC", f"{s['ROC-AUC']:.3f}", help="Random guessing = 0.5")
    k4.metric("PR-AUC", f"{s['PR-AUC']:.3f}", help=f"Random guessing ≈ {base_rate:.2f}")
    st.table(pd.DataFrame({"People": [s["Caught"], s["Missed"], s["False alarms"]]},
                          index=["Diabetics caught", "Diabetics missed", "False alarms"]))
    per_1000 = 1000 / s["Test people"]
    callout(f"<b style='color:{FOREST_2}'>Per 1,000 people screened:</b> about {(s['Caught'] + s['False alarms']) * per_1000:.0f} flagged "
            f"({s['Caught'] * per_1000:.0f} really have diabetes/prediabetes, "
            f"{s['False alarms'] * per_1000:.0f} false alarms) and {s['Missed'] * per_1000:.0f} missed. "
            f"About 1 in {1 / s['Precision']:.1f} flagged people has the condition, against 1 in "
            f"{1 / base_rate:.0f} without screening.", GOLD, "#FFFFFF")

with tab_about:
    st.markdown("""
**Data:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, via Alex Teboul on Kaggle: 253,680 US adults,
self-reported answers. 13.9% have diabetes or prediabetes.

**Model:** LightGBM (gradient-boosted trees) on 13 answers. Income, education, healthcare access, cholesterol check,
fruit, vegetables and heavy alcohol use are left out (fairness, and avoiding misleading patterns). Monotonic rules make
sure a worse risk factor can never lower the risk. The threshold is set so the model catches 78% of people with
diabetes in cross-validation, because a missed case costs far more than a false alarm (one blood test).

**Limits:** self-reported survey, US adults in 2015, no blood tests, diabetes and prediabetes combined.
**This is a learning project, not a medical tool.**
""")
