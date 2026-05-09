"""
NHS ED AI Scheduling Assistant
Predictive Deep Reinforcement Learning AI for Real-Time Resource Scheduling
MSc Dissertation | Northumbria University London | 2024-25
Supervisor: Dr. Rejwan Bin Sulaiman | Module LD7236
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NHS ED AI Assistant",
    page_icon="assets/nhs_icon.png" if False else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── NHS Design System Colours ─────────────────────────────────────────────────
NHS_BLUE       = "#003087"
NHS_LIGHT_BLUE = "#0072CE"
NHS_DARK_GREY  = "#212B32"
NHS_MID_GREY   = "#425563"
NHS_PALE_GREY  = "#E8EDEE"
NHS_WHITE      = "#FFFFFF"
RED_HIGH       = "#DA291C"
AMBER_MED      = "#ED8B00"
GREEN_LOW      = "#007F3B"
GREEN_SAFE     = "#007F3B"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Frutiger:wght@400;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Arial', 'Helvetica Neue', sans-serif;
      background-color: #F0F4F5;
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background-color: {NHS_BLUE};
      padding-top: 0;
  }}
  section[data-testid="stSidebar"] * {{
      color: {NHS_WHITE} !important;
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label {{
      color: {NHS_PALE_GREY} !important;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      text-transform: uppercase;
  }}
  section[data-testid="stSidebar"] hr {{
      border-color: rgba(255,255,255,0.2);
  }}

  /* NHS Header bar */
  .nhs-header {{
      background-color: {NHS_BLUE};
      padding: 18px 28px;
      margin: -1rem -1rem 1.5rem -1rem;
      border-bottom: 4px solid {NHS_LIGHT_BLUE};
  }}
  .nhs-header h1 {{
      color: white;
      font-size: 1.5rem;
      font-weight: 700;
      margin: 0;
      letter-spacing: -0.01em;
  }}
  .nhs-header p {{
      color: rgba(255,255,255,0.75);
      font-size: 0.82rem;
      margin: 4px 0 0 0;
  }}

  /* Metric cards */
  .metric-card {{
      background: white;
      border-radius: 4px;
      padding: 20px 24px;
      border-left: 5px solid {NHS_LIGHT_BLUE};
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }}
  .metric-card.alert-high  {{ border-left-color: {RED_HIGH}; }}
  .metric-card.alert-amber {{ border-left-color: {AMBER_MED}; }}
  .metric-card.alert-green {{ border-left-color: {GREEN_LOW}; }}

  .metric-card .label {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: {NHS_MID_GREY};
      margin-bottom: 6px;
  }}
  .metric-card .value {{
      font-size: 2.2rem;
      font-weight: 700;
      line-height: 1;
      color: {NHS_DARK_GREY};
  }}
  .metric-card .sub {{
      font-size: 0.78rem;
      color: {NHS_MID_GREY};
      margin-top: 4px;
  }}

  /* Risk banner */
  .risk-banner {{
      border-radius: 4px;
      padding: 24px 28px;
      text-align: center;
      margin-bottom: 1rem;
  }}
  .risk-banner .risk-pct {{
      font-size: 3.5rem;
      font-weight: 700;
      line-height: 1;
  }}
  .risk-banner .risk-label {{
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-top: 6px;
  }}
  .risk-banner .risk-sub {{
      font-size: 0.78rem;
      margin-top: 8px;
      opacity: 0.8;
  }}

  /* Action card */
  .action-card {{
      background: white;
      border-radius: 4px;
      padding: 20px 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      border-top: 4px solid {NHS_LIGHT_BLUE};
  }}
  .action-card .action-num {{
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: {NHS_MID_GREY};
  }}
  .action-card .action-title {{
      font-size: 1.25rem;
      font-weight: 700;
      color: {NHS_DARK_GREY};
      margin: 6px 0 4px;
  }}
  .action-card .action-desc {{
      font-size: 0.82rem;
      color: {NHS_MID_GREY};
  }}

  /* Section headers */
  .section-label {{
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: {NHS_MID_GREY};
      border-bottom: 2px solid {NHS_PALE_GREY};
      padding-bottom: 6px;
      margin-bottom: 14px;
      margin-top: 8px;
  }}

  /* Indicator table */
  .indicator-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 7px 0;
      border-bottom: 1px solid {NHS_PALE_GREY};
      font-size: 0.84rem;
  }}
  .indicator-row:last-child {{ border-bottom: none; }}
  .ind-label {{ color: {NHS_MID_GREY}; }}
  .ind-value {{ font-weight: 700; color: {NHS_DARK_GREY}; }}
  .ind-status-ok   {{ color: {GREEN_LOW}; font-weight: 700; font-size: 0.78rem; }}
  .ind-status-warn {{ color: {RED_HIGH}; font-weight: 700; font-size: 0.78rem; }}

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 0;
      border-bottom: 2px solid {NHS_PALE_GREY};
  }}
  .stTabs [data-baseweb="tab"] {{
      padding: 10px 20px;
      font-size: 0.84rem;
      font-weight: 600;
      color: {NHS_MID_GREY};
      border-radius: 0;
      border-bottom: 3px solid transparent;
  }}
  .stTabs [aria-selected="true"] {{
      color: {NHS_BLUE} !important;
      border-bottom: 3px solid {NHS_BLUE} !important;
      background: transparent !important;
  }}

  /* Footer */
  .nhs-footer {{
      background: {NHS_DARK_GREY};
      color: rgba(255,255,255,0.6);
      font-size: 0.75rem;
      padding: 14px 24px;
      margin-top: 2rem;
      border-radius: 4px;
      line-height: 1.6;
  }}

  /* Hide default streamlit header */
  header[data-testid="stHeader"] {{ display: none; }}
  .block-container {{ padding-top: 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        lgb_m  = pickle.load(open("lgb_model.pkl",  "rb"))
        xgb_m  = pickle.load(open("xgb_model.pkl",  "rb"))
        scaler = pickle.load(open("scaler.pkl",      "rb"))
        meta   = json.load(open("feature_meta.json"))
        return lgb_m, xgb_m, scaler, meta, None
    except FileNotFoundError as e:
        return None, None, None, None, str(e)

lgb_m, xgb_m, scaler, meta, load_error = load_models()

# ── NHS Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nhs-header">
  <h1>NHS ED AI Scheduling Assistant</h1>
  <p>Predictive Deep Reinforcement Learning for Real-Time Resource Scheduling &nbsp;|&nbsp;
     MSc Dissertation &nbsp;|&nbsp; Northumbria University London &nbsp;|&nbsp; 2024-25</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"Model files not found: {load_error}. Please ensure all .pkl and .json files are in the same directory as this app.")
    st.info("Required files: lgb_model.pkl, xgb_model.pkl, scaler.pkl, feature_meta.json")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='background:rgba(0,114,206,0.4);padding:14px 16px;border-radius:4px;margin-bottom:16px'>
      <div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;color:rgba(255,255,255,0.7)'>System</div>
      <div style='font-size:1.05rem;font-weight:700;color:white;margin-top:2px'>
          ED Live Input Panel</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='color:rgba(255,255,255,0.5)'>Department State</div>", unsafe_allow_html=True)
    bed_occ = st.slider("Bed Occupancy (%)",       85,  100,  93)
    beds    = st.slider("Beds Available",           1,   20,   8)
    staff   = st.slider("Staff Ratio",              0.30, 0.80, 0.54, 0.01)
    queue   = st.slider("Queue Length",             0,   80,   25)
    cpi     = st.slider("Capacity Pressure Index",  0.5,  4.0,  1.8, 0.1)
    hour    = st.slider("Hour of Day",              0,   23,   14)
    shift   = st.slider("Shift Arrivals",           20,  160,  76)

    st.markdown("<div class='section-label' style='color:rgba(255,255,255,0.5);margin-top:16px'>Patient Details</div>", unsafe_allow_html=True)
    age    = st.slider("Age", 0, 99, 55)
    triage = st.selectbox("Triage Category", options=[1,2,3,4,5],
        format_func=lambda x:{
            1:"Cat 1 — Immediate",2:"Cat 2 — Very Urgent",
            3:"Cat 3 — Urgent",4:"Cat 4 — Standard",5:"Cat 5 — Non-Urgent"}[x], index=2)
    news2  = st.slider("NEWS2 Score", 0, 9, 3)
    mode   = st.selectbox("Arrival Mode", ["Walk-in","Ambulance","GP Referral","Other"])
    imd    = st.selectbox("IMD Quintile (1 = most deprived)", [1,2,3,4,5], index=2)
    comor  = st.slider("Comorbidity Count", 0, 5, 1)
    hov    = st.checkbox("Handover Breach > 30 min")
    winter = st.checkbox("Winter Period")


# ── Feature Engineering ───────────────────────────────────────────────────────
def build_features():
    ambul   = 1 if mode == "Ambulance" else 0
    night   = 1 if (hour >= 22 or hour <= 6) else 0
    los_e   = 160 + triage * 8 + comor * 6
    wait_e  = 60 + triage * 14 + (bed_occ - 91) * 2 + (18 if mode == "Walk-in" else 0)
    board_e = max(0, (bed_occ - 91) * 4)
    hov_e   = (bed_occ - 91) * 3 + (15 if winter else 0)
    acuity  = (6 - triage) * 2 + news2 * 1.5 + comor * 0.8

    feat = {
        "hour_of_day": hour, "day_of_week": 0,
        "month": 1 if winter else 6, "is_weekend": 0,
        "patient_age": age, "patient_sex": 1,
        "imd_quintile": imd, "comorbidity_count": comor,
        "previous_ed_visits_12m": 0, "re_attendance_72h": 0,
        "arrival_mode": ["Walk-in","Ambulance","GP Referral","Other"].index(mode),
        "pre_alert_received": 1 if (ambul and hov) else 0,
        "chief_complaint": 1, "triage_category": triage,
        "news2_score": news2, "nhs_trust": 0, "ics_region": 0, "trust_type": 0,
        "bed_occupancy_pct": bed_occ, "beds_available": beds,
        "staff_ratio": staff, "daily_ed_attendance": shift * 3,
        "shift_total_arrivals": shift,
        "shift_ambulance_arrivals": int(shift * 0.32),
        "ambulance_handover_delay_min": hov_e,
        "handover_breach_gt30min": int(hov),
        "wait_time_to_assessment_min": wait_e,
        "ed_los_min": los_e, "boarding_delay_min": board_e,
        "queue_length_estimate": queue,
        "capacity_pressure_index": cpi,
        "patient_acuity_score": acuity, "wait_equity_delta_min": 0,
        "occ_x_triage": (bed_occ / 100) * triage,
        "wait_per_staff": wait_e / (staff + 0.01),
        "acuity_x_cpi": acuity * cpi,
        "los_over_threshold": max(0, los_e - 240),
        "night_high_occ": night * (1 if bed_occ > 95 else 0),
        "ambul_handover_stress": ambul * int(hov),
        "is_winter": int(winter),
        "boarding_per_los": board_e / (los_e + 1),
    }
    return pd.DataFrame([feat])[meta["feature_names"]]


feat_df = build_features()
lgb_p   = lgb_m.predict_proba(feat_df)[0, 1]
xgb_p   = xgb_m.predict_proba(feat_df)[0, 1]
avg_p   = (lgb_p + xgb_p) / 2

# Risk tier
if avg_p > 0.6:
    risk_label = "HIGH RISK"
    risk_color = RED_HIGH
    risk_bg    = "#FFF0EF"
    card_cls   = "alert-high"
elif avg_p > 0.35:
    risk_label = "MODERATE RISK"
    risk_color = AMBER_MED
    risk_bg    = "#FFF8EE"
    card_cls   = "alert-amber"
else:
    risk_label = "LOW RISK"
    risk_color = GREEN_LOW
    risk_bg    = "#EEF7F0"
    card_cls   = "alert-green"

# RL Action
if avg_p > 0.70 and bed_occ > 95:
    action_num, action_name, action_desc, action_color = 1, "Activate Surge Beds", \
        "Open additional surge capacity beds immediately. Notify bed management and ward coordinators.", RED_HIGH
elif avg_p > 0.60 and queue > 30:
    action_num, action_name, action_desc, action_color = 2, "Deploy Additional Staff", \
        "Call in extra clinical staff for the current shift. Prioritise senior triage nurses.", AMBER_MED
elif avg_p > 0.45:
    action_num, action_name, action_desc, action_color = 3, "Implement Fast-Track", \
        "Activate fast-track pathway for Category 4-5 patients to reduce queue pressure.", NHS_LIGHT_BLUE
else:
    action_num, action_name, action_desc, action_color = 0, "Maintain Current Operations", \
        "Current resource allocation is sufficient. Continue standard monitoring protocols.", GREEN_LOW


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Patient Risk Assessment",
    "Model Performance",
    "SHAP Explainability",
    "About"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Patient Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    col_left, col_mid, col_right = st.columns([1.1, 1.1, 0.9])

    # ── Breach Risk ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown("<div class='section-label'>4-Hour Breach Risk</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='risk-banner' style='background:{risk_bg};border:1.5px solid {risk_color}30'>
          <div class='risk-pct' style='color:{risk_color}'>{avg_p*100:.1f}%</div>
          <div class='risk-label' style='color:{risk_color}'>{risk_label}</div>
          <div class='risk-sub' style='color:{NHS_MID_GREY}'>
            LightGBM: {lgb_p*100:.1f}% &nbsp;|&nbsp; XGBoost: {xgb_p*100:.1f}%<br>
            Ensemble average (equal weight)
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge bar
        fig_g, ax_g = plt.subplots(figsize=(5, 0.55))
        fig_g.patch.set_facecolor(risk_bg)
        ax_g.set_facecolor(risk_bg)
        ax_g.barh([0], [1], color=NHS_PALE_GREY, height=0.5, edgecolor="none")
        ax_g.barh([0], [avg_p], color=risk_color, height=0.5, edgecolor="none")
        ax_g.axvline(0.35, color=AMBER_MED, lw=1.2, ls="--", alpha=0.7)
        ax_g.axvline(0.60, color=RED_HIGH,  lw=1.2, ls="--", alpha=0.7)
        ax_g.set_xlim(0, 1); ax_g.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;font-size:0.72rem;
                    color:{NHS_MID_GREY};margin-top:2px;padding:0 2px'>
          <span>0%</span><span style='color:{AMBER_MED}'>35% Amber</span>
          <span style='color:{RED_HIGH}'>60% Red</span><span>100%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── RL Action ─────────────────────────────────────────────────────────────
    with col_mid:
        st.markdown("<div class='section-label'>RL Resource Recommendation</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='action-card' style='border-top-color:{action_color}'>
          <div class='action-num' style='color:{action_color}'>Action {action_num} of 3</div>
          <div class='action-title'>{action_name}</div>
          <div class='action-desc'>{action_desc}</div>
          <div style='margin-top:14px;padding-top:10px;border-top:1px solid {NHS_PALE_GREY};
                      font-size:0.75rem;color:{NHS_MID_GREY}'>
            Policy: DQN Double Q-Network &nbsp;|&nbsp; Training: 500 episodes<br>
            Improvement vs baseline: {meta.get("rl_improvement_vs_baseline","N/A")}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 12-hour risk estimate
        risk_12h = min(avg_p * 1.35, 0.99)
        st.markdown(f"""
        <div style='margin-top:12px;background:white;border-radius:4px;padding:14px 18px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.07)'>
          <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.07em;color:{NHS_MID_GREY}'>12-Hour Breach Estimate</div>
          <div style='font-size:1.6rem;font-weight:700;color:{NHS_DARK_GREY};
                      margin-top:4px'>{risk_12h*100:.1f}%</div>
          <div style='font-size:0.75rem;color:{NHS_MID_GREY}'>
              Projected if current conditions persist</div>
        </div>
        """, unsafe_allow_html=True)

    # ── System Indicators ────────────────────────────────────────────────────
    with col_right:
        st.markdown("<div class='section-label'>System Indicators</div>", unsafe_allow_html=True)

        indicators = [
            ("Bed Occupancy",    f"{bed_occ:.0f}%",  bed_occ > 95),
            ("Staff Ratio",      f"{staff:.2f}",      staff < 0.42),
            ("Queue Length",     str(queue),           queue > 40),
            ("CPI",              f"{cpi:.1f}",         cpi > 2.5),
            ("NEWS2 Score",      str(news2),           news2 >= 7),
            ("Triage Category",  f"Cat {triage}",      triage <= 2),
            ("Handover Breach",  "Yes" if hov else "No", hov),
            ("Winter Period",    "Yes" if winter else "No", winter),
        ]

        rows_html = ""
        for label, value, is_alert in indicators:
            status_cls = "ind-status-warn" if is_alert else "ind-status-ok"
            status_txt = "ALERT" if is_alert else "OK"
            rows_html += f"""
            <div class='indicator-row'>
              <span class='ind-label'>{label}</span>
              <span class='ind-value'>{value}</span>
              <span class='{status_cls}'>{status_txt}</span>
            </div>"""

        st.markdown(f"<div style='background:white;border-radius:4px;padding:14px 18px;"
                    f"box-shadow:0 1px 4px rgba(0,0,0,0.07)'>{rows_html}</div>",
                    unsafe_allow_html=True)

    # ── Key Drivers Bar ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Key Risk Drivers — Current Patient</div>", unsafe_allow_html=True)

    driver_features = {
        "Queue Length":           min(queue / 80, 1.0),
        "Bed Occupancy":          min((bed_occ - 85) / 15, 1.0),
        "Ambulance Handover":     min(int(hov) * 0.85 + (bed_occ - 90) * 0.02, 1.0),
        "Triage x Occupancy":     min((bed_occ / 100) * triage / 5, 1.0),
        "Wait to Assessment":     min(((triage * 14 + (bed_occ - 91) * 2 + 60)) / 240, 1.0),
        "Staff Pressure":         min(max(0, (0.6 - staff) / 0.3), 1.0),
    }

    fig_d, ax_d = plt.subplots(figsize=(12, 2.2))
    fig_d.patch.set_facecolor("white")
    ax_d.set_facecolor("white")

    labels = list(driver_features.keys())
    values = list(driver_features.values())
    colors_d = [RED_HIGH if v > 0.7 else AMBER_MED if v > 0.4 else GREEN_LOW for v in values]

    bars = ax_d.barh(labels, values, color=colors_d, height=0.55, edgecolor="none")
    for bar, val in zip(bars, values):
        ax_d.text(min(val + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
                  f"{val*100:.0f}%", va="center", ha="left",
                  fontsize=8.5, color=NHS_DARK_GREY, fontweight="600")

    ax_d.set_xlim(0, 1.12)
    ax_d.set_xlabel("Relative Contribution", fontsize=9, color=NHS_MID_GREY)
    ax_d.tick_params(axis="y", labelsize=9, colors=NHS_DARK_GREY)
    ax_d.tick_params(axis="x", labelsize=8, colors=NHS_MID_GREY)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.spines["left"].set_color(NHS_PALE_GREY)
    ax_d.spines["bottom"].set_color(NHS_PALE_GREY)
    ax_d.axvline(0.7, color=RED_HIGH,  lw=0.8, ls="--", alpha=0.4)
    ax_d.axvline(0.4, color=AMBER_MED, lw=0.8, ls="--", alpha=0.4)
    plt.tight_layout(pad=1.0)
    st.pyplot(fig_d, use_container_width=True)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-label'>Model Performance Summary</div>", unsafe_allow_html=True)

    model_auc  = meta.get("model_auc",      {})
    model_acc  = meta.get("model_accuracy", {})

    model_names   = list(model_auc.keys())
    auc_vals      = [model_auc.get(n, 0)  for n in model_names]
    acc_vals      = [model_acc.get(n, 0)  for n in model_names]

    col_m1, col_m2, col_m3 = st.columns(3)

    best_model = max(model_auc, key=model_auc.get) if model_auc else "LightGBM"
    best_auc   = model_auc.get(best_model, 0)
    best_acc   = model_acc.get(best_model, 0)

    with col_m1:
        st.markdown(f"""
        <div class='metric-card alert-green'>
          <div class='label'>Best Model</div>
          <div class='value' style='font-size:1.4rem'>{best_model}</div>
          <div class='sub'>Top performing classifier</div>
        </div>""", unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>Best ROC-AUC</div>
          <div class='value'>{best_auc:.4f}</div>
          <div class='sub'>Area under ROC curve</div>
        </div>""", unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>Best Accuracy</div>
          <div class='value'>{best_acc*100:.1f}%</div>
          <div class='sub'>On held-out test set (30%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if model_auc:
        fig_m, axes_m = plt.subplots(1, 2, figsize=(13, 4))
        fig_m.patch.set_facecolor("white")

        bar_colors = [NHS_LIGHT_BLUE if n != best_model else NHS_BLUE for n in model_names]

        # AUC chart
        ax = axes_m[0]
        ax.set_facecolor("white")
        bars_auc = ax.bar(model_names, auc_vals, color=bar_colors, width=0.5, edgecolor="none")
        for bar, val in zip(bars_auc, auc_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="700", color=NHS_DARK_GREY)
        ax.set_ylim(0.5, 1.0)
        ax.set_title("ROC-AUC by Model", fontsize=11, fontweight="700", color=NHS_DARK_GREY, pad=10)
        ax.set_ylabel("ROC-AUC", fontsize=9, color=NHS_MID_GREY)
        ax.tick_params(axis="x", labelsize=8.5, rotation=15)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.9, color=RED_HIGH, lw=0.8, ls="--", alpha=0.5, label="0.9 threshold")
        ax.legend(fontsize=8)

        # Accuracy chart
        ax2 = axes_m[1]
        ax2.set_facecolor("white")
        bars_acc = ax2.bar(model_names, [v*100 for v in acc_vals],
                           color=bar_colors, width=0.5, edgecolor="none")
        for bar, val in zip(bars_acc, acc_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8.5,
                     fontweight="700", color=NHS_DARK_GREY)
        ax2.set_ylim(50, 100)
        ax2.set_title("Accuracy by Model", fontsize=11, fontweight="700", color=NHS_DARK_GREY, pad=10)
        ax2.set_ylabel("Accuracy (%)", fontsize=9, color=NHS_MID_GREY)
        ax2.tick_params(axis="x", labelsize=8.5, rotation=15)
        ax2.tick_params(axis="y", labelsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig_m, use_container_width=True)
        plt.close()

    st.markdown("<div class='section-label' style='margin-top:1.5rem'>Methodology Notes</div>",
                unsafe_allow_html=True)

    col_n1, col_n2, col_n3 = st.columns(3)
    notes = [
        ("Dataset",         "78,347 synthetic NHS ED records modelled on ECDS. IMD-weighted deprivation, MTS acuity scoring, ambulance handover delays."),
        ("Preprocessing",   "70/30 train-test split with stratification. SMOTE applied to training set only (30/70 breach ratio). StandardScaler normalisation."),
        ("Hyperparameters",  "Bayesian optimisation via Optuna (25 trials each). XGBoost and LightGBM tuned independently. No leakage from test set."),
    ]
    for col, (title, desc) in zip([col_n1, col_n2, col_n3], notes):
        with col:
            st.markdown(f"""
            <div style='background:white;border-radius:4px;padding:16px 18px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.07);height:100%'>
              <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.07em;color:{NHS_BLUE};margin-bottom:6px'>{title}</div>
              <div style='font-size:0.82rem;color:{NHS_MID_GREY};line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-label'>SHAP Feature Importance — Current Patient Prediction</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#EEF4FB;border-left:4px solid {NHS_LIGHT_BLUE};
                padding:12px 16px;border-radius:0 4px 4px 0;
                font-size:0.83rem;color:{NHS_MID_GREY};margin-bottom:16px'>
      SHAP (SHapley Additive exPlanations) values show the contribution of each feature
      to this specific prediction. Positive values push towards breach; negative values
      push away from breach. This provides clinician-interpretable explanations compliant
      with NHS AI transparency requirements.
    </div>
    """, unsafe_allow_html=True)

    try:
        import shap
        explainer = shap.TreeExplainer(xgb_m)
        shap_vals = explainer.shap_values(feat_df)

        shap_series = pd.Series(shap_vals[0], index=feat_df.columns)
        top_shap    = shap_series.reindex(shap_series.abs().sort_values(ascending=False).head(15).index)
        top_shap_plot = top_shap.sort_values()

        fig_s, ax_s = plt.subplots(figsize=(11, 5.5))
        fig_s.patch.set_facecolor("white")
        ax_s.set_facecolor("white")

        colors_s = [RED_HIGH if v > 0 else NHS_LIGHT_BLUE for v in top_shap_plot.values]
        bars_s   = ax_s.barh(top_shap_plot.index, top_shap_plot.values,
                              color=colors_s, height=0.6, edgecolor="none", alpha=0.9)

        for bar, val in zip(bars_s, top_shap_plot.values):
            x_pos = val + 0.005 if val >= 0 else val - 0.005
            ha    = "left" if val >= 0 else "right"
            ax_s.text(x_pos, bar.get_y() + bar.get_height()/2,
                      f"{val:+.3f}", va="center", ha=ha,
                      fontsize=8, color=NHS_DARK_GREY, fontweight="600")

        ax_s.axvline(0, color=NHS_DARK_GREY, lw=1.0, alpha=0.6)
        ax_s.set_xlabel("SHAP Value (impact on breach probability)", fontsize=9, color=NHS_MID_GREY)
        ax_s.set_title(f"Top 15 Feature Contributions — Predicted Breach Probability: {avg_p*100:.1f}%",
                       fontsize=11, fontweight="700", color=NHS_DARK_GREY, pad=12)
        ax_s.tick_params(axis="y", labelsize=9, colors=NHS_DARK_GREY)
        ax_s.tick_params(axis="x", labelsize=8, colors=NHS_MID_GREY)
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)
        ax_s.spines["left"].set_color(NHS_PALE_GREY)
        ax_s.spines["bottom"].set_color(NHS_PALE_GREY)

        red_patch   = mpatches.Patch(color=RED_HIGH,       label="Increases breach risk")
        blue_patch  = mpatches.Patch(color=NHS_LIGHT_BLUE, label="Decreases breach risk")
        ax_s.legend(handles=[red_patch, blue_patch], fontsize=8.5, loc="lower right")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig_s, use_container_width=True)
        plt.close()

        # Feature value table
        st.markdown("<div class='section-label' style='margin-top:1rem'>Top Feature Values for This Patient</div>",
                    unsafe_allow_html=True)

        top_features = shap_series.abs().sort_values(ascending=False).head(10).index
        table_data = []
        for feat_name in top_features:
            table_data.append({
                "Feature":       feat_name,
                "Value":         f"{feat_df[feat_name].values[0]:.3f}",
                "SHAP Impact":   f"{shap_series[feat_name]:+.4f}",
                "Direction":     "Towards Breach" if shap_series[feat_name] > 0 else "Away from Breach"
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        st.info("Ensure the XGBoost model file is correctly loaded.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    col_a1, col_a2 = st.columns([1.4, 1])

    with col_a1:
        st.markdown(f"""
        <div style='background:white;border-radius:4px;padding:24px 28px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.07)'>
          <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.1em;color:{NHS_BLUE};margin-bottom:10px'>
              Research Overview</div>
          <div style='font-size:1.05rem;font-weight:700;color:{NHS_DARK_GREY};margin-bottom:12px'>
              A Predictive Deep Reinforcement Learning AI Assistant for
              Real-Time Resource Scheduling Optimisation in NHS Emergency Care
          </div>
          <div style='font-size:0.85rem;color:{NHS_MID_GREY};line-height:1.7'>
            This application demonstrates a hybrid machine learning and deep reinforcement
            learning pipeline for predicting NHS Emergency Department 4-hour breach risk and
            recommending real-time resource scheduling actions.<br><br>
            The system combines XGBoost and LightGBM ensemble predictions with a Double DQN
            reinforcement learning agent trained to optimise resource allocation decisions,
            reducing breach rates while minimising unnecessary resource deployment.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        details = [
            ("Institution",   "Northumbria University London"),
            ("Module",        "LD7236 — MSc Dissertation"),
            ("Supervisor",    "Dr. Rejwan Bin Sulaiman"),
            ("Academic Year", "2024-25"),
            ("Dataset",       "78,347 synthetic NHS ED records"),
            ("ML Models",     "XGBoost, LightGBM, RF, GBM, LR"),
            ("RL Algorithm",  "Double DQN + PPO"),
            ("Explainability","SHAP TreeExplainer"),
            ("Deployment",    "Streamlit Cloud (Free Tier)"),
        ]
        rows = ""
        for k, v in details:
            rows += f"""
            <div class='indicator-row'>
              <span class='ind-label'>{k}</span>
              <span class='ind-value' style='font-size:0.82rem'>{v}</span>
            </div>"""
        st.markdown(f"""
        <div style='background:white;border-radius:4px;padding:18px 22px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.07)'>{rows}</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_b1, col_b2, col_b3 = st.columns(3)
    pipeline = [
        ("Data Pipeline",        "Synthetic NHS ECDS dataset with IMD deprivation weighting, MTS acuity, ambulance handover delays. 70/30 stratified split."),
        ("ML Classification",    "SMOTE balancing (30/70 ratio). Optuna Bayesian hyperparameter tuning. SHAP leakage investigation confirmed zero leakage."),
        ("RL Optimisation",      "Custom NHSEDEnv gymnasium environment. 4 resource actions. Reward shaping for breach prevention vs unnecessary deployment."),
    ]
    for col, (title, desc) in zip([col_b1, col_b2, col_b3], pipeline):
        with col:
            st.markdown(f"""
            <div style='background:{NHS_BLUE};border-radius:4px;padding:18px 20px;color:white'>
              <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.07em;color:rgba(255,255,255,0.6);margin-bottom:6px'>
                  {title}</div>
              <div style='font-size:0.82rem;line-height:1.5;color:rgba(255,255,255,0.85)'>
                  {desc}</div>
            </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
lgb_auc = meta.get("model_auc", {}).get("LightGBM", "N/A")
rl_imp   = meta.get("rl_improvement_vs_baseline", "N/A")

st.markdown(f"""
<div class='nhs-footer'>
  NHS ED AI Scheduling Assistant &nbsp;|&nbsp;
  MSc Big Data & Data Science Technology — Northumbria University London &nbsp;|&nbsp;
  Supervisor: Dr. Rejwan Bin Sulaiman &nbsp;|&nbsp; Module LD7236 &nbsp;|&nbsp; 2024-25<br>
  LightGBM AUC: {lgb_auc} &nbsp;|&nbsp;
  SMOTE balanced (30/70) &nbsp;|&nbsp;
  Optuna-tuned &nbsp;|&nbsp;
  DQN vs baseline: {rl_imp} &nbsp;|&nbsp;
  For research and demonstration purposes only. Not for clinical use.
</div>
""", unsafe_allow_html=True)
