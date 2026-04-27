import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DeliverIQ — Delivery Time Predictor",
    page_icon="🛵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Mulish:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Mulish', sans-serif; }

/* ── chameleon animated background ── */
@keyframes chameleon-bg {
    0%   { background-position: 0% 50%;   }
    25%  { background-position: 50% 100%; }
    50%  { background-position: 100% 50%; }
    75%  { background-position: 50% 0%;   }
    100% { background-position: 0% 50%;   }
}

.stApp {
    background: linear-gradient(
        270deg,
        #0f0c29, #302b63, #24243e,
        #0f2027, #203a43, #2c5364,
        #1a1a2e, #16213e, #0f3460,
        #533483, #e94560, #0f3460
    );
    background-size: 400% 400%;
    animation: chameleon-bg 14s ease infinite;
    min-height: 100vh;
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 820px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── hero ── */
.hero {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 28px;
    padding: 2.8rem 2rem 2.4rem;
    text-align: center;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: rgba(233,69,96,0.08);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 200px; height: 200px;
    background: rgba(83,52,131,0.10);
    border-radius: 50%;
}

/* chameleon shimmer title */
@keyframes title-shift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg,
        #ff6b6b, #feca57, #48dbfb,
        #ff9ff3, #54a0ff, #5f27cd,
        #ff6b6b);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: title-shift 5s linear infinite;
    line-height: 1.15;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    color: rgba(220,220,255,0.85);
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    font-family: 'Syne', sans-serif;
}
.hero-sub {
    color: rgba(200,200,240,0.65);
    font-size: 0.92rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-pills {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    color: rgba(210,210,255,0.75);
    font-size: 0.63rem;
    letter-spacing: 1.5px;
    padding: 0.22rem 0.75rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    text-transform: uppercase;
}
.pill.best {
    background: rgba(233,69,96,0.18);
    border-color: rgba(233,69,96,0.45);
    color: #ff6b6b;
}

/* ── glass card ── */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 1.8rem 2rem 1.4rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(16px);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
    border-color: rgba(255,255,255,0.22);
    box-shadow: 0 8px 40px rgba(0,0,0,0.25);
}

/* ── section label ── */
.sec-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: title-shift 4s linear infinite;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.2), transparent);
    display: inline-block;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── widget overrides ── */
label[data-testid="stWidgetLabel"] p {
    color: rgba(220,220,255,0.90) !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid rgba(255,255,255,0.16) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}
div[data-testid="stSlider"] { padding: 0.2rem 0; }
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1.5px solid rgba(255,255,255,0.16) !important;
    border-radius: 10px !important;
    color: #e0e0ff !important;
}
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label span p,
div[data-testid="stRadio"] label div,
div[data-testid="stRadio"] p {
    color: rgba(220,220,255,0.90) !important;
    opacity: 1 !important;
}

/* ── predict button ── */
@keyframes btn-shift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
div[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(90deg,
        #e94560, #533483, #0f3460,
        #48dbfb, #ff9ff3, #e94560);
    background-size: 300% auto;
    animation: btn-shift 4s linear infinite;
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 2rem;
    cursor: pointer;
    box-shadow: 0 6px 30px rgba(233,69,96,0.35);
    transition: transform 0.3s, box-shadow 0.3s;
    margin-top: 0.8rem;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(233,69,96,0.50);
}

/* ── result card ── */
.result-wrap {
    margin-top: 2rem;
    animation: riseUp 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes riseUp {
    from { opacity:0; transform: translateY(30px) scale(0.95); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}

@keyframes result-border {
    0%   { border-color: #e94560; box-shadow: 0 0 30px rgba(233,69,96,0.3); }
    25%  { border-color: #533483; box-shadow: 0 0 30px rgba(83,52,131,0.3); }
    50%  { border-color: #48dbfb; box-shadow: 0 0 30px rgba(72,219,251,0.3); }
    75%  { border-color: #feca57; box-shadow: 0 0 30px rgba(254,202,87,0.3); }
    100% { border-color: #e94560; box-shadow: 0 0 30px rgba(233,69,96,0.3); }
}
.result-box {
    background: rgba(10,10,30,0.75);
    border: 2px solid #e94560;
    border-radius: 24px;
    padding: 2.4rem 2rem;
    text-align: center;
    backdrop-filter: blur(20px);
    animation: result-border 4s linear infinite;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.6rem; }
.result-label {
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: rgba(200,200,240,0.5);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    font-family: 'Syne', sans-serif;
}
.result-time {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(90deg,
        #ff6b6b, #feca57, #48dbfb, #ff9ff3, #ff6b6b);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: title-shift 4s linear infinite;
    line-height: 1;
}
.result-unit {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    color: rgba(200,200,240,0.5);
    letter-spacing: 2px;
    margin-top: 0.2rem;
    text-transform: uppercase;
}
.result-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 0.3rem 1.1rem;
    border-radius: 50px;
    margin-top: 1rem;
    text-transform: uppercase;
    font-family: 'Syne', sans-serif;
    border: 1px solid;
}

/* metrics */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1.4rem;
}
.metric-tile {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 0.85rem 0.6rem;
    text-align: center;
}
.mt-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #fff;
}
.mt-lbl {
    font-size: 0.6rem;
    color: rgba(200,200,240,0.45);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* tips */
.tips-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    backdrop-filter: blur(10px);
}
.tips-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #feca57;
    margin-bottom: 0.6rem;
}
.tips-text { font-size: 0.83rem; color: rgba(210,210,240,0.75); line-height: 1.65; }

.footer {
    text-align: center;
    color: rgba(180,180,220,0.30);
    font-size: 0.72rem;
    margin-top: 2.5rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

try:
    model  = load_model()
    loaded = True
except Exception as e:
    loaded = False
    load_err = str(e)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Random Forest · GridSearchCV Tuned</div>
  <div class="hero-title">🛵 DeliverIQ</div>
  <div class="hero-sub">Enter the delivery details below and get an instant AI-powered estimated delivery time.</div>
  <div class="hero-pills">
    <span class="pill">Decision Tree — Tested</span>
    <span class="pill best">✦ Random Forest — Best Model</span>
    <span class="pill">GridSearchCV — Tuned</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.error(f"Could not load model. Place `rf_model.pkl` in the same folder.\n\n{load_err}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Delivery Info
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">📦 Delivery Information <span class="sec-label-line"></span></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    distance   = st.slider("Distance (km)", 0.5, 25.0, 5.0, 0.1,
                            help="Total delivery distance in kilometres")
with col2:
    prep_time  = st.slider("Preparation Time (min)", 1, 60, 15,
                            help="Time taken to prepare the order")

courier_exp = st.slider("Courier Experience (years)", 0.0, 10.0, 2.0, 0.5,
                         help="Years of experience of the delivery courier")
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Conditions
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">🌦️ Conditions <span class="sec-label-line"></span></div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    weather = st.selectbox("Weather", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
with col4:
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])

col5, col6 = st.columns(2)
with col5:
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
with col6:
    vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_features():
    vec = {
        'Distance_km':            distance,
        'Preparation_Time_min':   prep_time,
        'Courier_Experience_yrs': courier_exp,
        'Weather_Clear':          1 if weather == "Clear"  else 0,
        'Weather_Foggy':          1 if weather == "Foggy"  else 0,
        'Weather_Rainy':          1 if weather == "Rainy"  else 0,
        'Weather_Snowy':          1 if weather == "Snowy"  else 0,
        'Weather_Windy':          1 if weather == "Windy"  else 0,
        'Traffic_Level_High':     1 if traffic == "High"   else 0,
        'Traffic_Level_Low':      1 if traffic == "Low"    else 0,
        'Traffic_Level_Medium':   1 if traffic == "Medium" else 0,
        'Time_of_Day_Afternoon':  1 if time_of_day == "Afternoon" else 0,
        'Time_of_Day_Evening':    1 if time_of_day == "Evening"   else 0,
        'Time_of_Day_Morning':    1 if time_of_day == "Morning"   else 0,
        'Time_of_Day_Night':      1 if time_of_day == "Night"     else 0,
        'Vehicle_Type_Bike':      1 if vehicle == "Bike"    else 0,
        'Vehicle_Type_Car':       1 if vehicle == "Car"     else 0,
        'Vehicle_Type_Scooter':   1 if vehicle == "Scooter" else 0,
    }
    ordered = model.feature_names_in_
    return np.array([[vec[f] for f in ordered]])

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if st.button("⚡  Predict Delivery Time"):
    with st.spinner("Calculating estimated time..."):
        try:
            X            = build_features()
            delivery_min = float(model.predict(X)[0])
            delivery_min = max(delivery_min, 1)
            hrs          = int(delivery_min // 60)
            mins         = int(round(delivery_min % 60))

            # Speed category
            if delivery_min <= 20:
                tag, tag_color = "Express Delivery ⚡", "#48dbfb"
            elif delivery_min <= 40:
                tag, tag_color = "Standard Delivery 🛵", "#feca57"
            elif delivery_min <= 60:
                tag, tag_color = "Delayed Delivery ⏳", "#ff9f43"
            else:
                tag, tag_color = "Slow Delivery 🐢",    "#ff6b6b"

            time_display = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}"
            unit_display = "minutes" if hrs == 0 else ""

            st.markdown(f"""
            <div class="result-wrap">
              <div class="result-box">
                <div class="result-icon">🛵</div>
                <div class="result-label">Estimated Delivery Time</div>
                <div class="result-time">{time_display}</div>
                <div class="result-unit">{unit_display}</div>
                <div class="result-tag" style="color:{tag_color}; border-color:{tag_color}55;">
                  {tag}
                </div>
                <div class="metrics-grid">
                  <div class="metric-tile">
                    <div class="mt-val">{distance} km</div>
                    <div class="mt-lbl">Distance</div>
                  </div>
                  <div class="metric-tile">
                    <div class="mt-val">{prep_time} min</div>
                    <div class="mt-lbl">Prep Time</div>
                  </div>
                  <div class="metric-tile">
                    <div class="mt-val">{courier_exp} yrs</div>
                    <div class="mt-lbl">Experience</div>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Contextual tip
            if traffic == "High" and weather in ["Rainy", "Snowy"]:
                tip = "Heavy traffic combined with bad weather is significantly increasing delivery time. If possible, ordering during off-peak hours or in clear weather can cut this estimate by 30-40%."
            elif delivery_min > 60:
                tip = "Long distance and high preparation time are the main factors here. A more experienced courier or lighter traffic could reduce this noticeably."
            elif delivery_min <= 20:
                tip = "Great conditions — short distance, clear weather, and low traffic are all working in your favour. This is an optimal delivery scenario."
            else:
                tip = f"The {traffic.lower()} traffic level and {weather.lower()} weather are the primary factors in this estimate. Courier experience of {courier_exp} years helps offset some delay."

            st.markdown(f"""
            <div class="tips-card">
              <div class="tips-title">Delivery Insight</div>
              <div class="tips-text">{tip}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("""
<div class="footer">
  DeliverIQ &nbsp;·&nbsp; Random Forest Regressor · GridSearchCV Tuned &nbsp;·&nbsp; Built with Streamlit<br>
  Models tested: Decision Tree · Random Forest (selected) · Hyperparameters tuned via GridSearchCV
</div>
""", unsafe_allow_html=True)