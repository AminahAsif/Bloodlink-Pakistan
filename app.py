import os
import random
import warnings
from datetime import datetime, timedelta

import gradio as gr
import numpy as np
import pandas as pd
import xgboost as xgb
from geopy.distance import geodesic
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

#  CONFIG
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CITIES = [
    {"city": "Karachi", "lat": 24.8607, "lon": 67.0011, "urban": 1},
    {"city": "Lahore", "lat": 31.5497, "lon": 74.3436, "urban": 1},
    {"city": "Islamabad", "lat": 33.6844, "lon": 73.0479, "urban": 1},
    {"city": "Quetta", "lat": 30.1798, "lon": 66.9750, "urban": 0},
    {"city": "Peshawar", "lat": 34.0150, "lon": 71.5805, "urban": 0},
    {"city": "Multan", "lat": 30.1575, "lon": 71.5249, "urban": 0},
    {"city": "Faisalabad", "lat": 31.4180, "lon": 73.0791, "urban": 1},
]

COMPAT_MATRIX = {
    "O-": ["O-", "O+", "A+", "A-", "B+", "B-", "AB+", "AB-"],
    "O+": ["O+", "A+", "B+", "AB+"],
    "A-": ["A-", "A+", "AB+", "AB-"],
    "A+": ["A+", "AB+"],
    "B-": ["B-", "B+", "AB+", "AB-"],
    "B+": ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}

# = GROQ (key from environment only)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None
GROQ_ENABLED = False
if GROQ_API_KEY:
    try:
        from groq import Groq

        groq_client = Groq(api_key=GROQ_API_KEY)
        GROQ_ENABLED = True
    except Exception as e:
        print(f"Groq not available: {e}")
print("Groq:", "enabled" if GROQ_ENABLED else "fallback mode (no GROQ_API_KEY)")


#  DONOR DATABASE
def generate_realistic_donors(n=10000):
    donors = []
    for i in range(n):
        gender = "M" if random.random() < 0.65 else "F"
        age = int(np.clip(int(np.random.normal(33, 11)), 18, 60))
        rnd = random.random()
        if rnd < 0.34:
            blood_type = "O+"
        elif rnd < 0.59:
            blood_type = "B+"
        elif rnd < 0.84:
            blood_type = "A+"
        else:
            blood_type = random.choice(["AB+", "O-", "B-", "A-", "AB-"])

        c = random.choice(CITIES)
        donation_history = max(0, int(np.random.gamma(2.5, 1.5)))
        days_since_last = (
            np.random.randint(7, 120) if donation_history > 0 else random.randint(150, 400)
        )
        preferred_contact = random.choices(["App", "SMS", "Phone"], weights=[0.5, 0.35, 0.15])[0]
        health_status = random.choices(
            ["Healthy", "Minor issues", "Unhealthy"], weights=[0.75, 0.2, 0.05]
        )[0]
        true_response_prob = min(
            0.95,
            max(
                0.05,
                0.6 - days_since_last / 200 + donation_history * 0.05
                + (preferred_contact == "App") * 0.1 + (health_status == "Healthy") * 0.1,
            ),
        )
        donors.append(
            {
                "donor_id": f"DNR_{i:05d}",
                "age": age,
                "gender": gender,
                "blood_type": blood_type,
                "city": c["city"],
                "lat": c["lat"] + random.uniform(-0.08, 0.08),
                "lon": c["lon"] + random.uniform(-0.08, 0.08),
                "urban": c["urban"],
                "donation_history": donation_history,
                "days_since_last": days_since_last,
                "preferred_contact": preferred_contact,
                "health_status": health_status,
                "true_response_prob": true_response_prob,
            }
        )
    return pd.DataFrame(donors)


donor_db = generate_realistic_donors(10000)


#  RULE-BASED MATCHING 
def calculate_donor_score(donor_row, patient_lat, patient_lon, urgency="normal"):
    distance = geodesic((donor_row["lat"], donor_row["lon"]), (patient_lat, patient_lon)).km

    if distance < 2:
        distance_score = 40
    elif distance < 5:
        distance_score = 35
    elif distance < 10:
        distance_score = 25
    elif distance < 15:
        distance_score = 15
    else:
        distance_score = max(0, 10 - distance)

    h = donor_row["donation_history"]
    history_score = 25 if h >= 10 else 20 if h >= 5 else 15 if h >= 2 else 10 if h >= 1 else 5

    d = donor_row["days_since_last"]
    recency_score = 5 if d > 300 else 10 if d > 150 else 15 if d > 90 else 18 if d > 60 else 20

    contact_score = {"App": 8, "SMS": 5, "Phone": 3}[donor_row["preferred_contact"]]
    health_score = {"Healthy": 5, "Minor issues": 3, "Unhealthy": 0}[donor_row["health_status"]]
    urban_score = 2 if donor_row["urban"] == 1 else 0

    total = distance_score + history_score + recency_score + contact_score + health_score + urban_score
    if urgency == "emergency":
        total = total * 0.7 + distance_score * 0.5

    return total, {
        "distance": distance,
        "distance_score": distance_score,
        "history_score": history_score,
        "recency_score": recency_score,
        "contact_score": contact_score,
        "health_score": health_score,
        "urban_score": urban_score,
        "total_score": total,
    }


def find_best_donors(need_blood_type, patient_lat, patient_lon, top_n=20,
                     urgency="normal", max_distance_km=25):
    compatible = [bt for bt, rec in COMPAT_MATRIX.items() if need_blood_type in rec]
    c = donor_db[donor_db["blood_type"].isin(compatible)]
    # cheap bounding-box prefilter (keeps results identical, avoids needless geodesic calls)
    c = c[(c["lat"].sub(patient_lat).abs() < 0.4) & (c["lon"].sub(patient_lon).abs() < 0.5)].copy()
    if len(c) == 0:
        return pd.DataFrame()

    scores, dists = [], []
    for _, donor in c.iterrows():
        s, b = calculate_donor_score(donor, patient_lat, patient_lon, urgency)
        scores.append(s)
        dists.append(b["distance"])
    c["score"] = scores
    c["distance"] = dists
    c = c[c["distance"] <= max_distance_km]
    return c.sort_values("score", ascending=False).head(top_n)


# quick validation at startup (smaller than the notebook's 500 to keep boot fast)
def run_validation(n_requests=200):
    rng_rows = []
    for _ in range(n_requests):
        city = random.choice(CITIES)
        lat = city["lat"] + random.uniform(-0.02, 0.02)
        lon = city["lon"] + random.uniform(-0.02, 0.02)
        m = find_best_donors(random.choice(BLOOD_TYPES), lat, lon, top_n=10, urgency="emergency")
        for rank, (_, d) in enumerate(m.iterrows(), start=1):
            rng_rows.append(
                {
                    "rank": rank,
                    "score": d["score"],
                    "distance": d["distance"],
                    "actual_response": np.random.binomial(1, d["true_response_prob"]),
                }
            )
    return pd.DataFrame(rng_rows)


results_df = run_validation(200)
response_rate = results_df["actual_response"].mean() * 100
top1_rate = results_df[results_df["rank"] == 1]["actual_response"].mean() * 100


#  SHORTAGE FORECAST (XGBoost) 
def generate_shortage_data(days=1500):
    data = []
    end_date = datetime(2025, 12, 31)
    start_date = end_date - timedelta(days=days - 1)
    type_factor = {"O+": 6, "B+": 4, "A+": 3, "AB+": 0, "O-": -9, "B-": -7, "A-": -6, "AB-": -10}
    for day in range(days):
        d = start_date + timedelta(days=day)
        dow, month, doy = d.weekday(), d.month, d.timetuple().tm_yday
        weekend = dow >= 5
        for bt in BLOOD_TYPES:
            daily = (
                22 + type_factor[bt] + (7 if weekend else 0)
                + 5 * np.sin(2 * np.pi * doy / 365) + 2 * np.sin(2 * np.pi * dow / 7)
                + (8 if month in [3, 4] else 0) + np.random.normal(0, 1.2)
            )
            data.append(
                {
                    "date": d,
                    "blood_type": bt,
                    "daily_requests": max(5, min(50, round(daily))),
                    "day_of_week": dow,
                    "month": month,
                    "is_weekend": int(weekend),
                    "day_of_year": doy,
                }
            )
    return pd.DataFrame(data)


shortage_df = generate_shortage_data(1500)
FEATURES = ["day_of_week", "month", "is_weekend", "day_of_year",
            "lag_1", "lag_2", "lag_3", "rolling_mean_7"]

models, metrics = {}, {}
for bt in BLOOD_TYPES:
    t = shortage_df[shortage_df["blood_type"] == bt].copy()
    t["lag_1"] = t["daily_requests"].shift(1)
    t["lag_2"] = t["daily_requests"].shift(2)
    t["lag_3"] = t["daily_requests"].shift(3)
    t["rolling_mean_7"] = t["daily_requests"].rolling(7, min_periods=1).mean()
    t = t.dropna()
    X, y = t[FEATURES], t["daily_requests"]
    split = int(len(X) * 0.8)
    m = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=SEED)
    m.fit(X[:split], y[:split], verbose=False)
    pred = m.predict(X[split:])
    models[bt] = m
    metrics[bt] = {
        "mae": mean_absolute_error(y[split:], pred),
        "rmse": float(np.sqrt(mean_squared_error(y[split:], pred))),
        "r2": r2_score(y[split:], pred),
    }
avg_mae = np.mean([v["mae"] for v in metrics.values()])
avg_r2 = np.mean([v["r2"] for v in metrics.values()])
print(f"Models ready | avg R2={avg_r2:.3f} MAE={avg_mae:.2f} | response rate={response_rate:.1f}%")

#  ELIGIBILITY 
ELIGIBILITY_QUESTIONS = [
    {"q": "Are you between 18-60 years old?", "type": "yes no", "required": "yes", "weight": 1.0},
    {"q": "Do you weigh at least 50kg?", "type": "yes no", "required": "yes", "weight": 0.9},
    {"q": "Are you currently in good health (no fever, cold, or flu)?", "type": "yesno", "required": "yes", "weight": 1.0},
    {"q": "Have you donated blood in the last 90 days?", "type": "yesno", "required": "no", "weight": 1.0},
    {"q": "Do you have any chronic conditions (diabetes, hypertension, heart disease)?", "type": "yesno", "required": "no", "weight": 0.8},
    {"q": "Are you currently taking any medications?", "type": "text", "required": "none", "weight": 0.7},
    {"q": "Have you had any tattoos or piercings in the last 6 months?", "type": "yesno", "required": "no", "weight": 0.6},
    {"q": "Have you traveled to malaria-risk areas in the last 3 months?", "type": "yesno", "required": "no", "weight": 0.7},
    {"q": "What is your hemoglobin level (g/dL)? (Enter number or 'unknown')", "type": "number", "required": ">12", "weight": 0.8},
]


def evaluate_eligibility(answers):
    eligible, score, max_score, issues = True, 0.0, 0.0, []
    for i, qa in enumerate(ELIGIBILITY_QUESTIONS):
        ans = str(answers[i]).lower().strip()
        max_score += qa["weight"]
        if qa["type"] == "yesno":
            if qa["required"] == "yes" and ans != "yes":
                eligible = False
                issues.append(f"Question {i+1}: Required 'yes'")
            elif qa["required"] == "no" and ans == "yes":
                eligible = False
                issues.append(f"Question {i+1}: Should be 'no'")
            else:
                score += qa["weight"]
        elif qa["type"] == "number":
            try:
                if float(ans) < 12:
                    eligible = False
                    issues.append(f"Question {i+1}: Hemoglobin too low (<12)")
                else:
                    score += qa["weight"]
            except ValueError:
                if ans != "unknown":
                    score += qa["weight"] * 0.5
        else:
            score += qa["weight"] if ans in ["none", "no", "nothing"] else qa["weight"] * 0.5
    return eligible, (score / max_score) * 100, issues


#  CHATBOT 
CHATBOT_KNOWLEDGE = {
    "eligibility": "You can donate blood if you:\n- Are 18-60 years old\n- Weigh at least 50kg\n- Are in good health\n- Haven't donated in last 90 days\n- Are not pregnant/breastfeeding\n- No fever, cold, or antibiotics",
    "frequency": "Donation frequency:\n- Males: Every 56 days (8 weeks)\n- Females: Every 84 days (12 weeks)\nThis allows full blood cell recovery.",
    "ramadan": "You CAN donate during Ramadan:\n- Best after Iftar (evening)\n- Ensure hydration at Sehri\n- Avoid if feeling weak\nIslamic scholars permit it as life-saving.",
    "halal": "Yes, blood donation is HALAL in Islam.\nIt saves lives, which is highly encouraged.\nAll major Islamic scholars support it.",
    "process": "Donation process:\n1. Registration & screening (10 min)\n2. Donation (8-10 min)\n3. Rest & refreshments (10 min)\nTotal: ~30 minutes, minimal pain.",
    "benefits": "Benefits:\n- Saves up to 3 lives per donation\n- Free health screening\n- Earn reward points (JazzCash)",
    "covid": "After COVID-19:\n- Wait 14 days after symptoms end\n- Wait 7 days after vaccination\n- Must be fully recovered",
    "tattoo": "Wait 6 months after tattoo/piercing.\nThis prevents infection transmission.",
}

SYSTEM_PROMPT = """You are a helpful blood donation assistant for Pakistan.
You provide accurate information based on WHO and Red Cross guidelines.
Always emphasize that this is informational only and users should consult medical professionals.
Be culturally sensitive to Pakistani context (mention that blood donation is halal in Islam).
Keep responses concise (3-4 sentences max)."""


def chatbot_groq(question):
    if not question or not question.strip():
        return "Please type a question."
    if GROQ_ENABLED:
        try:
            r = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": question}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=300,
            )
            return (r.choices[0].message.content
                    + "\n\n⚠️ Disclaimer: This is informational only. All donations require full medical screening at certified centers.")
        except Exception as e:
            print(f"Groq API Error: {e}")

    q = question.lower()
    k = CHATBOT_KNOWLEDGE
    if any(w in q for w in ["eligible", "can i", "allowed", "qualify"]):
        return k["eligibility"]
    if any(w in q for w in ["how often", "frequency", "again", "wait"]):
        return k["frequency"]
    if any(w in q for w in ["ramadan", "fasting", "roza"]):
        return k["ramadan"]
    if any(w in q for w in ["halal", "haram", "islam"]):
        return k["halal"]
    if any(w in q for w in ["process", "procedure", "what happens"]):
        return k["process"]
    if any(w in q for w in ["benefit", "why", "advantage"]):
        return k["benefits"]
    if any(w in q for w in ["covid", "coronavirus", "vaccine"]):
        return k["covid"]
    if any(w in q for w in ["tattoo", "piercing"]):
        return k["tattoo"]
    return ("I can help with: eligibility, frequency, Ramadan, halal status, process, benefits, "
            "COVID-19, and tattoos. What would you like to know?\n\n⚠️ For medical advice, consult certified professionals.")


# GAMIFICATION / INVENTORY / PATIENTS 
GAMIFICATION_STATE = {}


def update_gamification(donor_id):
    s = GAMIFICATION_STATE.setdefault(donor_id, {"points": 0, "donations": 0, "badges": []})
    s["points"] += 50
    s["donations"] += 1
    for n, badge in [(1, "First Drop"), (5, "LifeSaver"), (10, "Hero")]:
        if s["donations"] == n and badge not in s["badges"]:
            s["badges"].append(badge)
    return s


def get_leaderboard(city=None):
    rows = []
    for did, st in GAMIFICATION_STATE.items():
        info = donor_db[donor_db["donor_id"] == did]
        dcity = info.iloc[0]["city"] if len(info) else "Unknown"
        if city is None or dcity == city:
            rows.append({"donor_id": did, "points": st["points"], "donations": st["donations"],
                         "badges": ", ".join(st["badges"]), "city": dcity})
    if not rows:
        return pd.DataFrame(columns=["donor_id", "points", "donations", "badges", "city"])
    return pd.DataFrame(rows).sort_values("points", ascending=False).head(10)


HOSPITAL_INVENTORY = {
    "Indus Hospital (Karachi)": {"O+": 5, "A+": 3, "B+": 2, "O-": 1, "A-": 1, "B-": 1, "AB+": 0, "AB-": 0},
    "Shaukat Khanum (Lahore)": {"O+": 4, "A+": 4, "B+": 3, "O-": 2, "A-": 1, "B-": 1, "AB+": 1, "AB-": 0},
    "Aga Khan Hospital (Karachi)": {"O+": 6, "A+": 2, "B+": 2, "O-": 1, "A-": 0, "B-": 1, "AB+": 1, "AB-": 0},
    "PIMS (Islamabad)": {"O+": 3, "A+": 2, "B+": 1, "O-": 0, "A-": 1, "B-": 0, "AB+": 0, "AB-": 0},
    "Bahawal Victoria Hospital (Bahawalpur)": {"O+": 4, "A+": 3, "B+": 2, "O-": 1, "A-": 1, "B-": 1, "AB+": 0, "AB-": 0},
}


def update_inventory(hospital, blood_type, units):
    if hospital in HOSPITAL_INVENTORY and blood_type:
        HOSPITAL_INVENTORY[hospital][blood_type] = int(units)
        return f"✅ Updated {hospital}: {blood_type} = {int(units)} units"
    return "❌ Select a hospital and blood type"


def get_inventory(city):
    rows = [[h] + [inv[bt] for bt in BLOOD_TYPES]
            for h, inv in HOSPITAL_INVENTORY.items() if city == "All" or city in h]
    return pd.DataFrame(rows, columns=["Hospital"] + BLOOD_TYPES)


EMERGENCY_PATIENTS = {
    "patient_001": {"name": "Aisha Khan", "age": 45, "gender": "F", "hospital": "Indus Hospital (Karachi)",
                    "blood_needed": "O+", "units_needed": 2, "urgency": "critical", "reason": "Road Accident",
                    "time_critical": "within 2 hours", "contact": "0300-1234567",
                    "lat": 24.8607 + 0.01, "lon": 67.0011 + 0.01},
    "patient_002": {"name": "Usman Tariq", "age": 62, "gender": "M", "hospital": "Shaukat Khanum (Lahore)",
                    "blood_needed": "A-", "units_needed": 3, "urgency": "high", "reason": "Surgery Complications",
                    "time_critical": "within 6 hours", "contact": "0321-9876543",
                    "lat": 31.5497 - 0.02, "lon": 74.3436 - 0.02},
}

DISCLAIMER = """
⚠️ DISCLAIMER: This is pre-screening only.
All donations require full medical screening at certified centers for:
- Infectious diseases (HIV, Hepatitis B/C, Malaria, Syphilis)
- Blood pressure, hemoglobin verification
- Complete health assessment
"""


#  HANDLERS
def handle_registration(name, age, gender, blood_type, city, weight, *answers):
    try:
        eligible, score, issues = evaluate_eligibility(list(answers))
        if not eligible:
            return "❌ INELIGIBLE for donation\n\nIssues:\n" + "\n".join(issues) + f"\n\n{DISCLAIMER}"
        donor_id = f"DNR_{len(donor_db):05d}"
        return f"""
✅ REGISTRATION SUCCESSFUL

Donor ID: {donor_id}
Name: {name}
Age: {age} | Gender: {gender} | Blood Type: {blood_type}
City: {city} | Weight: {weight}kg
Eligibility Score: {score:.1f}%

{DISCLAIMER}

Next Steps:
1. Visit nearest blood bank for verification
2. Complete medical screening
3. Donate and earn rewards!
"""
    except Exception as e:
        return f"❌ Error: {e}"


def list_patients():
    return pd.DataFrame(
        [{"ID": pid, "Name": i["name"], "Hospital": i["hospital"], "Blood": i["blood_needed"],
          "Units": i["units_needed"], "Urgency": i["urgency"]} for pid, i in EMERGENCY_PATIENTS.items()]
    )


def handle_sos(patient_id, blood_type, lat, lon):
    try:
        if patient_id and patient_id != "Custom" and patient_id in EMERGENCY_PATIENTS:
            p = EMERGENCY_PATIENTS[patient_id]
            blood_type, lat, lon = p["blood_needed"], p["lat"], p["lon"]
            info = f"""
🆘 EMERGENCY ALERT

Patient: {p['name']} (Age {p['age']}, {p['gender']})
Hospital: {p['hospital']}
Blood Required: {blood_type} ({p['units_needed']} units)
Urgency: {p['urgency']}
Reason: {p['reason']}
Time Critical: {p['time_critical']}
Contact: {p['contact']}

📍 Location: {lat:.4f}, {lon:.4f}
"""
        else:
            info = f"\n🆘 CUSTOM EMERGENCY REQUEST\n\nBlood Required: {blood_type}\n📍 Location: {lat:.4f}, {lon:.4f}\n"

        m = find_best_donors(blood_type, lat, lon, top_n=10, urgency="emergency")
        if len(m) == 0:
            return pd.DataFrame(), info + "\n\n❌ No compatible donors found in 25km radius\n⚠️ Contact blood banks directly!"

        df = m[["donor_id", "blood_type", "distance", "score", "city", "preferred_contact"]].copy()
        df["distance"] = df["distance"].round(2).astype(str) + " km"
        df["score"] = df["score"].round(1).astype(str) + "/100"
        df.columns = ["Donor ID", "Blood Type", "Distance", "Match Score", "City", "Contact"]

        backups = ", ".join(m["donor_id"].iloc[1:3].tolist()) or "n/a"
        compat = ", ".join(bt for bt, r in COMPAT_MATRIX.items() if blood_type in r)
        msg = info + f"""
---
✅ FOUND {len(m)} COMPATIBLE DONORS

🎯 Recommendations:
1. Contact {m.iloc[0]['donor_id']} first (Score: {m.iloc[0]['score']:.1f}/100)
2. Prepare backups: {backups}
3. Average distance: {m['distance'].mean():.2f} km

⚠️ CRITICAL REMINDERS:
- All donations require immediate medical screening
- Cross-match testing mandatory
- Blood type verification essential
- Contact hospital blood bank simultaneously

Compatible Blood Types (can donate to {blood_type}):
{compat}
"""
        return df, msg
    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {e}"


def predict_forecast():
    try:
        start = datetime.now() + timedelta(days=1)
        rows = []
        for day in range(7):
            d = start + timedelta(days=day)
            dow, month, doy = d.weekday(), d.month, d.timetuple().tm_yday
            preds = {}
            for bt in BLOOD_TYPES:
                hist = shortage_df[shortage_df["blood_type"] == bt]["daily_requests"]
                lag1, lag2, lag3 = hist.iloc[-1], hist.iloc[-2], hist.iloc[-3]
                roll = hist.tail(7).mean()
                x = pd.DataFrame([[dow, month, int(dow >= 5), doy, lag1, lag2, lag3, roll]], columns=FEATURES)
                preds[bt] = round(max(float(models[bt].predict(x)[0]), 0), 1)
            rows.append({"Date": d.strftime("%b %d, %Y"), "Day": d.strftime("%A"), **preds})
        df = pd.DataFrame(rows)

        shortages = []
        for bt in BLOOD_TYPES:
            below = df[bt].values < 5
            if below.any():
                i = np.where(below)[0][0]
                shortages.append(f"⚠️ {bt}: {df.iloc[i]['Day']} ({df.iloc[i]['Date']}) - predicted {df[bt].values[i]:.1f} units")
        alert = "\n".join(shortages) if shortages else "✅ No critical shortages predicted"
        msg = f"""
📅 Forecast Period: {df.iloc[0]['Date']} - {df.iloc[-1]['Date']}
🔮 XGBoost models (Avg R²: {avg_r2:.3f}) trained on synthetic demand data

{alert}

💡 Recommendation: Schedule donation drives 3-5 days before predicted shortages.
"""
        return df, msg
    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {e}"


def predict_response(age, gender, blood_type, health, contact, city, history, days_since):
    try:
        profile = pd.Series({"age": age, "gender": gender, "blood_type": blood_type,
                             "lat": 29.3956, "lon": 71.6836, "urban": 1,
                             "donation_history": history, "days_since_last": days_since,
                             "preferred_contact": contact, "health_status": health})
        score, b = calculate_donor_score(profile, 29.3956, 71.6836, "normal")
        prob = min(max(score / 100, 0.1), 0.95)
        level = ("✅ HIGH Priority - Contact first" if prob > 0.6
                 else "⚠️ MEDIUM Priority - Backup option" if prob > 0.3
                 else "❌ LOW Priority - Last resort")
        return f"""
🎯 DONOR RESPONSE PREDICTION

Profile: {age}y, {gender}, {blood_type}, {health}, contact via {contact}
History: {history} donations | Days since last: {days_since}

Rule-Based Score: {score:.1f}/100
- Distance: {b['distance_score']}/40 pts (assumed at patient location)
- History: {b['history_score']}/25 pts
- Recency: {b['recency_score']}/20 pts
- Contact: {b['contact_score']}/8 pts
- Health: {b['health_score']}/5 pts
- Urban: {b['urban_score']}/2 pts

Estimated Response Probability: {prob*100:.1f}%

Recommendation:
{level}

⚠️ Based on proven criteria (distance, history, recency). Individual circumstances may vary.
"""
    except Exception as e:
        return f"❌ Error: {e}"


def show_gamification(donor_id):
    donor_id = (donor_id or "").strip()
    if donor_id not in GAMIFICATION_STATE:
        return """
No gamification data found for this donor.

🎮 Start Donating to Earn Rewards!
- Donate blood → Earn 50 points
- Unlock badges at 1, 5, and 10 donations

Rewards:
- 150 pts: Rs. 500 JazzCash voucher
- 300 pts: Rs. 1000 Careem credit
- 500 pts: Priority hospital access card

Badges: 🥉 First Drop (1) | 🥈 LifeSaver (5) | 🥇 Hero (10)
"""
    s = GAMIFICATION_STATE[donor_id]
    return f"""
GAMIFICATION PROFILE

Donor ID: {donor_id}
Total Points: {s['points']}
Total Donations: {s['donations']}
Badges Earned: {', '.join(s['badges']) if s['badges'] else 'None yet'}
"""


#  UI 
with gr.Blocks(title="BloodLink Pakistan", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🩸 BloodLink Pakistan
## AI-Powered Blood Donation Matching System

**Save Lives with Technology** | Hybrid System: Rule-Based Matching + ML Forecasting

⚠️ **DISCLAIMER**: Pre-screening and matching assistance only. All donations require complete medical screening.
""")

    with gr.Tabs():
        with gr.Tab("👤 Donor Registration"):
            gr.Markdown("### Complete Pre-Eligibility Screening")
            with gr.Row():
                with gr.Column():
                    reg_name = gr.Textbox(label="Full Name", placeholder="Muhammad Ali")
                    reg_age = gr.Slider(18, 60, value=30, step=1, label="Age")
                    reg_gender = gr.Radio(["M", "F"], label="Gender", value="M")
                    reg_blood = gr.Dropdown(BLOOD_TYPES, label="Blood Type", value="O+")
                    reg_city = gr.Dropdown([c["city"] for c in CITIES], label="City", value="Karachi")
                    reg_weight = gr.Slider(40, 120, value=65, step=1, label="Weight (kg)")
                with gr.Column():
                    gr.Markdown("### Pre-Eligibility Questionnaire (WHO Guidelines)")
                    reg_answers = []
                    for i, qa in enumerate(ELIGIBILITY_QUESTIONS):
                        if qa["type"] == "yesno":
                            reg_answers.append(gr.Radio(["yes", "no"], label=f"{i+1}. {qa['q']}",
                                                        value="yes" if i < 3 else "no"))
                        else:
                            reg_answers.append(gr.Textbox(label=f"{i+1}. {qa['q']}",
                                                          placeholder="none" if qa["type"] == "text" else "13.5"))
            reg_button = gr.Button("✅ Register as Donor", variant="primary", size="lg")
            reg_output = gr.Textbox(label="Registration Result", lines=15)
            reg_button.click(handle_registration,
                             [reg_name, reg_age, reg_gender, reg_blood, reg_city, reg_weight] + reg_answers,
                             reg_output)

        with gr.Tab("🆘 SOS Emergency"):
            gr.Markdown("### Emergency Blood Request System")
            with gr.Row():
                with gr.Column(scale=2):
                    patient_list_btn = gr.Button("📋 View All Emergency Cases")
                    patient_list_output = gr.Dataframe(label="Current Emergencies")
                    patient_selector = gr.Dropdown(choices=["Custom"] + list(EMERGENCY_PATIENTS.keys()),
                                                   label="Select Patient", value="patient_001")
                    patient_list_btn.click(list_patients, outputs=patient_list_output)
                with gr.Column(scale=1):
                    gr.Markdown("#### Custom Request (select 'Custom' patient)")
                    custom_blood = gr.Dropdown(BLOOD_TYPES, label="Blood Type", value="O+")
                    custom_lat = gr.Slider(24.0, 35.0, value=29.3956, step=0.001, label="Latitude")
                    custom_lon = gr.Slider(66.0, 75.0, value=71.6836, step=0.001, label="Longitude")
                    gr.Markdown("**Quick Coordinates:**\n- Bahawalpur: 29.3956, 71.6836\n- Karachi: 24.8607, 67.0011\n- Lahore: 31.5497, 74.3436")
            sos_button = gr.Button("🚨 ACTIVATE SOS", variant="stop", size="lg")
            with gr.Row():
                sos_table = gr.Dataframe(label="Compatible Donors (Rule-Based Ranking)")
                sos_msg = gr.Textbox(label="Emergency Details & Status", lines=25)
            sos_button.click(handle_sos, [patient_selector, custom_blood, custom_lat, custom_lon],
                             [sos_table, sos_msg])

        with gr.Tab("📊 Shortage Predictions"):
            gr.Markdown(f"### 7-Day Blood Shortage Forecast\n*XGBoost Models (Avg R²: {avg_r2:.3f})*")
            forecast_button = gr.Button("🔮 Generate 7-Day Forecast", variant="primary")
            forecast_table = gr.Dataframe(label="Daily Predictions")
            forecast_msg = gr.Textbox(label="Shortage Alerts & Recommendations", lines=12)
            forecast_button.click(predict_forecast, outputs=[forecast_table, forecast_msg])

        with gr.Tab("🎯 Donor Response Predictor"):
            gr.Markdown("### Predict Donor Likelihood\n*Rule-Based Scoring System*")
            with gr.Row():
                pred_age = gr.Slider(18, 60, value=30, step=1, label="Age")
                pred_gender = gr.Radio(["M", "F"], label="Gender", value="M")
                pred_blood = gr.Dropdown(BLOOD_TYPES, label="Blood Type", value="O+")
            with gr.Row():
                pred_health = gr.Dropdown(["Healthy", "Minor issues", "Unhealthy"], label="Health Status", value="Healthy")
                pred_contact = gr.Dropdown(["App", "SMS", "Phone"], label="Preferred Contact", value="App")
                pred_city = gr.Dropdown([c["city"] for c in CITIES], label="City", value="Karachi")
            with gr.Row():
                pred_history = gr.Slider(0, 20, value=3, step=1, label="Past Donations")
                pred_days = gr.Slider(1, 365, value=60, step=1, label="Days Since Last Donation")
            pred_button = gr.Button("🎯 Calculate Response Score", variant="primary")
            pred_output = gr.Textbox(label="Prediction Result", lines=20)
            pred_button.click(predict_response,
                              [pred_age, pred_gender, pred_blood, pred_health, pred_contact,
                               pred_city, pred_history, pred_days], pred_output)

        with gr.Tab("💬 Blood Donation Chatbot"):
            gr.Markdown("### Ask Questions About Blood Donation\n*Powered by Groq API (Llama 3.3 70B)*"
                        if GROQ_ENABLED else
                        "### Ask Questions About Blood Donation\n*Keyword mode (Groq key not configured)*")
            chat_input = gr.Textbox(label="Your Question", placeholder="e.g., Can I donate during Ramadan?", lines=2)
            chat_button = gr.Button("🤖 Ask AI", variant="primary")
            chat_output = gr.Textbox(label="AI Response", lines=12)
            chat_button.click(chatbot_groq, chat_input, chat_output)

        with gr.Tab("🏆 Gamification & Rewards"):
            gr.Markdown("### Track Your Impact & Earn Rewards")
            gam_donor_id = gr.Textbox(label="Enter Donor ID", placeholder="DNR_00001")
            gam_button = gr.Button("📊 View My Profile", variant="primary")
            gam_output = gr.Textbox(label="Your Gamification Profile", lines=12)
            gam_button.click(show_gamification, gam_donor_id, gam_output)
            gr.Markdown("### Leaderboard (Top Donors)")
            gam_city_filter = gr.Dropdown(["All"] + [c["city"] for c in CITIES], label="Filter by City", value="All")
            gam_lb_button = gr.Button("🏅 Show Leaderboard")
            gam_lb_output = gr.Dataframe(label="Top 10 Donors")
            gam_lb_button.click(lambda city: get_leaderboard(None if city == "All" else city),
                                gam_city_filter, gam_lb_output)

        with gr.Tab("🏥 Hospital Inventory"):
            gr.Markdown("### Blood Inventory Tracking")
            inv_city_filter = gr.Dropdown(["All", "Karachi", "Lahore", "Islamabad", "Bahawalpur"],
                                          label="Filter by City", value="All")
            inv_button = gr.Button("🔍 View Inventory", variant="primary")
            inv_output = gr.Dataframe(label="Hospital Blood Stock")
            inv_button.click(get_inventory, inv_city_filter, inv_output)
            gr.Markdown("---\n### Update Inventory (Hospital Staff Only)")
            inv_hospital = gr.Dropdown(list(HOSPITAL_INVENTORY.keys()), label="Hospital")
            inv_blood_type = gr.Dropdown(BLOOD_TYPES, label="Blood Type")
            inv_units = gr.Slider(0, 50, value=5, step=1, label="Units Available")
            inv_update_button = gr.Button("💾 Update Inventory")
            inv_update_output = gr.Textbox(label="Update Status")
            inv_update_button.click(update_inventory, [inv_hospital, inv_blood_type, inv_units], inv_update_output)

        with gr.Tab("ℹ️ About & Disclaimers"):
            gr.Markdown(f"""
### About BloodLink Pakistan

Hybrid blood donation matching system combining rule-based logic with ML forecasting.

**Performance Metrics (synthetic data):**
- **Donor Matching:** {response_rate:.1f}% simulated response rate (200 simulated emergencies)
- **Top-ranked donor response:** {top1_rate:.1f}%
- **Shortage Forecast:** R² = {avg_r2:.3f}, MAE = {avg_mae:.2f}
- **Average Distance:** {results_df['distance'].mean():.2f} km

**Data:** {len(donor_db):,} synthetic donor profiles, {len(shortage_df):,} synthetic shortage records, 8 blood types, {len(CITIES)} cities

⚠️ **CRITICAL DISCLAIMER:** This is a PRE-SCREENING prototype. All donations MUST undergo complete medical screening at certified centers including HIV, Hepatitis B/C, Malaria testing, blood pressure verification, and blood type cross-matching.

**Version:** 1.0.0 (Prototype)
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
