import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="AI Stress Detection System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Stress Detection & Measuring System")
st.markdown("Predict stress percentage, analyze psychological triggers, and receive personalized AI-driven recommendations.")

# ----------------------------------
# Train Synthetic Model
# ----------------------------------
@st.cache_resource
def train_model():

    np.random.seed(42)
    size = 2500

    sleep = np.random.randint(2, 10, size)
    work_hours = np.random.randint(4, 16, size)
    social_interaction = np.random.randint(0, 10, size)
    physical_activity = np.random.randint(0, 10, size)
    mood = np.random.randint(1, 10, size)
    caffeine = np.random.randint(0, 8, size)
    screen_time = np.random.randint(2, 14, size)
    financial_pressure = np.random.randint(0, 10, size)
    work_pressure = np.random.randint(0, 10, size)
    relationship_stress = np.random.randint(0, 10, size)
    anxiety_level = np.random.randint(0, 10, size)

    stress_score = (
        (10 - sleep) * 3 +
        work_hours * 2 +
        (10 - social_interaction) * 1.5 +
        (10 - physical_activity) * 1.5 +
        (10 - mood) * 2 +
        caffeine * 1.5 +
        screen_time * 1.5 +
        financial_pressure * 2 +
        work_pressure * 2.5 +
        relationship_stress * 2 +
        anxiety_level * 3
    )

    stress_score = stress_score / stress_score.max() * 100

    df = pd.DataFrame({
        "sleep": sleep,
        "work_hours": work_hours,
        "social_interaction": social_interaction,
        "physical_activity": physical_activity,
        "mood": mood,
        "caffeine": caffeine,
        "screen_time": screen_time,
        "financial_pressure": financial_pressure,
        "work_pressure": work_pressure,
        "relationship_stress": relationship_stress,
        "anxiety_level": anxiety_level,
        "stress_percentage": stress_score
    })

    X = df.drop("stress_percentage", axis=1)
    y = df["stress_percentage"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("📋 Enter Daily Lifestyle Parameters")

sleep = st.sidebar.slider("Sleep Hours", 0, 12, 6)
work_hours = st.sidebar.slider("Work Hours", 0, 16, 8)
social_interaction = st.sidebar.slider("Social Interaction (0-10)", 0, 10, 5)
physical_activity = st.sidebar.slider("Physical Activity (0-10)", 0, 10, 5)
mood = st.sidebar.slider("Mood Level (1-10)", 1, 10, 5)
caffeine = st.sidebar.slider("Caffeine Intake (cups)", 0, 10, 2)
screen_time = st.sidebar.slider("Screen Time (hrs)", 0, 16, 6)
financial_pressure = st.sidebar.slider("Financial Pressure (0-10)", 0, 10, 3)
work_pressure = st.sidebar.slider("Work Pressure (0-10)", 0, 10, 5)
relationship_stress = st.sidebar.slider("Relationship Stress (0-10)", 0, 10, 3)
anxiety_level = st.sidebar.slider("Anxiety Level (0-10)", 0, 10, 4)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict Stress Level"):

    input_df = pd.DataFrame([[ 
        sleep, work_hours, social_interaction,
        physical_activity, mood, caffeine,
        screen_time, financial_pressure,
        work_pressure, relationship_stress,
        anxiety_level
    ]], columns=[
        "sleep", "work_hours", "social_interaction",
        "physical_activity", "mood", "caffeine",
        "screen_time", "financial_pressure",
        "work_pressure", "relationship_stress",
        "anxiety_level"
    ])

    stress_percentage = round(float(model.predict(input_df)[0]), 2)

    # ----------------------------------
    # Category
    # ----------------------------------
    if stress_percentage < 35:
        category = "Low Stress 😊"
        explanation = "Your mental balance appears stable with minimal psychological strain."
    elif stress_percentage < 70:
        category = "Moderate Stress ⚠️"
        explanation = "Noticeable stress patterns detected. Lifestyle optimization recommended."
    else:
        category = "High Stress 🔴"
        explanation = "Significant stress accumulation detected. Immediate corrective action advised."

    st.subheader("📊 Stress Prediction Result")
    st.markdown(f"### Predicted Stress Level: `{stress_percentage}%`")
    st.markdown(f"### Category: {category}")
    st.info(explanation)

    # ----------------------------------
    # Contribution Analysis
    # ----------------------------------
    contributions = {
        "Sleep Deficit": (10 - sleep) * 3,
        "Work Hours": work_hours * 2,
        "Low Social Interaction": (10 - social_interaction) * 1.5,
        "Low Physical Activity": (10 - physical_activity) * 1.5,
        "Low Mood": (10 - mood) * 2,
        "Caffeine": caffeine * 1.5,
        "Screen Time": screen_time * 1.5,
        "Financial Pressure": financial_pressure * 2,
        "Work Pressure": work_pressure * 2.5,
        "Relationship Stress": relationship_stress * 2,
        "Anxiety Level": anxiety_level * 3
    }

    # Horizontal Bar Chart
    st.subheader("📊 Stress Contribution Analysis")
    sorted_contrib = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(list(sorted_contrib.keys()), list(sorted_contrib.values()), color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Impact Score")
    ax.set_title("Parameter Contribution to Stress")
    st.pyplot(fig)

    # Pie Chart
    st.subheader("📊 Stress Contribution Breakdown (Pie Chart)")
    labels = list(contributions.keys())
    values = list(contributions.values())
    fig1, ax1 = plt.subplots(figsize=(7,7))
    ax1.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax1.set_title("Stress Parameter Contribution")
    st.pyplot(fig1)

    # ----------------------------------
    # Dynamic 24 Hour Projection
    # ----------------------------------
    st.subheader("📈 24-Hour Stress Projection")
    hours = np.arange(0, 24)
    amplitude = (stress_percentage / 100) * (10 + anxiety_level + caffeine)
    phase_shift = (work_hours / 24) * np.pi
    curve = np.sin(np.linspace(0, 2*np.pi, 24) + phase_shift)
    projection = np.clip(stress_percentage + curve * amplitude, 0, 100)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(hours, projection, marker="o")
    ax2.fill_between(hours, projection, alpha=0.3)
    ax2.set_ylim(0,100)
    ax2.set_xticks(range(0,24))
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Stress Level (%)")
    ax2.set_title("Dynamic Daily Stress Fluctuation")
    ax2.grid(True)
    st.pyplot(fig2)

    # ----------------------------------
    # Detailed Psychological Analysis
    # ----------------------------------
    st.subheader("🧠 Detailed Psychological Insights")
    insights = []
    recommendations = []

    def analyze(param, value, low, high, high_text, low_text, recommendation):
        if value > high:
            insights.append(f"🔴 {high_text}")
            recommendations.append(recommendation)
        elif value < low:
            insights.append(f"🟡 {low_text}")
            recommendations.append(recommendation)
        else:
            insights.append(f"🟢 {param} is within healthy range.")

    analyze("Sleep", sleep, 6, 9,
            "Sleep deprivation increasing cortisol and emotional fatigue.",
            "Excess sleep may signal burnout.",
            "Maintain consistent 7–8 hour sleep cycle.")

    analyze("Work Hours", work_hours, 4, 9,
            "Overworking elevating chronic stress response.",
            "Low engagement reducing motivation.",
            "Introduce structured work breaks.")

    analyze("Caffeine", caffeine, 0, 4,
            "High caffeine intake intensifying anxiety signals.",
            "Low caffeine (no issue).",
            "Limit caffeine to 1–2 cups daily.")

    analyze("Screen Time", screen_time, 2, 9,
            "Excessive screen exposure impairing recovery.",
            "Very low digital interaction.",
            "Implement evening digital detox.")

    analyze("Anxiety Level", anxiety_level, 0, 6,
            "Elevated anxiety strongly driving stress.",
            "Low anxiety baseline.",
            "Practice breathing exercises & mindfulness.")

    analyze("Relationship Stress", relationship_stress, 0, 6,
            "Relationship conflicts affecting emotional stability.",
            "Low relational strain.",
            "Encourage calm communication & boundaries.")

    analyze("Financial Pressure", financial_pressure, 0, 6,
            "Financial strain impacting psychological security.",
            "Low financial strain.",
            "Create structured budget planning.")

    analyze("Mood", mood, 5, 9,
            "Low mood reducing resilience and optimism.",
            "Extremely elevated mood.",
            "Practice gratitude journaling.")

    analyze("Physical Activity", physical_activity, 4, 9,
            "Low movement reducing endorphin production.",
            "Excessive exertion risk.",
            "Include 30 min daily exercise.")

    analyze("Social Interaction", social_interaction, 4, 9,
            "Social isolation risk detected.",
            "Excessive social fatigue.",
            "Increase meaningful conversations.")

    st.markdown("### 🔍 Parameter Insights")
    for i in insights:
        st.write(i)

    st.markdown("### 🤖 AI Personalized Recommendations")
    if recommendations:
        for r in set(recommendations):
            st.write("•", r)
    else:
        st.success("Your psychological parameters are well balanced.")

    if stress_percentage > 75:
        st.error("⚠️ Consider consulting a licensed mental health professional if high stress persists.")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Machine Learning by Sabarni Guha")
