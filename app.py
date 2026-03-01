import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="AI Stress Detection System", page_icon="🧠", layout="wide")

st.title("🧠 AI-Based Stress Detection & Measuring System")
st.markdown("Predict stress percentage, identify root causes, and receive AI-powered recommendations.")

# ----------------------------------
# Generate Synthetic Training Data
# ----------------------------------
@st.cache_resource
def train_model():

    np.random.seed(42)
    data_size = 2000

    sleep = np.random.randint(2, 10, data_size)
    work_hours = np.random.randint(4, 16, data_size)
    social_interaction = np.random.randint(0, 10, data_size)
    physical_activity = np.random.randint(0, 10, data_size)
    mood = np.random.randint(1, 10, data_size)
    caffeine = np.random.randint(0, 8, data_size)
    screen_time = np.random.randint(2, 14, data_size)
    financial_pressure = np.random.randint(0, 10, data_size)
    work_pressure = np.random.randint(0, 10, data_size)
    relationship_stress = np.random.randint(0, 10, data_size)
    anxiety_level = np.random.randint(0, 10, data_size)

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

    model = RandomForestRegressor()
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
    # Stress Category
    # ----------------------------------
    if stress_percentage < 35:
        category = "Low Stress 😊"
        color = "green"
        explanation = "You are maintaining a healthy and balanced routine."
    elif stress_percentage < 70:
        category = "Moderate Stress ⚠️"
        color = "orange"
        explanation = "Your stress is noticeable. Lifestyle adjustments recommended."
    else:
        category = "High Stress 🔴"
        color = "red"
        explanation = "Your stress level is significantly high. Immediate intervention advised."

    st.subheader("📊 Stress Prediction Result")
    st.markdown(f"### Predicted Stress Level: `{stress_percentage}%`")
    st.markdown(f"### Category: :{color}[{category}]")
    st.info(explanation)

    # ----------------------------------
    # Pie Chart (All 11 Parameters)
    # ----------------------------------
    st.subheader("📊 Stress Contribution Breakdown")

    contributions = {
        "Sleep": (10 - sleep) * 3,
        "Work Hours": work_hours * 2,
        "Social Interaction": (10 - social_interaction) * 1.5,
        "Physical Activity": (10 - physical_activity) * 1.5,
        "Mood": (10 - mood) * 2,
        "Caffeine": caffeine * 1.5,
        "Screen Time": screen_time * 1.5,
        "Financial Pressure": financial_pressure * 2,
        "Work Pressure": work_pressure * 2.5,
        "Relationship Stress": relationship_stress * 2,
        "Anxiety Level": anxiety_level * 3
    }

    labels = list(contributions.keys())
    values = list(contributions.values())

    fig1, ax1 = plt.subplots(figsize=(7,7))
    ax1.pie(values, labels=labels, autopct="%1.1f%%")
    ax1.set_title("Stress Parameter Contribution")
    st.pyplot(fig1)

    # ----------------------------------
    # 24-Hour Dynamic Projection
    # ----------------------------------
    st.subheader("📈 24-Hour Stress Projection")

    hours = np.arange(0, 25)
    amplitude = stress_percentage / 5
    variation = np.sin(np.linspace(0, 2*np.pi, 25)) * amplitude

    projection = np.clip(stress_percentage + variation, 0, 100)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(hours, projection, marker="o")
    ax2.fill_between(hours, projection, alpha=0.3)
    ax2.set_xticks(range(0,25))
    ax2.set_ylim(0,100)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Stress Level (%)")
    ax2.set_title("Projected Stress Fluctuation Over 24 Hours")
    ax2.grid(True)
    st.pyplot(fig2)

    # ----------------------------------
    # AI Recommendations
    # ----------------------------------
    st.subheader("🤖 AI-Based Personalized Recommendations")

    suggestions = []

    if sleep < 6:
        suggestions.append("Increase sleep to 7–8 hours.")
    if work_hours > 9:
        suggestions.append("Reduce overworking and take structured breaks.")
    if social_interaction < 4:
        suggestions.append("Engage in more social interaction.")
    if physical_activity < 4:
        suggestions.append("Include 30 minutes of exercise daily.")
    if mood < 5:
        suggestions.append("Practice mindfulness or journaling.")
    if caffeine > 4:
        suggestions.append("Reduce caffeine intake gradually.")
    if screen_time > 9:
        suggestions.append("Adopt digital detox habits.")
    if financial_pressure > 6:
        suggestions.append("Create a budget management plan.")
    if work_pressure > 6:
        suggestions.append("Use task prioritization frameworks.")
    if relationship_stress > 6:
        suggestions.append("Improve communication in relationships.")
    if anxiety_level > 6:
        suggestions.append("Practice breathing exercises & meditation.")

    if stress_percentage > 75:
        suggestions.append("Consider consulting a mental health professional.")

    if suggestions:
        for s in suggestions:
            st.write("•", s)
    else:
        st.success("Your lifestyle indicators are balanced. Keep it up!")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Machine Learning by Sabarni Guha")
