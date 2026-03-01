import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="AI Stress Detection System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Stress Detection & Measuring System")
st.markdown("Predict stress percentage, identify root causes, and receive AI-powered recommendations.")

# ----------------------------------
# Load Trained Model
# ----------------------------------
model = joblib.load("model/stress_model.pkl")

# ----------------------------------
# Sidebar Inputs (All 11 Parameters)
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
# Prediction Button
# ----------------------------------
if st.button("🔍 Predict Stress Level"):

    # Create Input DataFrame
    input_data = pd.DataFrame([[
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

    # Model Prediction
    prediction = model.predict(input_data)[0]
    stress_percentage = round(float(prediction), 2)

    # ----------------------------------
    # Stress Category
    # ----------------------------------
    if stress_percentage < 35:
        category = "Low Stress 😊"
        color = "green"
    elif stress_percentage < 70:
        category = "Moderate Stress 😐"
        color = "orange"
    else:
        category = "High Stress ⚠️"
        color = "red"

    st.subheader("📊 Stress Prediction Result")
    st.markdown(f"### Predicted Stress Level: `{stress_percentage}%`")
    st.markdown(f"### Category: :{color}[{category}]")

    # ----------------------------------
    # Cause Detection (All 11 Considered)
    # ----------------------------------
    causes = []

    if sleep < 5:
        causes.append("Lack of Sleep")
    if work_hours > 10:
        causes.append("Overworking")
    if social_interaction < 3:
        causes.append("Low Social Interaction")
    if physical_activity < 3:
        causes.append("Low Physical Activity")
    if mood < 4:
        causes.append("Low Mood")
    if caffeine > 4:
        causes.append("High Caffeine Intake")
    if screen_time > 10:
        causes.append("Excessive Screen Time")
    if financial_pressure > 6:
        causes.append("Financial Pressure")
    if work_pressure > 7:
        causes.append("High Work Pressure")
    if relationship_stress > 6:
        causes.append("Relationship Stress")
    if anxiety_level > 6:
        causes.append("High Anxiety Level")

    st.subheader("⚠️ Detected Stress Causes")

    if causes:
        for cause in causes:
            st.write("•", cause)
    else:
        st.write("No major stress triggers detected.")

    # ----------------------------------
    # AI Suggestions
    # ----------------------------------
    st.subheader("💡 AI-Based Recommendations")

    suggestions = []

    if "Lack of Sleep" in causes:
        suggestions.append("Maintain a consistent 7–8 hour sleep schedule.")
    if "Overworking" in causes:
        suggestions.append("Introduce structured breaks during work.")
    if "Low Social Interaction" in causes:
        suggestions.append("Engage in meaningful daily conversations.")
    if "Low Physical Activity" in causes:
        suggestions.append("Include at least 30 minutes of exercise.")
    if "Low Mood" in causes:
        suggestions.append("Practice journaling or mindfulness.")
    if "High Caffeine Intake" in causes:
        suggestions.append("Gradually reduce caffeine consumption.")
    if "Excessive Screen Time" in causes:
        suggestions.append("Follow the 20-20-20 rule and digital detox.")
    if "Financial Pressure" in causes:
        suggestions.append("Create a structured budget and savings plan.")
    if "High Work Pressure" in causes:
        suggestions.append("Use task prioritization frameworks.")
    if "Relationship Stress" in causes:
        suggestions.append("Communicate openly and resolve conflicts calmly.")
    if "High Anxiety Level" in causes:
        suggestions.append("Practice breathing exercises and meditation.")

    if suggestions:
        for suggestion in suggestions:
            st.write("•", suggestion)
    else:
        st.write("Keep maintaining your healthy routine!")

    # ----------------------------------
    # 24-Hour Stress Projection Graph
    # ----------------------------------
    st.subheader("📈 24-Hour Stress Projection")

    hours = np.arange(0, 25)

    variation_pattern = np.array([
        -15, -18, -20, -22, -20,
        -10, 0, 10, 15, 18,
        22, 25, 28, 25, 20,
        15, 10, 5, 0, -5,
        -8, -10, -12, -15, -18
    ])

    daily_variation = stress_percentage + variation_pattern
    daily_variation = np.clip(daily_variation, 0, 100)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(hours, daily_variation, linewidth=3, marker="o")
    ax.fill_between(hours, daily_variation, alpha=0.2)

    ax.axhline(stress_percentage, linestyle="--", linewidth=1)

    ax.set_title("Projected Stress Fluctuation Over 24 Hours")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Stress Level (%)")
    ax.set_xticks(range(0, 25))
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Machine Learning by Sabarni Guha")
