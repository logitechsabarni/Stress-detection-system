import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="AI Stress Detection System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Stress Detection & Measuring System")
st.markdown("Predict stress percentage, detect root causes & receive AI recommendations.")

# ----------------------------------
# Generate Synthetic Dataset
# ----------------------------------

@st.cache_data
def generate_data(n=2000):
    np.random.seed(42)

    sleep = np.random.randint(2, 9, n)
    work_hours = np.random.randint(4, 16, n)
    social_interaction = np.random.randint(0, 10, n)
    physical_activity = np.random.randint(0, 10, n)
    mood = np.random.randint(1, 10, n)
    caffeine = np.random.randint(0, 6, n)
    screen_time = np.random.randint(2, 14, n)
    financial_pressure = np.random.randint(0, 10, n)
    work_pressure = np.random.randint(0, 10, n)
    relationship_stress = np.random.randint(0, 10, n)
    anxiety_level = np.random.randint(0, 10, n)

    stress_percentage = (
        (work_hours / 16) * 15 +
        (screen_time / 14) * 10 +
        (caffeine / 6) * 5 +
        ((9 - sleep) / 9) * 15 +
        (work_pressure / 10) * 15 +
        (financial_pressure / 10) * 10 +
        (relationship_stress / 10) * 10 +
        (anxiety_level / 10) * 10 +
        ((10 - social_interaction) / 10) * 5 +
        ((10 - physical_activity) / 10) * 3 +
        ((10 - mood) / 10) * 7
    )

    stress_percentage += np.random.normal(0, 5, n)
    stress_percentage = np.clip(stress_percentage, 0, 100)

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
        "stress_percentage": stress_percentage
    })

    return df


# ----------------------------------
# Train Model
# ----------------------------------

@st.cache_resource
def train_model():
    df = generate_data()
    X = df.drop("stress_percentage", axis=1)
    y = df["stress_percentage"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model


model = train_model()

# ----------------------------------
# Sidebar Inputs
# ----------------------------------

st.sidebar.header("📋 Enter Lifestyle Parameters")

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
        sleep, work_hours, social_interaction, physical_activity, mood,
        caffeine, screen_time, financial_pressure, work_pressure,
        relationship_stress, anxiety_level
    ]], columns=[
        "sleep", "work_hours", "social_interaction", "physical_activity",
        "mood", "caffeine", "screen_time", "financial_pressure",
        "work_pressure", "relationship_stress", "anxiety_level"
    ])

    stress_percentage = round(float(model.predict(input_df)[0]), 2)

    # Category
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
    # Pie Chart
    # ----------------------------------

    st.subheader("📊 Stress Composition")

    labels = ["Healthy Portion", "Stress Portion"]
    values = [100 - stress_percentage, stress_percentage]

    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct="%1.1f%%")
    st.pyplot(fig1)

    # ----------------------------------
    # 24 Hour Projection
    # ----------------------------------

    st.subheader("📈 24-Hour Stress Projection")

    hours = np.arange(0, 25)

    variation = np.sin(np.linspace(0, 2*np.pi, 25)) * 15
    daily_variation = np.clip(stress_percentage + variation, 0, 100)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(hours, daily_variation, marker="o")
    ax2.fill_between(hours, daily_variation, alpha=0.3)
    ax2.set_xticks(range(0,25))
    ax2.set_ylim(0,100)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Stress Level (%)")
    ax2.set_title("Projected Daily Stress Fluctuation")
    ax2.grid(True)

    st.pyplot(fig2)

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning & Streamlit")
