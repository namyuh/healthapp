import streamlit as st
import numpy as np
import pickle

# Load Models and Scaler
with open("health_models.pkl", "rb") as f:
    models = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Enhanced Diet Plans
DIET_PLANS = {
    "Obesity Output": {
        "icon": "⚖️",
        "title": "Weight Management Plan",
        "recommendations": [
            "High-protein meals (chicken, fish, tofu)",
            "Low-carb vegetables (spinach, broccoli)",
            "Avoid processed and fried foods",
            "Drink 2L+ water daily",
            "Portion control techniques",
            "Healthy fats (avocados, nuts)"
        ],
        "action": "150+ mins exercise weekly"
    },
    "Cholesterol Output": {
        "icon": "❤️",
        "title": "Heart Health Plan",
        "recommendations": [
            "Daily oats and nuts",
            "Avoid red meat",
            "Leafy greens (kale, spinach)",
            "Olive oil instead of butter",
            "Fatty fish 2-3x/week",
            "Soluble fiber (beans, lentils)"
        ],
        "action": "Regular heart checkups"
    },
    "GERD Output": {
        "icon": "🔥",
        "title": "Acid Reflux Plan",
        "recommendations": [
            "Banana, oatmeal, ginger tea",
            "Avoid spicy/citrus foods",
            "Smaller, frequent meals",
            "No eating 3hrs before bed",
            "Elevate bed head 6-8 inches",
            "Identify trigger foods"
        ],
        "action": "Sleep on left side"
    },
    "Diabetes Output": {
        "icon": "🩸",
        "title": "Blood Sugar Control",
        "recommendations": [
            "Low GI foods (quinoa, sweet potato)",
            "Whole grains only",
            "No sugary drinks",
            "High fiber intake",
            "Balanced macros each meal",
            "Consistent meal timing"
        ],
        "action": "Monitor glucose regularly"
    },
    "nutritional Deficiency Ouput": {
        "icon": "🍎",
        "title": "Nutrition Boost Plan",
        "recommendations": [
            "Iron-rich foods (spinach, lentils)",
            "Vitamin D supplements",
            "B12 sources (eggs, dairy)",
            "Colorful fruits/veggies",
            "Calcium-rich foods",
            "Consider multivitamin"
        ],
        "action": "Annual blood tests"
    }
}

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🏥 Personalized Health Risk Analyzer")

# Input Section
with st.container():
    st.header("📝 Your Health Profile")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.slider("Age", 5, 100, 25)
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)

    with col2:
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0, 0.1)
        glucose = st.slider("Glucose Level (mg/dL)", 70, 300, 95)
        pedigree = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5, 0.01)
        alcohol = st.slider("Alcohol (units/week)", 0, 50, 0)

# Prediction
if st.button("🔍 Analyze My Health Risks", type="primary"):
    input_data = np.array([[1 if gender == "Male" else 0, age, height, weight, bmi, pedigree, glucose, alcohol]])
    input_scaled = scaler.transform(input_data)

    results = {}
    for target, model in models.items():
        pred = model.predict(input_scaled)[0]

        if hasattr(model, "predict_proba"):
            proba_list = model.predict_proba(input_scaled)
            if len(proba_list[0]) > 1:
                proba = proba_list[0][1]
            else:
                proba = None
        else:
            proba = None

        results[target] = {"prediction": pred, "probability": proba}

    # Display results
    st.markdown("---")
    st.header("📊 Your Health Report")

    any_risk = False
    for target, result in results.items():
        disease_name = target.replace(" Output", "")

        if result["prediction"] == 1:
            any_risk = True
            prob = result["probability"] * 100 if result["probability"] is not None else 100
            plan = DIET_PLANS.get(target, {
                "icon": "❓",
                "title": disease_name,
                "recommendations": [],
                "action": ""
            })

            with st.expander(f"{plan['icon']} {disease_name} Risk ({prob:.1f}%)", expanded=True):
                st.subheader(f"{plan['title']}")
                st.markdown("**Recommended Actions:**")
                for item in plan["recommendations"]:
                    st.markdown(f"- {item}")
                st.markdown(f"\n**Pro Tip:** {plan['action']}")

    if not any_risk:
        st.success("🎉 No significant health risks detected!")
        st.markdown("""
        **Maintain your health with:**
        - Balanced diet
        - Regular exercise
        - Good sleep
        - Stress management
        """)

st.markdown("---")
st.caption("Note: This tool provides general health information, not medical advice.")