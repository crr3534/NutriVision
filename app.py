# ==================================================
# IMPORTS
# ==================================================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import altair as alt
from fpdf import FPDF
from analyze_module1 import analyze_uploaded_food_image


# ==================================================
# MODEL + CLASSES
# ==================================================
MODEL_PATH = "food_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    'Bread','Dairy product','Dessert','Egg','Fried food',
    'Meat','Noodles-Pasta','Rice','Seafood','Soup','Vegetable-Fruit'
]


# ==================================================
# HELPERS
# ==================================================
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img,axis=0)

def reset_app():
    st.session_state.page="landing"
    st.session_state.results=None

if "page" not in st.session_state:
    reset_app()


# ==================================================
# LANDING PAGE
# ==================================================
def landing_page():
   st.markdown("""
       <style>
           body { background-color: #0D1117; }
           .hero {
               height: 100vh; display: flex; flex-direction: column;
               justify-content: center; align-items: center; text-align: center; color: white;
           }
           .hero-title {
               font-size: 3.8rem !important; font-weight: 900; margin-bottom: 10px;
           }
           .hero-sub {
               font-size: 1.3rem; line-height: 2rem; max-width: 700px; opacity: .88;
           }
           .glass-box {
               margin-top: 25px; padding: 25px 40px;
               backdrop-filter: blur(14px);
               background: rgba(255,255,255,0.07);
               border-radius: 16px; border: 1px solid rgba(255,255,255,0.12);
           }
           .start-button button {
               background: #4ade80 !important; color: black !important;
               padding: 1.1rem 3rem !important; border-radius: 12px !important;
               font-size: 1.4rem !important; font-weight: 700 !important; margin-top: 40px;
           }
           .start-button button:hover { background: #22c55e !important; transform: scale(1.04); }
       </style>


       <div class="hero">
           <h1 class="hero-title">🥗 Smart AI Food Health Dashboard</h1>
           <div class="glass-box">
               <p class="hero-sub">
                   Upload a meal photo and our AI will:
                   <br> 🍽️ Identify the dish category instantly
                   <br> 🔥 Rank recipes by their healthiness score
                   <br> 🌿 Suggest healthier alternatives you’ll actually eat
                   <br> 📄 Export a polished PDF nutrition summary
               </p>
           </div>
       </div>
   """, unsafe_allow_html=True)


   # BUTTON
   with st.container():
       col1, col2, col3 = st.columns([1,2,1])
       with col2:
           if st.button("🚀 Start Your Meal Analysis", use_container_width=True):
               st.session_state.page = "upload"



# ==================================================
# UPLOAD PAGE
# ==================================================
def upload_page():
    st.title("📸 Upload Your Meal")
    file = st.file_uploader("Choose a food image",type=["jpg","jpeg","png"])

    if file:
        img = Image.open(file)
        st.image(img,use_container_width=True)
        focus = st.selectbox("Health Focus:",["Sodium","Cholesterol"])

        if st.button("Analyze Meal"):
            pred = model.predict(preprocess(img))
            label = class_names[np.argmax(pred)]

            pred_class, same_cat, other_cat = analyze_uploaded_food_image(
                file, pred_class=label, craving_nutrient=focus
            )

            st.session_state.results={
                "image":file,
                "label":label,
                "focus":focus,
                "same":same_cat,
                "other":other_cat
            }

            st.session_state.page="results"

    if st.button("⬅ Home"):
        reset_app()


# ==================================================
# RESULTS PAGE
# ==================================================
def results_page():
    res = st.session_state.results
    st.title("📊 Meal Analysis Results")
    st.image(res["image"],caption="Your Meal",use_container_width=True)

    st.success(f"**Predicted Category:** {res['label']}")

    # ==================================================
    # HEALTH GAUGE (ALWAYS VISIBLE)
    # ==================================================
    score = float(res["same"]["healthiness_score"].mean())
    pct = int(score)

    st.markdown("### ⚡ Category Health Score")
    gauge = f"""
    <div style='display:flex;justify-content:center;'>
    <svg width="200" height="200" viewBox="0 0 36 36">
      <path d="M18 2.0845
               a 15.9155 15.9155 0 0 1 0 31.831"
            fill="none" stroke="#374151" stroke-width="4"/>
      <path d="M18 2.0845
               a 15.9155 15.9155 0 0 1 0 31.831"
            fill="none" stroke="#4ade80" stroke-width="4"
            stroke-dasharray="{pct},100"/>
      <text x="18" y="20" fill="white" text-anchor="middle"
            font-size="8">{pct}/100</text>
    </svg></div>
    """
    st.markdown(gauge,unsafe_allow_html=True)

    # ==================================================
    # SAME CATEGORY TABLE
    # ==================================================
    st.markdown("### 🥇 Healthiest Recipes in This Category")
    st.dataframe(
        res["same"][["recipe_name","cnn_category","ingredients","healthiness_score"]],
        use_container_width=True
    )

    # ==================================================
    # ALTERNATIVES TABLE
    # ==================================================
    st.markdown("### 🌱 Healthier Alternative Foods")
    other = res["other"].copy()
    st.dataframe(other,use_container_width=True)

    # ==================================================
    # NUTRITION CHART (ALWAYS SHOWS)
    # ==================================================
    st.markdown("### 📊 Nutrition Comparison Across Alternatives")

    if len(other) > 0:
        nutr_cols=["Calories","Protein","TotalFat","Cholesterol"]

        other_long = other.melt(
            id_vars=["recipe_name"],
            value_vars=nutr_cols,
            var_name="Nutrient",
            value_name="NutrientValue"
        )

        chart = (
            alt.Chart(other_long)
            .mark_bar()
            .encode(
                x=alt.X("recipe_name:N",title="Recipe Name"),
                y=alt.Y("NutrientValue:Q",title="Nutrient Value"),
                color=alt.Color("Nutrient:N"),
                tooltip=["recipe_name","Nutrient","NutrientValue"]
            ).properties(height=420)
        )

        st.altair_chart(chart,use_container_width=True)

    else:
        st.warning("⚠️ No alternative recipes available with nutrition data.")
        st.info("Try another meal image.")

    # ==================================================
    # ACTION BUTTONS
    # ==================================================
    if st.button("🔄 Analyze Another Meal"):
        reset_app()


# ==================================================
# ROUTER
# ==================================================
if __name__=="__main__":
    if st.session_state.page=="landing": landing_page()
    elif st.session_state.page=="upload": upload_page()
    elif st.session_state.page=="results": results_page()
