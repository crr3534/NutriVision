import pandas as pd
from PIL import Image
import numpy as np

RECIPES_DF = pd.read_csv("full_recipes_with_nutrition_and_cnn_class-1.csv")

def analyze_uploaded_food_image(
    image_file,
    pred_class,
    craving_nutrient="Sodium",
    top_n_same=3,
    top_n_other_cat=3,
):
    """
    Returns:
    - healthiest recipes in same CNN category (higher score = healthier)
    - healthier individual recipe suggestions from OTHER categories
    """

    # LOAD IMAGE (not used for prediction here)
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    _ = np.array(img) / 255.0  # unused; CNN prediction happens in app.py

    # --------------------------------
    # 1️⃣ SAME CATEGORY — SORT BY SCORE
    # --------------------------------
    same_cat = RECIPES_DF[RECIPES_DF["cnn_category"] == pred_class].copy()
    same_cat = same_cat.sort_values(by="healthiness_score", ascending=False)
    same_cat_recs = same_cat.head(top_n_same)

    # --------------------------------
    # 2️⃣ OTHER CATEGORY SUGGESTIONS
    # Pick top scoring individual recipes from other categories
    # --------------------------------
    other = RECIPES_DF[
    (RECIPES_DF["cnn_category"] != pred_class) &
    (~RECIPES_DF["recipe_name"].str.contains("Water|Sauce|Baby|Drink|Cocktail|Punch|Tea|Soda|Juice|Dressing|Smoothie|Pancakes|Batido|Roughy|Milkshake", case=False, na=False)) &
    (RECIPES_DF["Calories"] > 120) &
    (RECIPES_DF["Protein"] > 3)
].copy()


    # Sort descending — healthiest first
    other = other.sort_values(by="healthiness_score", ascending=False)

    # Return top N healthiest recipes total
    other_recs = other.head(top_n_other_cat)

    return pred_class, same_cat_recs, other_recs



    # remove current category from alternatives
    category_scores = category_scores[category_scores["cnn_category"] != pred_class]

    other_cat_recs = category_scores.head(top_n_other_cat)

    return pred_class, same_cat_recs, other_cat_recs
