# NutriVision

This project explores how machine learning can improve the way people understand the nutritional impact of the food they eat. While most food recognition models stop at identifying a dish or estimating calories, our approach tries to answer a more useful question: is this a healthy choice right now, and what should I eat instead? Using image classification, nutrition mapping, and a health scoring system, we built a workflow that takes a picture of a meal and produces a numeric health score along with healthier alternatives in the same category.

To accomplish this, we trained convolutional neural networks on the Food-11 dataset to classify meals into 11 major food groups, and then combined those predictions with USDA nutrition data and recipe-level features to estimate health scores. We also constructed a dashboard that translates these outputs into simple suggestions, allowing users to compare meals and see how different food choices affect their score. The goal of this project is not only to identify what a food item is, but to help users make better decisions by connecting images to meaningful nutritional context.

Links to datasets:
* Food-11 Image Dataset: https://www.kaggle.com/datasets/trolukovich/food11-image-dataset
* USDA Dataset: https://www.kaggle.com/datasets/demomaster/usda-national-nutrient-database
* Recipes Dataset: https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life?select=recipes.csv
