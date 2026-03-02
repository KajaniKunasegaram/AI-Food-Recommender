import pickle
import numpy as np
import pandas as pd

# 1. Load the Saved Model and Encoders
try:
    with open('food_recommender.pkl', 'rb') as f:
        saved_obj = pickle.load(f)
    model   = saved_obj['model']
    le_food = saved_obj['le_food']
    le_cust = saved_obj['le_cust']
    scaler  = saved_obj['scaler']
    df      = saved_obj['data']
    print("--- Food Recommendation System Loaded ---")
    # print(f"    Total records : {len(df)}")
    # print(f"    Known customers: {df['customer_id'].nunique()}")
except FileNotFoundError:
    print("Error: 'food_recommender.pkl' not found. Please run train_model.py first.")
    exit()

def get_recommendations():
    print("\n" + "="*60)
    user_input = input("Enter Customer ID: ").strip()
    print("="*60)

    try:
        # Check if Customer ID exists
        user_history = df[df['customer_id'] == user_input]

        if user_history.empty:
            print(f"Result: Customer ID '{user_input}' not found in history.")
            return

        # Last order
        last_order = user_history.iloc[-1:]

        # Features — now includes sub_cat_id
        features = last_order[['cust_encoded', 'food_encoded', 'sub_cat_id', 'Rating', 'subtotal']]

        # Scale
        scaled_features = scaler.transform(features)

        # KNN
        n_neighbors      = min(6, len(df))
        distances, indices = model.kneighbors(scaled_features, n_neighbors=n_neighbors)

        print(f"\nProfile Found for Customer: {user_input[:20]}...")
        print(f"Previous Order item_id: {last_order['item_id'].values[0]}")
        # print(f"Sub Category          : {last_order['sub_cat_id'].values[0]}")
        print("-" * 60)
        print("AI PREDICTED RECOMMENDATIONS:")

        recommendations = []
        for i in range(1, len(indices.flatten())):
            food_code = df.iloc[indices.flatten()[i]]['food_encoded']
            food_id   = le_food.inverse_transform([int(food_code)])[0]
            recommendations.append(str(food_id))

        unique_recs = list(dict.fromkeys(recommendations))
        for idx, food in enumerate(unique_recs, 1):
            print(f"Suggestion {idx}: item_id = {food}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    while True:
        get_recommendations()
        choice = input("\nWould you like to check another ID? (yes/no): ").lower()
        if choice != 'yes':
            print("Exiting Recommendation System.")
            break