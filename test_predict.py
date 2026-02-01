import pickle
import numpy as np
import pandas as pd

# 1. Load the Saved Model and Encoders
try:
    with open('food_recommender.pkl', 'rb') as f:
        saved_obj = pickle.load(f)

    model = saved_obj['model']
    le_food = saved_obj['le_food']
    scaler = saved_obj['scaler']
    df = saved_obj['data']
    print("--- Food Recommendation System Loaded ---")
except FileNotFoundError:
    print("Error: 'food_recommender.pkl' not found. Please run train_model.py first.")
    exit()

def get_recommendations():
    print("\n" + "="*60)
    # Asking user for Input
    user_input = input("Enter Customer ID to fetch recommendations: ").strip()
    print("="*60)

    try:
        # Check if Customer ID exists in the dataset
        user_history = df[df['Customer ID'] == user_input]

        if user_history.empty:
            print(f"Result: Customer ID '{user_input}' not found in history.")
            return

        # Prepare features for the model (Last order context)
        last_order = user_history.iloc[-1:]
        features = last_order[['cust_encoded', 'food_encoded', 'Rating', 'Bill subtotal']]
        
        # Scaling the input features
        scaled_features = scaler.transform(features)

        # Finding the 5 Nearest Neighbors (Similar Customers/Orders)
        distances, indices = model.kneighbors(scaled_features, n_neighbors=6)

        print(f"\nProfile Found for Customer: {user_input}")
        print(f"Previous Order: {last_order['Items in order'].values[0]}")
        print("-" * 60)
        print("AI PREDICTED RECOMMENDATIONS:")

        recommendations = []
        # Loop through indices (excluding the first one as it's the item itself)
        for i in range(1, len(indices.flatten())):
            food_code = df.iloc[indices.flatten()[i]]['food_encoded']
            food_name = le_food.inverse_transform([int(food_code)])[0]
            recommendations.append(food_name)

        # Remove duplicates and display
        unique_recs = list(dict.fromkeys(recommendations))
        for idx, food in enumerate(unique_recs, 1):
            print(f"Suggestion {idx}: {food}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main execution loop
if __name__ == "__main__":
    while True:
        get_recommendations()
        choice = input("\nWould you like to check another ID? (yes/no): ").lower()
        if choice != 'yes':
            print("Exiting Recommendation System. Goodbye!")
            break