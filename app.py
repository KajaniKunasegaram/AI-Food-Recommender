from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and data
with open('food_recommender.pkl', 'rb') as f:
    saved_obj = pickle.load(f)

model = saved_obj['model']
le_food = saved_obj['le_food']
scaler = saved_obj['scaler']
df = saved_obj['data']

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    cust_id = data.get('customer_id')

    try:
        user_history = df[df['Customer ID'] == cust_id]
        if user_history.empty:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404

        # Logic to get neighbors
        last_order = user_history.iloc[-1:]
        features = last_order[['cust_encoded', 'food_encoded', 'Rating', 'Bill subtotal']]
        scaled_features = scaler.transform(features)
        distances, indices = model.kneighbors(scaled_features, n_neighbors=5)

        recommendations = []
        for i in range(1, len(indices.flatten())):
            food_code = df.iloc[indices.flatten()[i]]['food_encoded']
            food_name = le_food.inverse_transform([int(food_code)])[0]
            recommendations.append(food_name)

        return jsonify({
            'status': 'success',
            'recommendations': list(set(recommendations))
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000) # This runs on http://127.0.0.1:5000