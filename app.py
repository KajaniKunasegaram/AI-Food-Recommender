
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sqlalchemy import create_engine

app = Flask(__name__)
CORS(app)

DB_USER = 'root'
DB_PASS = ''
DB_HOST = '127.0.0.1'
DB_NAME = 'food_ordering_db'
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

# Load Model
try:
    with open('food_recommender.pkl', 'rb') as f:
        saved_obj = pickle.load(f)
    model       = saved_obj['model']
    scaler      = saved_obj['scaler']
    le_cust     = saved_obj['le_cust']
    le_food     = saved_obj['le_food']
    original_df = saved_obj['data']
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Load Error:", e)
    exit()

# Popular Items Fallback
def get_popular_items(limit=5):
    query = """
        SELECT t.item_id, COUNT(*) as order_count
        FROM tbl_torder t
        JOIN tbl_morder m ON t.order_id = m.id
        GROUP BY t.item_id
        ORDER BY order_count DESC
        LIMIT %s
    """
    df = pd.read_sql(query, engine, params=(limit,))
    return df['item_id'].astype(str).tolist()

#  Content-Based (same sub_cat_id)
def get_same_category_items(item_id, exclude_item_id, limit=5):
    """Return items from the same sub_category as the given item."""
    query = """
        SELECT i2.item_id
        FROM tbl_items i1
        JOIN tbl_items i2 ON i1.sub_cat_id = i2.sub_cat_id
        WHERE i1.item_id = %s
          AND i2.item_id != %s
          AND i2.item_status = 1
        LIMIT %s
    """
    df = pd.read_sql(query, engine, params=(item_id, exclude_item_id, limit))
    return df['item_id'].astype(str).tolist()

@app.route('/')
def home():
    return "🍽 AI  Food Recommendation API Running"

@app.route('/api/recommend', methods=['POST'])
def recommend():
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'JSON required'}), 400

    data    = request.get_json()
    cust_id = str(data.get('customer_id')).strip()

    if not cust_id:
        return jsonify({'status': 'error', 'message': 'customer_id required'}), 400

    #  Unknown customer → Popular items
    if cust_id not in [str(c) for c in le_cust.classes_]:
        print(f"⚠️  Unknown customer — returning popular items")
        popular = get_popular_items(limit=5)
        if not popular:
            return jsonify({'status': 'error', 'message': 'No data available'}), 404
        return jsonify({
            'status'         : 'success',
            'customer_id'    : cust_id,
            'recommendations': popular,
            'type'           : 'popular'
        })

    # Known customer → Hybrid Recommendation
    # Collaborative (KNN) + Content-Based (sub_cat_id)
    try:
        query = """
            SELECT 
                m.customer_id,
                t.item_id,
                t.total_price AS subtotal,
                i.sub_cat_id
            FROM tbl_morder m
            JOIN tbl_torder t ON t.order_id = m.id
            JOIN tbl_items  i ON i.item_id  = t.item_id
            WHERE m.customer_id = %s
            ORDER BY m.created_at
        """
        df = pd.read_sql(query, engine, params=(cust_id,))

        if df.empty:
            popular = get_popular_items(limit=5)
            return jsonify({
                'status'         : 'success',
                'customer_id'    : cust_id,
                'recommendations': popular,
                'type'           : 'popular'
            })

        df['customer_id'] = df['customer_id'].astype(str)
        df['item_id']     = df['item_id'].astype(str)
        df['Rating']      = 3.5
        df['sub_cat_id']  = pd.to_numeric(df['sub_cat_id'], errors='coerce').fillna(0).astype(int)

        # Encode customer
        df['cust_encoded'] = le_cust.transform(df['customer_id'])

        # Filter unseen items
        known_items = set([str(c) for c in le_food.classes_])
        df = df[df['item_id'].isin(known_items)]

        if df.empty:
            popular = get_popular_items(limit=5)
            return jsonify({
                'status'         : 'success',
                'customer_id'    : cust_id,
                'recommendations': popular,
                'type'           : 'popular'
            })

        df['food_encoded'] = le_food.transform(df['item_id'])

        # Last order info
        last_order      = df.iloc[-1:]
        last_item_id    = last_order['item_id'].values[0]
        last_sub_cat_id = last_order['sub_cat_id'].values[0]

        features = last_order[['cust_encoded', 'food_encoded', 'sub_cat_id', 'Rating', 'subtotal']]
        scaled   = scaler.transform(features)

        # Collaborative Filtering (KNN)
        n_neighbors        = min(6, len(original_df))
        distances, indices = model.kneighbors(scaled, n_neighbors=n_neighbors)

        collaborative_recs = []
        for i in range(1, len(indices.flatten())):
            food_code = original_df.iloc[indices.flatten()[i]]['food_encoded']
            food_id   = str(le_food.inverse_transform([food_code])[0])
            collaborative_recs.append(food_id)

        collaborative_recs = list(dict.fromkeys(collaborative_recs))  

        # Content-Based Filtering (same sub_cat_id)        
        content_recs = get_same_category_items(
            item_id=last_item_id,
            exclude_item_id=last_item_id,
            limit=5
        )

        #  Merge both (Content first, then Collaborative)
        hybrid_recs = content_recs.copy()
        for item in collaborative_recs:
            if item not in hybrid_recs:
                hybrid_recs.append(item)

        hybrid_recs = hybrid_recs[:5]  

        print(f"✅ Collaborative : {collaborative_recs}")
        print(f"✅ Content-Based : {content_recs}")
        print(f"✅ Hybrid Final  : {hybrid_recs}")

        return jsonify({
            'status'         : 'success',
            'customer_id'    : cust_id,
            'recommendations': hybrid_recs,
            'type'           : 'ai'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)





























# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import pandas as pd
# from sqlalchemy import create_engine

# # -------------------------------
# # Flask Setup
# # -------------------------------
# app = Flask(__name__)
# CORS(app)

# # -------------------------------
# # Database Config
# # -------------------------------
# DB_USER = 'root'
# DB_PASS = ''
# DB_HOST = '127.0.0.1'
# DB_NAME = 'food_ordering_db'
# engine = create_engine(
#     f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
# )

# # -------------------------------
# # Load Trained Model
# # -------------------------------
# try:
#     with open('food_recommender.pkl', 'rb') as f:
#         saved_obj = pickle.load(f)
#     model       = saved_obj['model']
#     scaler      = saved_obj['scaler']
#     le_cust     = saved_obj['le_cust']
#     le_food     = saved_obj['le_food']
#     original_df = saved_obj['data']
#     print("✅ Model Loaded Successfully")
# except Exception as e:
#     print("❌ Model Load Error:", e)
#     exit()

# # -------------------------------
# # Helper: Get Popular Items
# # Fallback for new/unknown customers
# # -------------------------------
# def get_popular_items(limit=5):
#     query = """
#         SELECT t.item_id, COUNT(*) as order_count
#         FROM tbl_torder t
#         JOIN tbl_morder m ON t.order_id = m.id
#         GROUP BY t.item_id
#         ORDER BY order_count DESC
#         LIMIT %s
#     """
#     df = pd.read_sql(query, engine, params=(limit,))
#     return df['item_id'].astype(str).tolist()

# # -------------------------------
# # Root Route
# # -------------------------------
# @app.route('/')
# def home():
#     return "🍽 AI Food Recommendation API Running"

# # -------------------------------
# # Recommendation API
# # -------------------------------
# @app.route('/api/recommend', methods=['POST'])
# def recommend():
#     if not request.is_json:
#         return jsonify({'status': 'error', 'message': 'JSON required'}), 400

#     data    = request.get_json()
#     cust_id = str(data.get('customer_id')).strip()

#     if not cust_id:
#         return jsonify({'status': 'error', 'message': 'customer_id required'}), 400

#     # ---------------------------------------------------------------
#     # CASE 1: Customer not in training data
#     # (new customer or model not retrained yet)
#     # → Return most popular items as fallback
#     # ---------------------------------------------------------------
#     if cust_id not in [str(c) for c in le_cust.classes_]:
#         print(f"⚠️  Unknown customer {cust_id[:10]}... — returning popular items")
#         popular = get_popular_items(limit=5)
#         if not popular:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'No recommendation data available yet'
#             }), 404
#         return jsonify({
#             'status':          'success',
#             'customer_id':     cust_id,
#             'recommendations': popular,
#             'type':            'popular'
#         })

#     # ---------------------------------------------------------------
#     # CASE 2: Known customer → KNN AI recommendation
#     # ---------------------------------------------------------------
#     try:
#         query = """
#             SELECT m.customer_id,
#                    t.item_id,
#                    t.total_price as subtotal
#             FROM tbl_morder m
#             JOIN tbl_torder t ON t.order_id = m.id
#             WHERE m.customer_id = %s
#             ORDER BY m.created_at
#         """
#         df = pd.read_sql(query, engine, params=(cust_id,))

#         if df.empty:
#             popular = get_popular_items(limit=5)
#             return jsonify({
#                 'status':          'success',
#                 'customer_id':     cust_id,
#                 'recommendations': popular,
#                 'type':            'popular'
#             })

#         df['customer_id'] = df['customer_id'].astype(str)
#         df['item_id']     = df['item_id'].astype(str)
#         df['Rating']      = 3.5

#         df['cust_encoded'] = le_cust.transform(df['customer_id'])

#         # Filter out any item_ids not seen during training
#         known_items = set([str(c) for c in le_food.classes_])
#         df = df[df['item_id'].isin(known_items)]

#         if df.empty:
#             popular = get_popular_items(limit=5)
#             return jsonify({
#                 'status':          'success',
#                 'customer_id':     cust_id,
#                 'recommendations': popular,
#                 'type':            'popular'
#             })

#         df['food_encoded'] = le_food.transform(df['item_id'])

#         last_order = df.iloc[-1:]
#         features   = last_order[['cust_encoded', 'food_encoded', 'Rating', 'subtotal']]
#         scaled     = scaler.transform(features)

#         # distances, indices = model.kneighbors(scaled, n_neighbors=6)
#         n_neighbors = min(6, len(original_df))
#         distances, indices = model.kneighbors(scaled, n_neighbors=n_neighbors)

#         recommendations = []
#         for i in range(1, len(indices.flatten())):
#             food_code = original_df.iloc[indices.flatten()[i]]['food_encoded']
#             food_id   = le_food.inverse_transform([food_code])[0]
#             recommendations.append(str(food_id))

#         unique_recommendations = list(dict.fromkeys(recommendations))

#         return jsonify({
#             'status':          'success',
#             'customer_id':     cust_id,
#             'recommendations': unique_recommendations,
#             'type':            'ai'
#         })

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# # -------------------------------
# # Run Server
# # -------------------------------
# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000, debug=True)
































# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import pandas as pd
# from sqlalchemy import create_engine

# # -------------------------------
# # Flask Setup
# # -------------------------------
# app = Flask(__name__)
# CORS(app)

# # -------------------------------
# # Database Config
# # -------------------------------
# DB_USER = 'root'
# DB_PASS = ''
# DB_HOST = '127.0.0.1'
# DB_NAME = 'food_ordering_db'

# engine = create_engine(
#     f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
# )

# # -------------------------------
# # Load Trained Model
# # -------------------------------
# try:
#     with open('food_recommender.pkl', 'rb') as f:
#         saved_obj = pickle.load(f)

#     model = saved_obj['model']
#     scaler = saved_obj['scaler']
#     le_cust = saved_obj['le_cust']
#     le_food = saved_obj['le_food']
#     original_df = saved_obj['data']

#     print("✅ Model Loaded Successfully")

# except Exception as e:
#     print("❌ Model Load Error:", e)
#     exit()

# # -------------------------------
# # Root Route
# # -------------------------------
# @app.route('/')
# def home():
#     return "🍽 AI Food Recommendation API Running"

# # -------------------------------
# # Recommendation API
# # -------------------------------
# @app.route('/api/recommend', methods=['POST'])
# def recommend():

#     if not request.is_json:
#         return jsonify({'status': 'error', 'message': 'JSON required'}), 400

#     data = request.get_json()
#     cust_id = str(data.get('customer_id')).strip()

#     if not cust_id:
#         return jsonify({'status': 'error', 'message': 'customer_id required'}), 400

#     try:
#         # -------------------------------
#         # Fetch Customer Orders
#         # -------------------------------
#         query = """
#             SELECT m.customer_id,
#                    t.item_id,
#                    t.total_price as subtotal
#             FROM tbl_morder m
#             JOIN tbl_torder t ON t.order_id = m.id
#             WHERE m.customer_id = %s
#             ORDER BY m.created_at
#         """

#         df = pd.read_sql(query, engine, params=(cust_id,))

#         if df.empty:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'No order history found'
#             }), 404

#         # Add Rating
#         df['Rating'] = 3.5

#         # Encode using saved encoders
#         df['cust_encoded'] = le_cust.transform(df['customer_id'])
#         df['food_encoded'] = le_food.transform(df['item_id'])

#         # Take last order
#         last_order = df.iloc[-1:]

#         features = last_order[['cust_encoded', 'food_encoded', 'Rating', 'subtotal']]

#         # Scale
#         scaled = scaler.transform(features)

#         # Get neighbors
#         distances, indices = model.kneighbors(scaled, n_neighbors=6)

#         recommendations = []

#         for i in range(1, len(indices.flatten())):
#             food_code = original_df.iloc[indices.flatten()[i]]['food_encoded']
#             food_id = le_food.inverse_transform([food_code])[0]
#             recommendations.append(str(food_id))

#         unique_recommendations = list(dict.fromkeys(recommendations))

#         return jsonify({
#             'status': 'success',
#             'customer_id': cust_id,
#             'recommendations': unique_recommendations
#         })

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# # -------------------------------
# # Run Server
# # -------------------------------
# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000, debug=True)

























# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle
# import pandas as pd
# from sqlalchemy import create_engine

# # -------------------------------
# # Flask App Setup
# # -------------------------------
# app = Flask(__name__)
# CORS(app)

# # -------------------------------
# # Database connection
# # -------------------------------
# DB_USER = 'root'
# DB_PASS = ''
# DB_HOST = '127.0.0.1'
# DB_NAME = 'food_ordering_db'

# engine = create_engine(
#     f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}'
# )

# # -------------------------------
# # Load Trained Model
# # -------------------------------
# try:
#     with open('food_recommender.pkl', 'rb') as f:
#         saved_obj = pickle.load(f)

#     model = saved_obj['model']
#     scaler = saved_obj['scaler']
#     le_food = saved_obj['le_food']

#     print("✅ Recommendation Model Loaded Successfully")

# except Exception as e:
#     print("❌ Error loading model:", e)
#     exit()

# # -------------------------------
# # Root Route
# # -------------------------------
# @app.route('/')
# def home():
#     return "🍽 AI Food Recommendation API is running successfully!"

# # -------------------------------
# # Recommendation API
# # -------------------------------
# @app.route('/api/recommend', methods=['POST'])
# def recommend():

#     if not request.is_json:
#         return jsonify({
#             'status': 'error',
#             'message': 'Request must be JSON'
#         }), 400

#     data = request.get_json()
#     cust_id = str(data.get('customer_id')).strip()

#     if not cust_id:
#         return jsonify({
#             'status': 'error',
#             'message': 'customer_id is required'
#         }), 400

#     try:
#         # -------------------------------
#         # Fetch Order History from Database
#         # -------------------------------
#         query = """
#             SELECT m.customer_id, t.item_id, t.total_price as subtotal
#             FROM tbl_morder m
#             JOIN tbl_torder t ON t.order_id = m.id
#             WHERE m.customer_id = %s
#             ORDER BY m.created_at
#         """

#         df = pd.read_sql(query, engine, params=[cust_id])

#         if df.empty:
#             return jsonify({
#                 'status': 'error',
#                 'message': f"No orders found for customer {cust_id}"
#             }), 404

#         # -------------------------------
#         # Feature Encoding
#         # -------------------------------

#         # Encode customer_id (same logic as training)
#         df['cust_encoded'] = df['customer_id'].apply(
#             lambda x: int(int(x[:8], 16) % 1000)
#         )

#         # Encode food (must match training encoding)
#         df['food_encoded'] = df['item_id']

#         # Take last order only
#         last_order = df.iloc[-1:]

#         # IMPORTANT: Columns must match training
#         features = last_order[['cust_encoded', 'food_encoded', 'subtotal']]

#         # Scale features
#         scaled_features = scaler.transform(features)

#         # Get recommendations
#         distances, indices = model.kneighbors(
#             scaled_features, n_neighbors=6
#         )

#         recommendations = []

#         for i in range(1, len(indices.flatten())):
#             food_code = df.iloc[indices.flatten()[i]]['food_encoded']
#             food_name = le_food.inverse_transform(
#                 [int(food_code)]
#             )[0]
#             recommendations.append(food_name)

#         # Remove duplicates
#         unique_recommendations = list(dict.fromkeys(recommendations))

#         return jsonify({
#             'status': 'success',
#             'customer_id': cust_id,
#             'recommendations': unique_recommendations
#         })

#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 500


# # -------------------------------
# # Run Server
# # -------------------------------
# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000, debug=True)