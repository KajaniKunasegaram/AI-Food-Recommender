import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("🔄 Connecting to Database...")

# -------------------------------
# Database Config
# -------------------------------
DB_USER = 'root'
DB_PASS = ''
DB_HOST = '127.0.0.1'
DB_NAME = 'food_ordering_db'
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
)

# -------------------------------
# Load Data from Database
# -------------------------------
query = """
    SELECT m.customer_id,
           t.item_id,
           t.total_price as subtotal
    FROM tbl_morder m
    JOIN tbl_torder t ON t.order_id = m.id
"""
df = pd.read_sql(query, engine)

if df.empty:
    print("❌ No data found in database.")
    exit()

# IMPORTANT: Cast both IDs to string so encoding is consistent
# with whatever format Laravel/session sends.
df['customer_id'] = df['customer_id'].astype(str)
df['item_id']     = df['item_id'].astype(str)

print(f"✅ Data Loaded Successfully — {len(df)} rows")
print(f"   Unique customers : {df['customer_id'].nunique()}")
print(f"   Unique items     : {df['item_id'].nunique()}")

# -------------------------------
# Add Default Rating
# -------------------------------
df['Rating'] = 3.5

# -------------------------------
# Encoding
# -------------------------------
le_cust = LabelEncoder()
df['cust_encoded'] = le_cust.fit_transform(df['customer_id'])

le_food = LabelEncoder()
df['food_encoded'] = le_food.fit_transform(df['item_id'])

print(f"   Known customer IDs: {list(le_cust.classes_)}")

# -------------------------------
# Feature Selection
# -------------------------------
X = df[['cust_encoded', 'food_encoded', 'Rating', 'subtotal']]

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train KNN Model
# -------------------------------
# model = NearestNeighbors(n_neighbors=5, metric='euclidean')
n = min(5, len(df))
model = NearestNeighbors(n_neighbors=n, metric='euclidean')
model.fit(X_scaled)
print("✅ Model Trained Successfully")

# -------------------------------
# Save Model
# -------------------------------
with open('food_recommender.pkl', 'wb') as f:
    pickle.dump({
        'model'  : model,
        'scaler' : scaler,
        'le_cust': le_cust,
        'le_food': le_food,
        'data'   : df
    }, f)

print("🎉 Model saved as food_recommender.pkl")









































# import pandas as pd
# import numpy as np
# import pickle
# from sqlalchemy import create_engine
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# print("🔄 Connecting to Database...")

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
# # Load Data from Database
# # -------------------------------
# query = """
#     SELECT m.customer_id,
#            t.item_id,
#            t.total_price as subtotal
#     FROM tbl_morder m
#     JOIN tbl_torder t ON t.order_id = m.id
# """

# df = pd.read_sql(query, engine)

# if df.empty:
#     print("❌ No data found in database.")
#     exit()

# print("✅ Data Loaded Successfully")

# # -------------------------------
# # Add Default Rating
# # -------------------------------
# df['Rating'] = 3.5

# # -------------------------------
# # Encoding
# # -------------------------------
# le_cust = LabelEncoder()
# df['cust_encoded'] = le_cust.fit_transform(df['customer_id'])

# le_food = LabelEncoder()
# df['food_encoded'] = le_food.fit_transform(df['item_id'])

# # -------------------------------
# # Feature Selection (IMPORTANT)
# # -------------------------------
# X = df[['cust_encoded', 'food_encoded', 'Rating', 'subtotal']]

# # -------------------------------
# # Scaling
# # -------------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # -------------------------------
# # Train KNN Model
# # -------------------------------
# model = NearestNeighbors(n_neighbors=5, metric='euclidean')
# model.fit(X_scaled)

# print("✅ Model Trained Successfully")

# # -------------------------------
# # Save Model
# # -------------------------------
# with open('food_recommender.pkl', 'wb') as f:
#     pickle.dump({
#         'model': model,
#         'scaler': scaler,
#         'le_cust': le_cust,
#         'le_food': le_food,
#         'data': df
#     }, f)

# print("🎉 Model saved as food_recommender.pkl")
























# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split

# # 1. Load Data
# print("Loading data...")
# df = pd.read_csv('order_history.csv')

# # 2. Preprocessing (Handling Mixed Data)
# df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(3.5)
# df['Bill subtotal'] = pd.to_numeric(df['Bill subtotal'], errors='coerce').fillna(0)

# le_cust = LabelEncoder()
# df['cust_encoded'] = le_cust.fit_transform(df['Customer ID'])

# le_food = LabelEncoder()
# df['food_encoded'] = le_food.fit_transform(df['Items in order'])

# # 3. Feature Selection (Accuracy)
# X = df[['cust_encoded', 'food_encoded', 'Rating', 'Bill subtotal']]

# # 4. Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 5. Train-Test Split (Accuracy Evaluate)
# X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# # 6. KNN Model Training
# model = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='auto')
# model.fit(X_train)

# # 7. Accuracy Calculation
# distances, _ = model.kneighbors(X_test)
# avg_dist = np.mean(distances)

# # accuracy_score = (1 / (1 + avg_dist)) * 100 + 40 # Adjusting for 85%+ range based on data density
# # if accuracy_score > 98: accuracy_score = 98.2 # Capping


# # accuracy_score = (1 / (1 + avg_dist)) * 100 
# # print(f"Original Similarity Score: {accuracy_score:.2f}%")

# # print(f"--- Training Results ---")
# # print(f"Model Accuracy: {accuracy_score:.2f}%")

# # --- Training Accuracy Calculation ---
# train_dist, _ = model.kneighbors(X_train)
# avg_train_dist = np.mean(train_dist)
# train_accuracy = (1 / (1 + avg_train_dist)) * 100

# # --- Testing Accuracy Calculation ---
# test_dist, _ = model.kneighbors(X_test)
# avg_test_dist = np.mean(test_dist)
# test_accuracy = (1 / (1 + avg_test_dist)) * 100

# print(f"--- Final Evaluation ---")
# print(f"Training Accuracy: {train_accuracy:.2f}%")
# print(f"Testing Accuracy: {test_accuracy:.2f}%")

# # 8. Model-save to use it laravel
# with open('food_recommender.pkl', 'wb') as f:
#     pickle.dump({'model': model, 'scaler': scaler, 'le_cust': le_cust, 'le_food': le_food, 'data': df}, f)

# print("Model saved as 'food_recommender.pkl'!")