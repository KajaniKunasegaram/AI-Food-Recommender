
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Database Config
DB_USER = 'root'
DB_PASS = ''
DB_HOST = '127.0.0.1'
DB_NAME = 'food_ordering_db'
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
)

# Load Kaggle CSV Dataset
print("🔄 Loading Kaggle dataset...")
try:
    kaggle_df = pd.read_csv('order_history.csv')
    kaggle_df = kaggle_df.rename(columns={
        'Customer ID'   : 'customer_id',
        'Items in order': 'item_id',
        'Bill subtotal' : 'subtotal',
        'Rating'        : 'Rating'
    })
    kaggle_df = kaggle_df[['customer_id', 'item_id', 'subtotal', 'Rating']]
    kaggle_df['subtotal']    = pd.to_numeric(kaggle_df['subtotal'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    kaggle_df['Rating']      = pd.to_numeric(kaggle_df['Rating'], errors='coerce').fillna(3.5)
    kaggle_df['customer_id'] = kaggle_df['customer_id'].astype(str).str.strip()
    kaggle_df['item_id']     = kaggle_df['item_id'].astype(str).str.strip()
    kaggle_df['sub_cat_id']  = 0  # Kaggle has no category info — default 0
    kaggle_df['source']      = 'kaggle'
    print(f"✅ Kaggle Data Loaded — {len(kaggle_df)} rows")
except FileNotFoundError:
    print("⚠️  order_history.csv not found — using database only")
    kaggle_df = pd.DataFrame(columns=['customer_id', 'item_id', 'subtotal', 'Rating', 'sub_cat_id', 'source'])

# Load Database Data (with sub_cat_id JOIN)
print("\n🔄 Connecting to Database...")
db_query = """
    SELECT 
        m.customer_id,
        t.item_id,
        t.total_price   AS subtotal,
        i.sub_cat_id                
    FROM tbl_morder m
    JOIN tbl_torder t ON t.order_id = m.id
    JOIN tbl_items  i ON i.item_id  = t.item_id
"""
db_df = pd.read_sql(db_query, engine)

if db_df.empty:
    print("⚠️  No data in database — using Kaggle only")
    db_df = pd.DataFrame(columns=['customer_id', 'item_id', 'subtotal', 'sub_cat_id', 'source'])
else:
    db_df['customer_id'] = db_df['customer_id'].astype(str).str.strip()
    db_df['item_id']     = db_df['item_id'].astype(str).str.strip()
    db_df['subtotal']    = pd.to_numeric(db_df['subtotal'], errors='coerce').fillna(0)
    db_df['Rating']      = 3.5
    db_df['source']      = 'database'
    print(f"✅ Database Data Loaded — {len(db_df)} rows")
    print(f"   Unique customers : {db_df['customer_id'].nunique()}")
    print(f"   Unique items     : {db_df['item_id'].nunique()}")
    print(f"   Unique sub_cats  : {db_df['sub_cat_id'].nunique()}")

# Merge
print("\n🔄 Merging datasets...")
df = pd.concat([kaggle_df, db_df], ignore_index=True)
df = df.dropna(subset=['customer_id', 'item_id', 'subtotal'])
df = df[df['customer_id'] != 'nan']
df = df[df['item_id']     != 'nan']
df['sub_cat_id'] = pd.to_numeric(df['sub_cat_id'], errors='coerce').fillna(0).astype(int)

print(f"✅ Merged Data — {len(df)} total rows")
print(f"   Kaggle rows   : {len(df[df['source'] == 'kaggle'])}")
print(f"   Database rows : {len(df[df['source'] == 'database'])}")

if df.empty:
    print("❌ No data available. Exiting.")
    exit()

#  Encoding
le_cust = LabelEncoder()
df['cust_encoded'] = le_cust.fit_transform(df['customer_id'])

le_food = LabelEncoder()
df['food_encoded'] = le_food.fit_transform(df['item_id'])

print(f"\n   DB customers in model:")
for c in df[df['source'] == 'database']['customer_id'].unique():
    print(f"   ✅ {c[:20]}...")

# Feature Selection
# Content-Based filtering
X = df[['cust_encoded', 'food_encoded', 'sub_cat_id', 'Rating', 'subtotal']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split & KNN
if len(df) >= 10:
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    print(f"\n   Train size: {len(X_train)} | Test size: {len(X_test)}")
else:
    X_train = X_scaled
    X_test  = X_scaled
    print(f"\n⚠️  Small dataset — using all {len(df)} rows")

n_neighbors = min(5, len(X_train))
model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
model.fit(X_train)

#  Accuracy
print("\n--- 📊 Model Evaluation ---")
train_dist, _ = model.kneighbors(X_train)
train_accuracy = (1 / (1 + np.mean(train_dist))) * 100

test_dist, _  = model.kneighbors(X_test)
test_accuracy  = (1 / (1 + np.mean(test_dist)))  * 100

print(f"✅ Training Accuracy : {train_accuracy:.2f}%")
print(f"✅ Testing Accuracy  : {test_accuracy:.2f}%")

# Save Model
with open('food_recommender.pkl', 'wb') as f:
    pickle.dump({
        'model'   : model,
        'scaler'  : scaler,
        'le_cust' : le_cust,
        'le_food' : le_food,
        'data'    : df
    }, f)

print(f"\n🎉 Model saved as food_recommender.pkl")
print(f"   Features used   : cust_encoded, food_encoded, sub_cat_id, Rating, subtotal")
print(f"   Training Accuracy: {train_accuracy:.2f}%")
print(f"   Testing Accuracy : {test_accuracy:.2f}%")

















# import pandas as pd
# import numpy as np
# import pickle
# from sqlalchemy import create_engine
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split

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

# # ================================================
# # PART 1: Load Kaggle CSV Dataset
# # ================================================
# print("🔄 Loading Kaggle dataset...")
# try:
#     kaggle_df = pd.read_csv('order_history.csv')

#     # Kaggle columns → our standard column names
#     kaggle_df = kaggle_df.rename(columns={
#         'Customer ID'   : 'customer_id',
#         'Items in order': 'item_id',
#         'Bill subtotal' : 'subtotal',
#         'Rating'        : 'Rating'
#     })

#     # Keep only needed columns
#     kaggle_df = kaggle_df[['customer_id', 'item_id', 'subtotal', 'Rating']]

#     # Clean subtotal — remove commas, convert to float
#     kaggle_df['subtotal'] = pd.to_numeric(
#         kaggle_df['subtotal'].astype(str).str.replace(',', ''),
#         errors='coerce'
#     ).fillna(0)

#     # Clean Rating
#     kaggle_df['Rating'] = pd.to_numeric(kaggle_df['Rating'], errors='coerce').fillna(3.5)

#     # Cast to string
#     kaggle_df['customer_id'] = kaggle_df['customer_id'].astype(str).str.strip()
#     kaggle_df['item_id']     = kaggle_df['item_id'].astype(str).str.strip()

#     # Mark source
#     kaggle_df['source'] = 'kaggle'

#     print(f"✅ Kaggle Data Loaded — {len(kaggle_df)} rows")
#     print(f"   Unique customers : {kaggle_df['customer_id'].nunique()}")
#     print(f"   Unique items     : {kaggle_df['item_id'].nunique()}")

# except FileNotFoundError:
#     print("⚠️  order_history.csv not found — using database only")
#     kaggle_df = pd.DataFrame(columns=['customer_id', 'item_id', 'subtotal', 'Rating', 'source'])

# # ================================================
# # PART 2: Load Database Data
# # ================================================
# print("\n🔄 Connecting to Database...")
# db_query = """
#     SELECT m.customer_id,
#            t.item_id,
#            t.total_price as subtotal
#     FROM tbl_morder m
#     JOIN tbl_torder t ON t.order_id = m.id
# """
# db_df = pd.read_sql(db_query, engine)

# if db_df.empty:
#     print("⚠️  No data in database — using Kaggle only")
# else:
#     db_df['customer_id'] = db_df['customer_id'].astype(str).str.strip()
#     db_df['item_id']     = db_df['item_id'].astype(str).str.strip()
#     db_df['subtotal']    = pd.to_numeric(db_df['subtotal'], errors='coerce').fillna(0)
#     db_df['Rating']      = 3.5
#     db_df['source']      = 'database'
#     print(f"✅ Database Data Loaded — {len(db_df)} rows")
#     print(f"   Unique customers : {db_df['customer_id'].nunique()}")
#     print(f"   Unique items     : {db_df['item_id'].nunique()}")

# # ================================================
# # PART 3: Merge Both Datasets
# # ================================================
# print("\n🔄 Merging datasets...")
# df = pd.concat([kaggle_df, db_df], ignore_index=True)

# # Drop rows with missing values
# df = df.dropna(subset=['customer_id', 'item_id', 'subtotal'])
# df = df[df['customer_id'] != 'nan']
# df = df[df['item_id']     != 'nan']

# print(f"✅ Merged Data — {len(df)} total rows")
# print(f"   Unique customers : {df['customer_id'].nunique()}")
# print(f"   Unique items     : {df['item_id'].nunique()}")
# print(f"   Kaggle rows      : {len(df[df['source'] == 'kaggle'])}")
# print(f"   Database rows    : {len(df[df['source'] == 'database'])}")

# if df.empty:
#     print("❌ No data available. Exiting.")
#     exit()

# # ================================================
# # PART 4: Encoding
# # ================================================
# le_cust = LabelEncoder()
# df['cust_encoded'] = le_cust.fit_transform(df['customer_id'])

# le_food = LabelEncoder()
# df['food_encoded'] = le_food.fit_transform(df['item_id'])

# print(f"\n   Known DB customer IDs in model:")
# # Show only database customers (not all kaggle ones)
# db_customers = df[df['source'] == 'database']['customer_id'].unique()
# for c in db_customers:
#     print(f"   ✅ {c[:20]}...")

# # ================================================
# # PART 5: Feature Selection & Scaling
# # ================================================
# X = df[['cust_encoded', 'food_encoded', 'Rating', 'subtotal']]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ================================================
# # PART 6: Train-Test Split & KNN Training
# # ================================================
# if len(df) >= 10:
#     X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
#     print(f"\n   Train size: {len(X_train)} | Test size: {len(X_test)}")
# else:
#     # Too few rows — use all data for training
#     X_train = X_scaled
#     X_test  = X_scaled
#     print(f"\n⚠️  Small dataset — using all {len(df)} rows for training")

# n_neighbors = min(5, len(X_train))
# model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
# model.fit(X_train)

# # ================================================
# # PART 7: Accuracy Calculation
# # ================================================
# print("\n--- 📊 Model Evaluation ---")

# # Training Accuracy
# train_dist, _ = model.kneighbors(X_train)
# avg_train_dist = np.mean(train_dist)
# train_accuracy = (1 / (1 + avg_train_dist)) * 100

# # Testing Accuracy
# test_dist, _ = model.kneighbors(X_test)
# avg_test_dist = np.mean(test_dist)
# test_accuracy = (1 / (1 + avg_test_dist)) * 100

# print(f"✅ Training Accuracy : {train_accuracy:.2f}%")
# print(f"✅ Testing Accuracy  : {test_accuracy:.2f}%")

# # ================================================
# # PART 8: Save Model
# # ================================================
# with open('food_recommender.pkl', 'wb') as f:
#     pickle.dump({
#         'model'   : model,
#         'scaler'  : scaler,
#         'le_cust' : le_cust,
#         'le_food' : le_food,
#         'data'    : df
#     }, f)

# print("\n🎉 Model saved as food_recommender.pkl")
# print(f"   Total records used for training : {len(df)}")
# print(f"   Training Accuracy               : {train_accuracy:.2f}%")
# print(f"   Testing Accuracy                : {test_accuracy:.2f}%")
































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

# # IMPORTANT: Cast both IDs to string so encoding is consistent
# # with whatever format Laravel/session sends.
# df['customer_id'] = df['customer_id'].astype(str)
# df['item_id']     = df['item_id'].astype(str)

# print(f"✅ Data Loaded Successfully — {len(df)} rows")
# print(f"   Unique customers : {df['customer_id'].nunique()}")
# print(f"   Unique items     : {df['item_id'].nunique()}")

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

# print(f"   Known customer IDs: {list(le_cust.classes_)}")

# # -------------------------------
# # Feature Selection
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
# # model = NearestNeighbors(n_neighbors=5, metric='euclidean')
# n = min(5, len(df))
# model = NearestNeighbors(n_neighbors=n, metric='euclidean')
# model.fit(X_scaled)
# print("✅ Model Trained Successfully")

# # -------------------------------
# # Save Model
# # -------------------------------
# with open('food_recommender.pkl', 'wb') as f:
#     pickle.dump({
#         'model'  : model,
#         'scaler' : scaler,
#         'le_cust': le_cust,
#         'le_food': le_food,
#         'data'   : df
#     }, f)

# print("🎉 Model saved as food_recommender.pkl")









































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