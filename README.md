AI‑Food‑Recommender

This is a Python Flask + Machine Learning + MySQL based Food Recommendation system you built. It provides APIs and recommendation logic to suggest foods based on user inputs and a trained model.

🧠 Features

✔️ Flask backend server
✔ Uses ML model to suggest food
✔ MySQL database integration
✔ REST routes for predictions

📌 Requirements

Make sure you have:

Python 3.13 installed
MySQL server running (e.g. via XAMPP/WAMP)
pip package installer
Git


🛠️ Setup & Run
1️⃣ Clone the repository
git clone https://github.com/KajaniKunasegaram/AI-Food-Recommender.git

cd AI-Food-Recommender


2️⃣ Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate


3️⃣ Install dependencies

pip install flask
pip install flask flask-cors pandas numpy sqlalchemy pymysql scikit-learn
pip freeze > requirements.txt


4️⃣ Setup MySQL database

Start MySQL using XAMPP/WAMP.
Go to phpMyAdmin (e.g. http://localhost/phpmyadmin/) and create a database, for example:
CREATE DATABASE food_ordering_db;


5️⃣ Install ML model dependencies
pip install scikit-learn


6️⃣ Run the server
python .\app.py

⚡ Running on http://127.0.0.1:5000/

Open in browser:

http://127.0.0.1:5000/
