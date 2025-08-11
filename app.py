import os

import numpy as np
from flask import Flask, request, jsonify, render_template
from data.vectordb import vectordb
import re
import pandas as pd
from modules.image_recog import image_recog
from modules.cnn_model import cnn_model

df = pd.read_csv('data/cleaned_data.csv')
product_df = pd.read_csv('data/StockCode.csv').drop_duplicates().fillna('')

vdb = vectordb()
app = Flask(__name__)
product_names=['6 RIBBONS RUSTIC CHARM', 'ALARM CLOCK BAKELIKE RED', 'CHOCOLATE HOT WATER BOTTLE', 'JUMBO STORAGE BAG SUKI', 'LUNCH BAG PINK POLKADOT', 'LUNCH BAG WOODLAND', 'REGENCY CAKESTAND 3 TIER', 'RETROSPOT TEA SET CERAMIC 11 PC', 'REX CASHCARRY JUMBO SHOPPER']
cnn = cnn_model()
model = cnn.create_model()
model.load_weights("cnn_model\product_classifier.h5")
model_path = r"C:\Users\user\.cache\huggingface\hub\models--microsoft--trocr-large-handwritten\snapshots\e68501f437cd2587ae5d68ee457964cac824ddee"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/ocr-query', methods=['POST', 'GET'])
def ocr_query():
    try:
        if 'file' not in request.files:
            return render_template('hand_query.html')

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img_recog = image_recog(model_path)
        result = img_recog.read_image_text(file_path, df) 
        matches = result["products"].get("matches", [])
        response = result["final"]
        matches = process_match(matches)
        return render_template('hand_query.html', matches=matches, text=response)
    except Exception as ex:
        return render_template('hand_query.html', error=ex)
    

@app.route('/image-product-search', methods=['POST', 'GET'])
def image_product_search():
    if request.method == "POST":
        try:
            UPLOAD_FOLDER = 'static/uploads'
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            image = request.files['image']
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            arr = cnn.prepare_image(image_path)
            prediction = model.predict(arr)  # Call model's predict method
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_class = product_names[predicted_index]  # Must define this in cnn_model()
            df['Description'] = df['Description'].astype(str).str.strip().str.lower()  # Normalize descriptions
            df.fillna('', inplace=True)
            product_description = predicted_class.replace("_", " ").title()
            result = vdb.recommend_products(product_description, top_k=5)
            matches = result.get("matches", [])

            matches = process_match(matches)
            return render_template("cnn.html",
                               product_description=product_description,
                               predicted_class=predicted_class,
                               matching_products=matches)
        except Exception as ex:
                  render_template("cnn.html", error=ex)

    return render_template("cnn.html")

@app.route('/', methods=['GET', 'POST'])
def recommend_view():
    matches = []
    query = ""
    if request.method == 'POST':
        query = request.form.get("query", "").strip()
        is_valid, error = validate_search_input(query=query)
        if not is_valid:
             return render_template("recommend_product.html", error=error, matches=[])
        if not query:
            return render_template("recommend_product.html", error="Query is required", matches=[])
        try:
            df['Description'] = df['Description'].astype(str).str.strip().str.lower()  # Normalize descriptions
            df.fillna('', inplace=True)
            result = vdb.recommend_products(query, top_k=5)
            matches = result.get("matches", [])
            matches = process_match(matches)
        except Exception as ex:
            return render_template("recommend_product.html", error=ex)

        
        

    return render_template("recommend_product.html", matches=matches, query=query)

def process_match(matches):
    for match in matches:
           desc = match["description"].replace("$", "").strip().lower()
           match["description"] = desc  # Update cleaned version

    # Get matching rows
           row = df[df['Description'].str.strip().str.lower() == desc]
           print(row)
           if not row.empty:
            row = row.iloc[0]  # Take the first match
            match["UnitPrice"] = row["UnitPrice"]
            match["InvoiceNo"] = row["InvoiceNo"]
            match["Country"] = row["Cleaned_Country"]
            match["Quantity"] = row["Quantity"]
            match["InvoiceDate"] = row["InvoiceDate"]
            match["CustomerID"] = row["CustomerID"]
            match["StockCode"] = row["StockCode"]
           else:
        # If no match found, fill with fallback values
            match["UnitPrice"] = "N/A"
            match["InvoiceNo"] = "N/A"
            match["Country"] = "N/A"
            match["Quantity"] = 0
            match["InvoiceDate"] = "N/A"
            match["CustomerID"] = "N/A"
            match["StockCode"] = "N/A"
    return matches
def validate_search_input(query):
    if not isinstance(query,str):
        return False, "Query must be string"
    if len(query) == 0 or len(query) > 100:
        return False, 'Query length must be between 1 and 100'
    if not re.match(r'^[a-zA-Z0-9\s.,!?;:\'"\-_()]+$', query):
        print(query)
        return False, 'Query contains invalid character'
    return True, ''

if __name__ == '__main__':
    app.run(debug=True)
