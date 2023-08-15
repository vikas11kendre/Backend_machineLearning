import pandas as pd
import nltk
from flask_cors import CORS
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import flask
from flask import Flask, request, jsonify
#from ML_Model.Test import preprocess_text  # Import your ML model related code

app = Flask(__name__)
CORS(app)
modify_api_data = pd.read_csv(r"./modify_commands_exported_data.csv")
query_api_data = pd.read_csv(r"./query_commands_exported_data.csv")
api_data=pd.concat([modify_api_data,query_api_data])
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())

    # Remove punctuation and non-alphanumeric characters
    words = [re.sub(r"[^a-zA-Z0-9]", "", word) for word in words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)
api_data = api_data[['api_name', 'api_description']]
api_data["api_description"] = api_data["api_description"].apply(lambda x:x[12:]).apply(lambda x: x.replace("\n"," ")).apply(lambda x: x.replace("\r"," "))
api_data["Processed_Description"] = api_data["api_description"].apply(preprocess_text)
labeled_data = []
for i,j in zip(api_data["api_name"],api_data["Processed_Description"]):
  labeled_data.append((j,i))
text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", LinearSVC())
])

# Separate features (descriptions) and labels (API names) from the labeled data
features, labels = zip(*labeled_data)

# Train the model
text_clf.fit(features, labels)
def recommend_api(user_input):
        processed_input = preprocess_text(user_input)
        predicted_api = text_clf.predict([processed_input])[0]
        return predicted_api
def chatbot_interaction():
    print("Chatbot: Hi! How can I assist you today?")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        recommended_api = recommend_api(user_input)
        print(f"Chatbot: I recommend using the '{recommended_api}' API for your task.")

#chatbot_interaction()



@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        user_input = request.json['user_input']
        processed_input = preprocess_text(user_input)
        print(processed_input)
        predicted_api = text_clf.predict([processed_input])[0]
        return jsonify({'predicted_api': predicted_api})
        return 'sucess'
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/test', methods=['GET'])
def test_server():
    return jsonify({'error': 'Server is up and running.'})
# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
    print("Server started")