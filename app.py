from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the review text from form (prevents KeyError)
        review_text = request.form.get('review', '').strip()

        # Validate input
        if not review_text:
            return render_template('index.html', prediction="Error: Please enter a review.")

        # Transform input text using TF-IDF
        transformed_text = vectorizer.transform([review_text])

        # Predict sentiment (0 = Negative, 1 = Positive)
        prediction = model.predict(transformed_text)[0]

        # Convert numerical prediction to a human-readable label
        sentiment_label = "Positive 😊" if prediction == 1 else "Negative 😞"

        return render_template('index.html', prediction=f"Sentiment: {sentiment_label}")

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
