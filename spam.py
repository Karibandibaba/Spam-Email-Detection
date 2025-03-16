import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Streamlit app
def main():
    st.title("Email Spam Detection App")
    st.write("This app predicts whether a given message is Spam or Ham.")

    message = st.text_area("Enter your message:")

    if st.button("Predict"):
        if message.strip():
            transformed_msg = vectorizer.transform([message])
            prediction = model.predict(transformed_msg)[0]
            result = "Spam" if prediction == 1 else "Ham"
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter a valid message.")

    # Display model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"### Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
