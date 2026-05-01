import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("💳 Credit Card Fraud Detection App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(data.head())

    if "Class" not in data.columns:
        st.error("Dataset must contain 'Class' column (0 = Normal, 1 = Fraud)")
    else:
        X = data.drop("Class", axis=1)
        y = data["Class"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        st.subheader("📈 Model Evaluation")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        st.success("✅ Model Trained Successfully!")

        # Manual prediction
        st.subheader("🔍 Test New Transaction")

        input_data = []
        for col in X.columns[:5]:  # sirf first 5 features input ke liye
            val = st.number_input(f"Enter {col}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            # Remaining features ko 0 se fill kar diya
            full_input = input_data + [0]*(X.shape[1]-5)
            prediction = model.predict([full_input])

            if prediction[0] == 1:
                st.error("🚨 Fraud Transaction Detected!")
            else:
                st.success("✅ Normal Transaction")