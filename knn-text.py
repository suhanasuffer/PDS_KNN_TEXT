import warnings
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

df=pd.read_csv(r"C:\Users\91883\Downloads\Twitter_Data.csv")
print(df)

print(df.shape)

df=df.iloc[:100000,:]
df.dropna(inplace=True)
df.category.replace([-1.0, 0.0, 1.0], ['Negative', 'Neutral', 'Positive'], inplace=True)

X=df.clean_text.to_numpy()
y=df.category.to_numpy()

X_train=X[:60000,]
X_val=X[60000:80000,]
X_test=X[80000:, ]

y_train=y[:60000,]
y_val=y[60000:80000,]
y_test=y[80000:, ]

tfidf=TfidfVectorizer()
X_train_vect=tfidf.fit_transform(X_train)
X_val_vect=tfidf.transform(X_val)
X_test_vect=tfidf.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_vect, y_train)

knn_pred = knn.predict(X_test_vect)
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
accuracy=accuracy_score(y_test, knn_pred)
print(accuracy_score)

st.title("Sentiment Analysis App")
st.subheader("Enter text for sentiment analysis:")

st.sidebar.title("Customization")
n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 10, 5)

user_input = st.text_area("Enter text here:")
if user_input:
    user_input_vect = tfidf.transform([user_input])
    prediction = knn.predict(user_input_vect)[0]
    st.subheader("Sentiment:")
    st.write(prediction)

st.subheader("Model Evaluation:")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, knn_pred))
st.write("Classification Report:")
st.write(classification_report(y_test, knn_pred))
st.write(f"Accuracy: {accuracy:.2f}")

st.title("Sentiment Analysis App")
st.markdown("Enter a text and predict its sentiment.")

if not user_input:
    st.info("Please enter text for sentiment analysis.")
