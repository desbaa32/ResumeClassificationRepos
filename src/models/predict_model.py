from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical
#model prediction and classification report
def prediction_model(model,tfidf_test,y_test):
    #Predict on validation data
    prediction__rf = model.predict(tfidf_test)
    #Print classification report for validation data
    print("Classification Report (Validation Data):\n")
    print(classification_report(y_test, prediction__rf))
    accuracy=accuracy_score(y_test, prediction__rf)
    print("Accuracy is : ", accuracy)
def prediction_model_rnn(model,tfidf_test_arrays,y_test_label):
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(tfidf_test_arrays, y_test_label)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")
