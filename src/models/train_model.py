from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical
# split data :test and train
def split_data(data):
    X_train,X_test,y_train,y_test = train_test_split(data['Cleaned_Resume'], data['Category'], random_state=42, test_size=0.15,stratify=resume_data['Category'])
    # #Print the sizes of the split datasets
    print("Train data size :",X_train.shape)
    print("Validation data size:",X_test.shape)
    print(X_train[X_train.isna()])# Check for NaN values
    print(X_train[X_train == ''])# Check for empty strings

def train_model(model,tfidf_train,y_train):
    RF = model
    RF.fit(tfidf_train,y_train)
def train_knn_model(tfidf_train,y_train):
    k = 24 # Number of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    # Train the KNN classifier
    knn_classifier.fit(tfidf_train,y_train)

def train_model_rnn(tfidf_train,tfidf_test):
    # Convert sparse matrices to dense arrays
    tfidf_train_arrays = tfidf_train.toarray()
    tfidf_test_arrays = tfidf_test.toarray()
    # Build a simple neural network model
    num_classes = 24
    y_train_label = to_categorical(y_train, num_classes=num_classes)
    y_test_label = to_categorical(y_test, num_classes=num_classes)
    # Build a more complex neural network model
    model = Sequential()
    model.add(Dense(1000, input_dim=tfidf_train_arrays.shape[1]))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Use softmax for multi-class classification
    # Compile the model with a lower learning rate
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    # Train the model with more epochs
    history = model.fit(tfidf_train_arrays, y_train_label, epochs=50, batch_size=32, validation_data=(tfidf_test_arrays, y_test_label))

