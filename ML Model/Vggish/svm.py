from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

data_path = "dataval.json"

def load_data(path):
    with open(path, 'r') as fp:
        data = json.load(fp)

    x = np.array(data["features"])
    y = np.array(data["labels"])

    return x,y

def split_data(test_size, validation_size):

    X, Y = load_data(data_path)
    print(X.shape, Y.shape)

    # create training/testing split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # create training/validation split
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=validation_size)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def make_SVM():

    # create an SVM
    clf = SVC(kernel='rbf')

    return clf

if __name__ == "__main__":

    x_train, x_validation, x_test, y_train, y_validation, y_test = split_data(0.25, 0.2)
    print(x_train.shape)

    model = make_SVM()

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_val_pred = model.predict(x_validation)  # Assuming you have validation data

    # Accuracy Calculations
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    val_accuracy = accuracy_score(y_validation, y_val_pred)  # If available

    print("\nTraining Results:")
    print("Accuracy:", train_accuracy)

    print("\nTesting Results:")
    print("Accuracy:", test_accuracy)

    print("\nValidation Results:")
    print("Accuracy:", val_accuracy)

    genre_labels = {
        0: 'Pop', 1: 'Metal', 2: 'Disco', 3: 'Blues', 4: 'Reggae',
        5: 'Classical', 6: 'Rock', 7: 'Hip Hop', 8: 'Country', 9: 'Jazz'}

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_score = model.decision_function(x_test)

    # Precision-Recall curve and average precision for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(genre_labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    # Plot Precision-Recall curves with genre labels
    plt.figure(figsize=(12, 8))  # Increased figure size for better readability
    for i in range(len(genre_labels)):
        plt.plot(recall[i], precision[i], label=f'{genre_labels[i]} (AP = {average_precision[i]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve by Genre')
    plt.legend(loc="lower left")
    plt.show()

print()








