import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def neural_network(df_images, df_labels):
    classifier = MLPClassifier(solver="sgd", verbose=True, early_stopping=False, max_iter=2000)
    classifier.hidden_layer_sizes = (2500,)
    classifier.activation = "logistic"
    classifier.learning_rate_init = 0.0001
    # classifier.learning_rate = 'adaptive'

    # splitting the dataset
    train_X, test_X, train_y, test_y = train_test_split(df_images, df_labels, test_size=0.2, random_state=1)
    train_X = train_X.values
    test_X = test_X.values
    train_y = train_y.values.ravel()
    test_y = test_y.values.ravel()

    # fit the model
    classifier.fit(train_X, train_y)

    print("\n\nTraining set score: %f" % classifier.score(train_X, train_y))
    print("Test set score: %f" % classifier.score(test_X, test_y))

    # plotting the loss curve vs iterations
    loss_values = classifier.loss_curve_
    plt.plot(loss_values)
    plt.title('loss vs iterations for : ' + str(classifier.learning_rate_init) + ' learning rate')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()

    # save the model to disk
    current_date_time = time.strftime("%d/%m/%Y") + '_' + time.strftime("%H:%M:%S")
    filename = 'finalized_model_' + current_date_time + '.sav'
    # filename = 'finalized_model.sav'
    joblib.dump(classifier, filename.replace('/', '_'))
    print('model saved in file ' + filename)

    # printing train set classwise accuracy
    classwise_accuracy(classifier, test_X, test_y, train_X, train_y)

    compute_confusion_matrix(classifier, df_labels, test_X, test_y)
    # calculating accuracy for validation
    # print('--------------------------calculating accuracy---------------------------------------------')
    # scores = cross_val_score(classifier, train_X, train_y, cv=5)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def compute_confusion_matrix(classifier, df_labels, test_X, test_y):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_y, classifier.predict(test_X))
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(df_labels.values),
                          title='Confusion matrix, without normalization')
    plt.show()
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=np.unique(df_labels.values), normalize=True,
                          title='Normalized confusion matrix')
    plt.show()


def classwise_accuracy(classifier, test_X, test_y, train_X, train_y):
    df_train_labels = pd.DataFrame(train_y)
    df_train_labels.columns = ['label']
    df_train_labels["success"] = (train_y == classifier.predict(train_X))
    print('\n-------------------Train-set accuracy metrics--------------------\n')
    for name, group in df_train_labels.groupby('label'):
        frac = sum(group["success"]) / len(group)
        print("Success rate for labeling class %i was %f " % (name, frac))
    # printing test set classwise accuracy
    df_test_labels = pd.DataFrame(test_y)
    df_test_labels.columns = ['label']
    df_test_labels["success"] = (test_y == classifier.predict(test_X))
    print('\n-------------------Test-set accuracy metrics--------------------\n')
    for name, group in df_test_labels.groupby('label'):
        frac = sum(group["success"]) / len(group)
        print("Success rate for labeling class %i was %f " % (name, frac))
