import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from preprocessing import *
from sklearn.neural_network import MLPClassifier

parent_directory = '/media/vaneesh/Data/Datasets/German/training/training_5_classes/'
#parent_directory = '/media/vaneesh/Data/Datasets/German/training/training_5_classes (copy)/'
out_dir = '/media/vaneesh/Data/Datasets/German/training/training_5_classes/34/'

df_images, df_labels = make_dataframe_from_images(parent_directory)

classifier = MLPClassifier(solver="sgd", verbose=True, early_stopping=False)

classifier.hidden_layer_sizes = (60,)
classifier.activation = "logistic"
classifier.learning_rate_init = 0.0003
train_X, test_X, train_y, test_y = train_test_split(df_images, df_labels, test_size=0.2, random_state=1)

train_X = train_X.values
test_X = test_X.values

train_y = train_y.values.ravel()
test_y = test_y.values.ravel()

classifier.fit(train_X, train_y)

print("\n\nTraining set score: %f" % classifier.score(train_X, train_y))
print("Test set score: %f" % classifier.score(test_X, test_y))

loss_values = classifier.loss_curve_
plt.plot(loss_values)
plt.title('loss vs iterations for : '+str(classifier.learning_rate_init) + ' learning rate')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

df_train_labels = pd.DataFrame(train_y)
df_train_labels.columns =['label']
df_train_labels["success"] = (train_y == classifier.predict(train_X))

print('\n-------------------Train-set accuracy metrics--------------------\n')

for name, group in df_train_labels.groupby('label'):
    frac = sum(group["success"])/len(group)
    print("Success rate for labeling class %i was %f " %(name, frac))


df_test_labels = pd.DataFrame(test_y)
df_test_labels.columns =['label']
df_test_labels["success"] = (test_y == classifier.predict(test_X))

print('\n-------------------Test-set accuracy metrics--------------------\n')

for name, group in df_test_labels.groupby('label'):
    frac = sum(group["success"])/len(group)
    print("Success rate for labeling class %i was %f " %(name, frac))

