import preprocess
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, Y = preprocess.read_and_preprocess()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)

clf = RandomForestClassifier()
clf.fit(X_train, Y_train.astype(int))

Y_pred = clf.predict(X_test)

print "Random Forest successfully trained"
print "Prediction Vector :", Y_pred

acc_score = accuracy_score(Y_test, Y_pred)
print "Accuracy : {}".format(acc_score)
