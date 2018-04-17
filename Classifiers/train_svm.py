from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import preprocess

# get features and labels 
X,Y = preprocess.read_and_preprocess()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)

clf = svm.SVC()
clf = clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

print "SVM successfully trained"
print "prediction vector : " , Y_pred

acc_score = accuracy_score(Y_test, Y_pred)
print "Accuracy score : {}".format(acc_score)

