from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import preprocess

X,Y = preprocess.read_and_preprocess()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)

print X_test.shape

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

print "Decision tree successfully trained"
print "prediction vector : " , y_pred

acc_score = accuracy_score(Y_test, y_pred)
print "Accuracy score : {}".format(acc_score)

