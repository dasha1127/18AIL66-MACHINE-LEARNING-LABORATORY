from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import datasets

# load the iris dataset
dataset = datasets.load_iris()
x = dataset.data  # Use 'data' instead of indexing with column names
y = dataset.target  # Use 'target' instead of 'variety'

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svc_model = SVC()
svc_model.fit(x_train, y_train)
svc_pred = svc_model.predict(x_test)
svc_acc_score = accuracy_score(y_test, svc_pred)

print("SVC Classification Model Accuracy:", svc_acc_score)
print(svc_pred)
print(classification_report(y_test, svc_pred))
print(confusion_matrix(y_test, svc_pred))

