import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Read the CSV file
msg = pd.read_csv('document.csv', names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# Split the data into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(msg['message'], msg['labelnum'])

# Vectorize the text data
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

# Create a DataFrame from the training data matrix
df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names())
print(df[0:5])

# Train the Naive Bayes classifier
clf = MultinomialNB().fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)

# Print the predictions for each document
predictions = ['pos' if p == 1 else 'neg' for p in pred]
for doc, p in zip(Xtrain, predictions):
    print("%s -> %s" % (doc, p))

# Calculate and print accuracy metrics
print('Accuracy Metrics: \n')
print('Accuracy:', accuracy_score(ytest, pred))
print('Recall:', recall_score(ytest, pred))
print('Precision:', precision_score(ytest, pred))
print('Confusion Matrix:\n', confusion_matrix(ytest, pred))
