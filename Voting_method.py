import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import classification_report


data = pd.read_excel(input_file, sheet_name='Sheet1')

corpus = []
for sentences in data['Comment']:
    corpus.append(sentences)
 
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=0.01, max_features=1000) 
vectors = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df


X_train, X_test, y_train, y_test = train_test_split(df, data['Sentiment'], test_size=0.2, random_state=0)

logistic_reg = LogisticRegression(random_state=0, max_iter=1000)
logistic_reg.fit(X_train, y_train)
coefficients1 = logistic_reg.coef_

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
coefficients2 = lda.coef_

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
coefficients3 = svm.coef_

ridge_classifier = RidgeClassifier()
ridge_classifier.fit(X_train, y_train)
coefficients4 = ridge_classifier.coef_

pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)
coefficients5 = pls.coef_

classifiers = {
    'Logistic Regression': logistic_reg,
    'Linear Discriminant Analysis': lda,
    'SVM': svm,
    'PLS Regression': pls,
    'Ridge Classifier': ridge_classifier
} 

coefficients = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    if hasattr(clf, 'coef_'):
        coefficients[name] = clf.coef_[0]
    elif hasattr(clf, 'dual_coef_'):  # For SVM
        coefficients[name] = clf.dual_coef_[0]

votes_df = pd.DataFrame(index=df.columns, columns=classifiers.keys())

for feature in df.columns:
    for clf_name, coef in coefficients.items():
        if coef[df.columns.get_loc(feature)] > 0:
            votes_df.loc[feature, clf_name] = "Positive"
        else:
            votes_df.loc[feature, clf_name] = "Negative"

votes_df['Final Vote'] = votes_df.mode(axis=1)[0]

print("Votes based on coefficients:")
votes_df






