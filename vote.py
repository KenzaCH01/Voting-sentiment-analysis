import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.inspection import permutation_importance

class Voting(object):
    
    def __init__(self):    
        self.vectorizer = None
        self.X = None # comments: pandas dataframe
        self.y = None # score: pandas dataframe
        self.votes_df = None

    def vectorize(self, comments):
        corpus = []
        for sentences in comments:
            corpus.append(sentences)        
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=0.01, max_features=1000) 
        vectors = self.vectorizer.fit_transform(corpus)
        feature_names = self.vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        return df

    def fit(self, comments, score):
        comments = self.vectorize(comments)
        self.X = comments
        self.y = score
        
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=0, max_iter=1000),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'SVM': LinearSVC(),
            'PLS Regression': PLSRegression(n_components=3),
            'Ridge Classifier': RidgeClassifier()
        } 

        coefficients = {}
        for name, clf in classifiers.items():
            clf.fit(self.X, self.y)
            if name == 'PLS Regression': 
                coefficients[name] = clf.coef_
            else:
                coefficients[name] = clf.coef_[0]
        
        votes_df = pd.DataFrame(index=comments.columns, columns=classifiers.keys())
        
        for feature in comments.columns:
            for clf_name, coef in coefficients.items():
                if coef[comments.columns.get_loc(feature)] > 0:
                    votes_df.loc[feature, clf_name] = "Positive"
                else:
                    votes_df.loc[feature, clf_name] = "Negative"
        votes_df['Final Vote'] = votes_df.mode(axis=1)[0]
        self.votes_df = votes_df
        print("Votes based on coefficients:")
        return votes_df
    
    def predict(self):
        classifiers = {
            "RandomForest": RandomForestClassifier(random_state=0, max_depth=5),
            "AdaBoost": AdaBoostClassifier(),
            "MLP": MLPClassifier(),
            "MultinomialNB": MultinomialNB()
        }

        top_features = {}

        for clf_name, clf in classifiers.items():
            clf.fit(self.X, self.y)
            result = permutation_importance(clf, self.X, self.y, n_repeats=5, random_state=0)
            sorted_indices = np.argsort(result.importances_mean)[::-1]
            top_features[clf_name] = [self.X.columns[i] for i in sorted_indices[:100]]  
        # Top features from all classifiers
        intersection_features = set.intersection(*map(set, top_features.values()))
        if not intersection_features:
            print("No common top features found. Consider increasing top_n_features or reviewing model performance.")
        
        feature_votes = {feature: self.votes_df.loc[feature, 'Final Vote'] for feature in intersection_features}
        feature_votes_df = pd.DataFrame(list(feature_votes.items()), columns=['Feature', 'Final Vote'])
        return feature_votes_df
