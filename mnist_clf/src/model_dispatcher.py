'''
model_dispatcher.py imports tree from scikit-learn and defines a dictionary with
keys that are names of the models and values are the models themselves. Here, we
define two different decision trees, one with gini criterion and one with entropy. 

'''

from sklearn import tree
from sklearn import ensemble

models = {
"decision_tree_gini": tree.DecisionTreeClassifier(
criterion="gini"
),
"decision_tree_entropy": tree.DecisionTreeClassifier(
criterion="entropy"
),
"rf": ensemble.RandomForestClassifier(),
}