import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from textwrap import wrap
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# PART 3 OF 3 : MODELING

# 1. Split into train and test sets, 7:3
df = pd.read_csv('Modeling_df.csv')
# df = df.drop(['v25'],axis=1)  # attribute v25 is very important to the model, apparently. So cannot drop the column with most remaining NAs.
df = df.dropna()
print("df shape after",df.shape)
y = df['has_applied']    # dependent variable
X = df[df.columns[-214:]]    # independent variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4,shuffle=True)
print("Labels counter y train",Counter(y_train)) # imbalanced 1:0 ratios...about 3:1
print("Labels counter y test",Counter(y_test))
# 2. Oversampling label 0s to balance training dataset

oversampler = RandomOverSampler(sampling_strategy='minority')
X_train_over, y_train_over = oversampler.fit_resample(X_train,y_train)
print("Oversampled y_train",Counter(y_train_over)) # now has_applied labels are balanced


# 3. Model training and testing

# ML algorithms for model vector
knn = KNeighborsClassifier(n_neighbors=10)
logr = LogisticRegression(max_iter=200, C=2.5)
lda = LinearDiscriminantAnalysis()
rf = RandomForestClassifier()
xgb = XGBClassifier()

model_vector = [knn,logr,lda,rf,xgb]

alg_name_for_plot = ['KNN','LogReg','LDA','RF','XGB']
alg_fpr_s, alg_tpr_s, alg_auc_s = [],[],[]

for alg,alg_name in itertools.zip_longest(model_vector,alg_name_for_plot):

    # fit the model
    alg.fit(X_train_over, y_train_over)
    alg_probs = alg.predict_proba(X_test)
    alg_probs = alg_probs[:,1]

    # Confusion matrix
    alg_preds = alg.predict(X_test)
    print ("Conf_matrix of {} is {}\n".format(str(alg), metrics.confusion_matrix(y_test, alg_preds)))

    # AUC scores
    alg_auc = roc_auc_score(y_test,alg_probs)
    alg_auc_s.append(alg_auc)
    print ('{} AUC = {}'.format(alg_name,round(alg_auc,3)))
    #calc roc curves
    alg_fpr, alg_tpr, _ = roc_curve(y_test, alg_probs)
    alg_fpr_s.append(alg_fpr)
    alg_tpr_s.append(alg_tpr)
    print("************")

# Baseline for ROC
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

# ROC curve plotting
for af, at, alg_name, aauc in itertools.zip_longest(alg_fpr_s, alg_tpr_s, alg_name_for_plot, alg_auc_s):
    pyplot.plot(af, at, label=alg_name+str(" ( AUC = ")+str(round(aauc,2))+" )")
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.title('ROC curves when v25 is included in modeling')
pyplot.show()

# Comparison of ROC of tuned and untuned Logistic Regression models
logr_untuned = LogisticRegression(max_iter=200)
model_vector_lreg = [logr,logr_untuned]
lr_name_for_plot = ['Tuned LogReg', 'Untuned LogReg']
lr_auc_s, lr_fpr_s, lr_tpr_s = [],[],[]
for alg,alg_name in itertools.zip_longest(model_vector_lreg,lr_name_for_plot):

    # fit the model
    alg.fit(X_train_over, y_train_over)
    alg_probs = alg.predict_proba(X_test)
    alg_probs = alg_probs[:,1]

    # Confusion matrix
    alg_preds = alg.predict(X_test)
    print ("Conf_matrix of {} is {}\n".format(str(alg), metrics.confusion_matrix(y_test, alg_preds)))

    # AUC scores
    alg_auc = roc_auc_score(y_test,alg_probs)
    lr_auc_s.append(alg_auc)
    print ('{} AUC = {}'.format(alg_name,round(alg_auc,3)))
    #calc roc curves
    alg_fpr, alg_tpr, _ = roc_curve(y_test, alg_probs)
    lr_fpr_s.append(alg_fpr)
    lr_tpr_s.append(alg_tpr)

    print("****LR***")

# ROC plot for tuned and untuned Logistic Regression models
for af, at, alg_name, aauc in itertools.zip_longest(lr_fpr_s, lr_tpr_s, lr_name_for_plot, lr_auc_s):
    pyplot.plot(af, at, label=alg_name+str(" ( AUC = ")+str(round(aauc,2))+" )")
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.title('ROC curves when LR is tuned and untuned')
pyplot.show()

# K fold cross validation (just as a comparison measure)
# names,results, results_test=[],[],[]
# for model,alg_name in itertools.zip_longest(model_vector,alg_name_for_plot):
#     kfold = StratifiedKFold(n_splits=10, shuffle=True)
#     cv_results = cross_val_score(model, X_train_over, y_train_over, cv=kfold, scoring='roc_auc')
#     results.append(round(cv_results.mean(),2))
#     print('{} kfold cross-validation AUC : {}'.format (str(alg_name), cv_results.mean()))
#
#     # predictions on test set
#     preds = model.predict(X_test)
#     print(roc_auc_score(y_test,preds))
#     results_test.append(round(roc_auc_score(y_test,preds),2))
#     print('#######################################')
#
# print("results vector",results_test)

# LR tuning code (for hyperparameter tuning of final model)

from sklearn.model_selection import RandomizedSearchCV
lr = LogisticRegression(penalty='l2')
max_iter = [160, 180, 200, 220, 240]
C = [1.0, 1.5, 2.0, 2.5]

param_grid = dict(max_iter=max_iter, C=C)
random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid, cv=3, n_jobs=-1,scoring='roc_auc')

random_result = random.fit(X_train_over, y_train_over)
# Summarize results
print("Best random: %f using %s" % (random_result.best_score_, random_result.best_params_))


# Conclusion : Logistic Regression and LDA models give the best shot at has_applied label pred, with AUC = 0.81

#########END OF PART 3 OF 3##############################