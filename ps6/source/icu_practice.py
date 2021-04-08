"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2021 Jan 22
Description : Survival of ICU Patients

This code is adapted from course material by Jenna Wiens (UMichigan).
Docstrings based on scikit-learn format.
"""

# python libraries
import sys

# data science libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# scikit-learn libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

# project-specific helper libraries
import icu_config
import tests



######################################################################
# globals
######################################################################

RANDOM_SEED = 42    # seed for RepeatedStratifiedKFold
EPS = 10 * sys.float_info.epsilon

NRECORDS = 100      # number of patient records
FEATURES_TRAIN_FILENAME, LABELS_TRAIN_FILENAME = \
    icu_config.get_filenames(nrecords=NRECORDS)

METRICS = ["accuracy", "auroc", "f1score", "sensitivity", "specificity", "precision"] # sensitivity = recall



######################################################################
# functions
######################################################################

def score(y_true, y_score, metric='accuracy') :
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
    y_true : numpy array of shape (n_samples,)
        Ground truth (correct) labels.
    
    y_score : numpy array of shape (n_samples,)
        Target scores (continuous-valued) predictions.
    
    metric : {'accuracy', 'auroc', 'f1score', 'sensitivity', 'specificity', 'precision', 'recall'}
        Performance metric.
    
    Returns
    --------------------
    score : float
        Performance score.
    """
    
    # map continuous-valued predictions to binary labels
    y_pred = np.sign(y_score)
    y_pred[y_pred == 0] = 1 # map points on hyperplane to +1
    
    ### ========== TODO : START ========== ###
    # part a : compute classifier performance for specified metric
    # professor's solution: 16 lines
    if metric == 'auroc':
        score = roc_auc_score(y_true, y_score)
    else:
    # compute confusion matrix
        confusionMatrix = confusion_matrix(y_true, y_pred)
        print('----------------------------')
        print(confusionMatrix)
        print("hi")
        print('----------------------------')
    
    # compute scores
    
    
    ### ========== TODO : END ========== ###


def plot_cv_results(results, scorers, param_name) :
    """Plot performance for tuning LinearSVC.
    
    You do NOT have to understand the implementation of this function.
    It basically pulls together data from GridSearch.cv_results_ and plots it using seaborn.
    """
    
    # convert to data frame
    df = pd.DataFrame(results)
    df = df.filter(regex=f'{param_name}|split[0-9]+_*', axis=1)    # keep matching columns
    df = pd.melt(df, id_vars=[param_name], value_name='score')
    df['split'] = df['variable'].apply(lambda x: (x.split('_'))[0])
    df['dataset'] = df['variable'].apply(lambda x: (x.split('_'))[1])
    df.replace({'dataset': {'test': 'cv'}}, inplace=True)
    df['metric'] = df['variable'].apply(lambda x: (x.split('_'))[2])
    df.drop(columns=['variable'], inplace=True)
    
    # plot
    grid = sns.relplot(data=df, x=param_name, y='score', ci='sd',
                       col='dataset', col_order=['train', 'cv'],
                       hue='metric', hue_order=list(scorers.keys()),
                       kind='line')
    if '__' in param_name :
        xlabel = param_name.split('__', 1)[1]
    elif '_' in param_name :
        xlabel = param_name.split('_', 1)[1]
    else :
        xlabel = param_name
    grid.set(xlabel=xlabel)
    grid.set(xscale="log")
    grid.set(ylim=[0,1])
    grid.tight_layout()
    
    plt.show()



######################################################################
# main
######################################################################

def main() :
    np.random.seed(42)
    
    #========================================
    # read data
    
    print('Reading data...')
    
    df_features_train = pd.read_csv(FEATURES_TRAIN_FILENAME)
    X_train = df_features_train.drop('RecordID', axis=1).values
    
    df_labels_train = pd.read_csv(LABELS_TRAIN_FILENAME)
    y_train = df_labels_train['In-hospital_death'].values
    
    # make copies so we do not risk changing training set
    X = np.copy(X_train)
    y = np.copy(y_train)
    
    print()
    
    #========================================
    # try out preprocessors and classifiers
    # as always, make sure to set the correct hyperparameters for each estimator
    
    print('Preprocessing and classifying...')
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_train)
    assert np.any(~np.isnan(X_imputed))         # sanity check
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_imputed_scaled = scaler.fit_transform(X_imputed)
    assert X_imputed_scaled.ravel().min() > -1-EPS and \
        X_imputed_scaled.ravel().max() < 1+EPS  # sanity check
    
    clf = SVC(kernel='linear', class_weight='balanced', max_iter=1e6)
    clf.fit(X_imputed_scaled, y)
    
    #========================================
    # score
    
    print('Computing multiple metrics...')
    
    tests.test_score()
    
    print()
    
    y_score = clf.decision_function(X_imputed_scaled)
    for metric in METRICS :
        print(f'{metric}: ', score(y, y_score, metric))
    
    print()
    
    #========================================
    # search over Linear SVM hyperparameters using various scoring metrics
    
    print('Tuning Linear SVM...')
    
    # create scoring dictionary to maps scorer name (metric) to scoring function
    # make_scorer(score_func, needs_threshold=True, **kwargs)
    #   score_func has signature score_func(y, y_pred, **kwargs) and returns a scalar score
    #   needs_threshold is True says score_func requires continuous decisions
    #   **kwargs allows us to pass additional arguments to score_func (e.g. metric)
    scoring = {}
    for metric in METRICS :
        scoring[metric] = make_scorer(score, needs_threshold=True, metric=metric)
    
    # run exhaustive grid search
    # GridSearchCV(..., scoring, refit, ...)
    #   you should be familiar with most of the parameters to GridSearchCV
    #   scoring with a list or dict allows us to specify multiple metrics
    #   refit allows us to refit estimator on whole dataset
    #     refit=False says do NOT refit estimator
    #     if want to refit when using multiple metrics, use string corresponding to key in scoring dict
    param_grid = {'C': np.logspace(-3, 3, 7)}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_SEED)
    search = GridSearchCV(clf, param_grid,
                          scoring=scoring, cv=cv, refit=False,
                          return_train_score=True)
    search.fit(X_imputed_scaled, y)
    results = search.cv_results_
    
    # plot results
    # param_name specifies the hyperparameter of interest (the key to look up in results)
    plot_cv_results(results, search.scorer_, param_name='param_C')
    
    print()
        
    #========================================
    # put everything together in a pipeline
    
    print('Making and tuning pipeline...')
    
    # make pipeline
    # Pipeline(...) takes in list of (name, transform) pairs
    #   where name is name of the step and transform is corresponding transformer or estimator
    steps = [
        ('imputer', imputer),
        ('scaler', scaler),
        ('clf', clf)
    ]
    pipe = Pipeline(steps)
    
    # make parameter grid
    # nested parameters use the syntax <estimator>__<parameter>
    # estimator is the name of the step
    param_grid = {'clf__C' : np.logspace(-3, 3, 7)}
    
    # tune pipeline
    search = GridSearchCV(pipe, param_grid,
                          scoring=scoring, cv=cv, refit=False,
                          return_train_score=True)
    search.fit(X, y)
    results = search.cv_results_
    
    # plot results
    plot_cv_results(results, search.scorer_, param_name='param_clf__C')
    
    ### ========== TODO : START ========== ###
    # part c : find optimal hyperparameter setting for each metric
    #          report corresponding mean train score and test score
    #          everything you need is in results variable
    # professor's solution: 12 lines
    
    for scorer in sorted(scoring) :
        pass
    
    ### ========== TODO : END ========== ###
    
    print()


if __name__ == '__main__' :
    main()
