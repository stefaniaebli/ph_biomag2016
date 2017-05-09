import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

from load import create_train_test_sets
from scipy.io import savemat
from persistent_homology import persistent_homology


if __name__ == '__main__':
    print("Biomag2016: Competition 3")
    print("Our attempt uses pyRiemann with a sklearn classifier.")
    subject = 1
    window_size = 150  # temporal window of interst, in timesteps
    t_offset = 15  # beginning of the time window of, from onset
    normalize = False  # unnecessary
    estimator = 'oas'  # covariance estimator
    metric = 'riemann'  # metric for the tangent space
    scoring = 'roc_auc'  # scoring metric
    label = 4  # the label of interest is 4, i.e. "happy"
    cv = 10  # folds for cross-validation
    betti_number = 0

    print("Loading data of subject %d." % subject)
    X_train, y_train, X_test = create_train_test_sets(subject=subject,
                                                      window_size=window_size,
                                                      t_offset=t_offset,
                                                      normalize=normalize)
    y_train = (y_train == label).astype(np.int)  # the labels
    X_all = np.vstack([X_train, X_test])

    print("Estimating covariance matrices with covariance estimator '%s'."
          % estimator)
    cov_all = Covariances(estimator=estimator).fit_transform(X_all)

    cov_train = cov_all[:X_train.shape[0], :, :]
    cov_test = cov_all[X_train.shape[0]:, :, :]

    ph_train = persistent_homology(cov_train, order=betti_number)
    ph_test = persistent_homology(cov_test, order=betti_number)

    print("Cross validated %s:" % scoring)
    clf = LogisticRegressionCV()
    print("Classifier: %s" % clf)
    cv = StratifiedKFold(y_train, n_folds=cv)
    score = cross_val_score(clf, ph_train, y_train, scoring=scoring,
                            cv=cv, n_jobs=-1)
    print("Label %d, %s = %f" % (label, scoring, score.mean()))

    print("")
    print("Training on training data.")
    clf = LogisticRegressionCV()
    clf.fit(ph_train, y_train)
    print("Predicting test data.")
    y_test = clf.predict_proba(ph_test)
    filename = 'subject%d.mat' % subject
    print("Saving predictions to %s" % filename)
    savemat(file_name=filename,
            mdict={'predicted_probability': y_test[:, 1]})

