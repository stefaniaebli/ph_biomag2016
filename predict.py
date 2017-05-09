import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

from load import create_train_test_sets
from scipy.io import savemat
from persistent_homology import persistent_homology, persistent_homology_parallel


def ph_dataset(covs, order_max=0, bins=10):
    ph, features = persistent_homology_parallel(covs, order_max=order_max, bins=bins)
    ph_vectors = np.array([np.concatenate(f[0][1:]) for f in features])
    if order_max > 0:
        ph_vectors1 = np.array([np.concatenate(f[1][1:]) for f in features])
        ph_vectors = np.hstack([ph_vectors, ph_vectors1])

    print("ph_vectors" + str(ph_vectors.shape))
    idx_nonzero = (ph_vectors.std(0) != 0)
    ph_vectors = ph_vectors[:, idx_nonzero]
    print("ph_vectors" + str(ph_vectors.shape))
    return ph_vectors


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
    order_max = 1
    bins = 10
    # betti_number = 0
    test_set = False

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
    # cov_all = cov_all * cov_all
    cov_all = np.abs(cov_all)
    # cov_all = (cov_all - np.min(cov_all))
    cov_all = cov_all/ np.median(cov_all)

    cov_train = cov_all[:X_train.shape[0], :, :]
    cov_test = cov_all[X_train.shape[0]:, :, :]

    print("Computing persistent homology of covariance matrices on the train set.")
    ph_train = ph_dataset(cov_train, order_max=order_max, bins=bins)

    if test_set:
        print("Computing persistent homology of covariance matrices on the test set.")
        ph_test = ph_dataset(cov_test, order_max=order_max, bins=bins)

    print("Cross validated %s:" % scoring)
    clf = LogisticRegressionCV()
    print("Classifier: %s" % clf)
    cv = StratifiedKFold(y_train, n_folds=cv)
    score = cross_val_score(clf, ph_train, y_train, scoring=scoring,
                            cv=cv, n_jobs=-1)
    print("Label %d, %s = %f" % (label, scoring, score.mean()))

    if test_set:
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


