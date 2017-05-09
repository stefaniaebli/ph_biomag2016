import numpy as np
from scipy.io import loadmat


def load_data(subject=1):
    filename = 'data/meg_data_%da.mat' % subject
    data = loadmat(filename)

    triggers = data['triggers']
    planardat = data['planardat']

    t1 = triggers['t1'][0, 0].squeeze()
    t2 = triggers['t2'][0, 0].squeeze()
    t3 = triggers['t3'][0, 0].squeeze()
    t4 = triggers['t4'][0, 0].squeeze()
    t5 = triggers['t5'][0, 0].squeeze()
    t6 = triggers['t6'][0, 0].squeeze()
    test = triggers['test'][0, 0].squeeze()
    behav = triggers['behav'][0, 0].squeeze()
    return t1, t2, t3, t4, t5, t6, test, behav, planardat


def create_train_test_sets(subject=1, window_size=125, t_offset=0,
                           normalize=True):
    t1, t2, t3, t4, t5, t6, test, behav, planardat = load_data(subject)
    t_all = np.concatenate([t1, t2, t3, t4, t5, t6])
    labels_all = np.concatenate([np.ones(t1.size), np.ones(t2.size) * 2,
                                 np.ones(t3.size) * 3, np.ones(t4.size) * 4,
                                 np.ones(t3.size) * 5, np.ones(t4.size) * 6])
    idx = np.argsort(t_all)
    if normalize:
        # This is a global normalization, not per-trial:
        planardat = np.nan_to_num((planardat.T - planardat.mean(1))
                                  / planardat.std(1)).T

    # Ntrials x Nchannels X Nsamples:
    X_train = np.array([planardat[:, t+t_offset: t+t_offset+window_size]
                        for t in t_all[idx]])
    y_train = labels_all[idx].astype(np.int)
    X_test = np.array([planardat[:, t+t_offset: t+t_offset+window_size]
                       for t in test])
    return X_train, y_train, X_test


if __name__ == '__main__':

    subject = 1
    t1, t2, t3, t4, t5, t6, test, behav, planardat = load_data(subject)

    import matplotlib.pyplot as plt
    plt.interactive(True)
    plt.figure()
    plt.plot(planardat[0, :], 'r-')
    plt.plot(t4, np.zeros(t4.size), 'b*')
    plt.plot(test, np.zeros(test.size), 'kx')
    plt.plot(behav, np.zeros(behav.size), 'go')

    X_train, y_train, X_test = create_train_test_sets(subject)

    plt.figure()
    plt.plot(X_train[0, 0, :], 'r-')

