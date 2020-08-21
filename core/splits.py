# coding:utf-8
"""
StratifiedKFold and AllTrain are avalueable. Others are not sure.
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import BaseCrossValidator as _BaseCrossValidator
from sklearn.model_selection._split import check_random_state, indexable, _num_samples


class AllTrain(_BaseCrossValidator):
    """
    Random permutation average-validator

    Yields indices to split data all into training sets.

    Parameters
    ----------
    ```txt
    n_splits        : int, default 10
                      Number of splitting iterations.
    shuffle         : bool, default False
                      Shuffle permutation.
    random_state    : int, random seed, default None.
    ```
    """
    def __init__(self, n_splits, shuffle=False, random_state=None):
        if not isinstance(n_splits, int):
            raise ValueError('The number of folds must be of int type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))

        if n_splits < 1:
            raise ValueError(
                "Cross-validation requires at least one"
                " train/test split by setting n_splits=1 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : None

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        X, y, groups = indexable(X, y, groups)
        for test, train in super().split(X, y, groups):
            if self.shuffle:
                np.random.seed(self.random_state)
                np.random.shuffle(train)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        n_splits = self.n_splits
        for n in range(n_splits):
            yield indices