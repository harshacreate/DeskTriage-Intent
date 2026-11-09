import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def split_indices(y, seed=42, groups=None, test_size=.25):
    y=np.asarray(y); idx=np.arange(len(y))
    if groups is not None:
        groups=np.asarray(groups)
        if groups.shape!=y.shape or len(np.unique(groups))<2: raise ValueError("group vector must align and have at least two groups")
        return next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(idx,y,groups))
    return train_test_split(idx,test_size=test_size,random_state=seed,stratify=y)
