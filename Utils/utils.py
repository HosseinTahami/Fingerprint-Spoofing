import numpy as np
from scipy import linalg as la

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))


"""
PCA Utilities

"""

def compute_C_mu(D):
    mu = D.mean(1)
    mu = vcol(mu)
    DC = D - mu
    C = (DC @ DC.T) / float(D.shape[1])
    return C, mu


def compute_pca(D, m):
    C, mu = compute_C_mu(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(D, P):
    return P.T @ D


"""
LDA Utilities
"""

def compute_Sb_Sw(D, L):
    S_w = 0
    S_b = 0
    muGlobal = vcol(D.mean(1))
    for i in np.unique(L):
        D_cls = D[:, L == i]
        mu = vcol(D_cls.mean(1))
        DC_cls = D_cls - mu
        S_b += (mu - muGlobal) @ (mu - muGlobal).T * D_cls.shape[1]
        S_w += DC_cls @ DC_cls.T
    S_b /=  D.shape[1] 
    S_w /=  D.shape[1]
    
    return S_b, S_w


def compute_lda(D, L, m):

    S_b, S_w = compute_Sb_Sw(D, L)
    s, U = la.eigh(S_b, S_w)
    W = U[:, ::-1][:, :m]
    return W

def apply_lda(W, D):

    return W.T @ D



def split_db(D, L, seed=0, train_size=(2/3)):
    """
    D: Whole Data
    L: Whole Labels
    Seed: for Randomness
    train_size: The portion of data which we want to be training data, default=(2/3)

    Splitting data and it's labels 
    as Training & Validation data, randomly.

    """
    nTrain = int(D.shape[1]*train_size)         # Size of the Training Data
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])     # Creates a shuffled list of index numbers splits that list in two
    idxTrain = idx[0:nTrain]                    
    idxTest = idx[nTrain:]

    """
        DTR:  Data  Training
        DVAL: Data  Validation
        LTR:  Label Training
        LVAL: Label Validation
    """
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]


    return (DTR, LTR), (DVAL, LVAL)


def load(file_path):

    with open(file_path, "r") as f:

        lines = f.readlines()
        final_data = []
        labels = []
        line_counter = 0
        for line in lines:
            features_line = line.split(",")[0:-1]
            labels.append(int(line.split(",")[-1].split(",")[0]))
            clean_data = []
            line_counter += 1
            for feature in features_line:
                feature = float(feature)
                clean_data.append(feature)

            final_data.append(clean_data)

        final_data = np.array(final_data).T
        labels = np.array(labels)   

    return final_data, labels             
