import random as rn
import numpy as np


def appendERR(combinations, k, max_rel):
    '''Compute ERRs for the combinations and append it as a column.

    Arguments:
        combinations {ndarray} -- combinations of rankings of relevance
        k {int} -- cut-off rank

    Returns:
        ndarray -- combinations horizontally appended with their respective ERRs
    '''
    thetas = (np.power(2, combinations) - 1) / 2 ** max_rel
    ERRs = np.empty(shape=(thetas.shape[0]), dtype=np.float32)
    for comb in np.arange(thetas.shape[0]):
        ERR = 0
        for r in np.arange(k):
            if r == 0:
                ERR += thetas[comb, 0] / (r + 1)
            elif r == 1:
                ERR += (1 - thetas[comb, 0]) * thetas[comb, 1] / (r + 1)
            else:
                tmp = thetas[comb, r]
                for i in np.arange(r - 1):
                    tmp *= 1 - thetas[comb, i]
                ERR += tmp / (r + 1)
        ERRs[comb] = ERR
    ERRs = np.reshape(ERRs, (thetas.shape[0], -1))
    return np.hstack((combinations, ERRs))


def getRankingPairs(combinations, k):
    '''Generate all the valid ranking pairs (E outperforms P)

    Arguments:
        combinations {ndarray} -- combinations of rankings of relevance
        k {int} -- cut-off rank

    Returns:
        list -- list of dictionaries containing the valid ranking pairs(dict)
    '''

    rankingPairs = []
    for P_ind in np.arange(combinations.shape[0]):
        for E_ind in np.arange(combinations.shape[0]):
            DERR = combinations[E_ind, k] - combinations[P_ind, k]
            if DERR > 0:
                rankingPairs.append(dict())
                tmp_pair = rankingPairs[-1]
                tmp_pair["P"] = combinations[P_ind, 0:k]
                tmp_pair["E"] = combinations[E_ind, 0:k]
                tmp_pair["DERR"] = DERR
    return rankingPairs


def main():
    k = 3
    max_rel = 2
    rels = np.arange(max_rel)
    combinations = np.array(np.meshgrid(rels, rels, rels)).T.reshape(-1, 3)
    combinations = appendERR(combinations, k, max_rel)
    rankingPairs = getRankingPairs(combinations, k)

    group1 = []
    group2 = []
    for rankingPair in rankingPairs:
        if (rankingPair["DERR"] <= 0.1) and (rankingPair["DERR"] > 0.05):
            group1.append(rankingPair)
        elif (rankingPair["DERR"] <= 0.2) and (rankingPair["DERR"] > 0.1):
            group2.append(rankingPair)

    print(len(group1))
    print(len(group2))


if __name__ == "__main__":
    main()
