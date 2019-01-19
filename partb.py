import random as rn
import numpy as np


def appendERR(combinations, k, max_rel):
    '''Compute ERRs for the combinations and append it as a column.

    Arguments:
        combinations {ndarray} -- combinations of rankings of relevance
        k {int} -- cut-off rank
        max_rel {int} -- highest grade for relevance

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
                for i in np.arange(r):
                    tmp *= 1 - thetas[comb, i]
                ERR += tmp / (r + 1)
        ERRs[comb] = ERR
    ERRs = np.reshape(ERRs, (thetas.shape[0], -1))
    # print(ERRs)
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


def getBins(rankingPairs):
    '''Collect the ranking pairs into bins in terms of DERR.

    Returns:
        list -- list of bins (which contains list of ranking pairs)
    '''

    groups = [[] for i in np.arange(10)]
    for rankingPair in rankingPairs:
        if (rankingPair["DERR"] <= 0.1) and (rankingPair["DERR"] > 0.05):
            groups[0].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.2) and (rankingPair["DERR"] > 0.1):
            groups[1].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.3) and (rankingPair["DERR"] > 0.2):
            groups[2].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.4) and (rankingPair["DERR"] > 0.3):
            groups[3].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.5) and (rankingPair["DERR"] > 0.4):
            groups[4].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.6) and (rankingPair["DERR"] > 0.5):
            groups[5].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.7) and (rankingPair["DERR"] > 0.6):
            groups[6].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.8) and (rankingPair["DERR"] > 0.7):
            groups[7].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.9) and (rankingPair["DERR"] > 0.8):
            groups[8].append(rankingPair)
        elif (rankingPair["DERR"] <= 0.95) and (rankingPair["DERR"] > 0.9):
            groups[9].append(rankingPair)
    return groups


def main():
    DEBUG = True
    k = 3
    max_rel = 1
    rels = np.arange(max_rel + 1)
    combinations = np.array(np.meshgrid(rels, rels, rels)).T.reshape(-1, 3)
    combinations = appendERR(combinations, k, max_rel)
    rankingPairs = getRankingPairs(combinations, k)

    groups = getBins(rankingPairs)

    if DEBUG:
        for i in np.arange(len(groups)):
            print('Group {} has {} pairs.'.format(i + 1, len(groups[i])))


if __name__ == "__main__":
    main()
