from itertools import permutations
from time import time
import numpy as np
import random as rn


def getCombinations(docs, k):
    '''Generate all combinations for ranking of relevence and docIDs.

    Arguments:
        docs {int} -- max number of distinct docs
        k {int} -- cut-off rank

    Returns:
        ndarray -- combinations with their corresponding docIDs, pattern: [0, 1, 0, 3, 5, 2]
    '''

    tmp_rels = np.hstack((np.zeros(docs), np.ones(docs))).reshape(-1, 1)
    tmp_docs = np.arange(docs + docs).reshape(-1, 1)  # from 0 to 11
    tmp_combs = np.hstack((tmp_rels, tmp_docs))
    perms = np.array(list(permutations(tmp_combs, r=k)))
    combinations = np.empty(shape=(perms.shape[0], perms.shape[1] * perms.shape[2]), dtype=np.float32)
    combinations[:, 0:k] = perms[:, :, 0]
    combinations[:, k:k + k] = perms[:, :, 1]

    return combinations


def appendERR(combinations, k, max_rel):
    '''Compute ERRs for the combinations and append it as a column.

    Arguments:
        combinations {ndarray} -- combinations of rankings of relevance, pattern: [0, 1, 0, 4(docID), 2(docID), 1(docID)]
        k {int} -- cut-off rank
        max_rel {int} -- highest grade for relevance

    Returns:
        ndarray -- combinations horizontally appended with their respective ERRs
    '''
    thetas = (np.power(2, combinations[:, :3]) - 1) / 2 ** max_rel
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
    return np.hstack((combinations, ERRs))


def getRankingPairs(combinations, k):
    '''Generate all the valid ranking pairs (E outperforms P)

    Arguments:
        combinations {ndarray} -- combinations of rankings of relevance, pattern: [0, 1, 0, 4(docID), 2(docID), 1(docID), 0.6(ERR)]
        k {int} -- cut-off rank

    Returns:
        list -- list of dictionaries containing the valid ranking pairs(dict)
    '''

    rankingPairs = []
    for P_ind in np.arange(combinations.shape[0]):
        for E_ind in np.arange(combinations.shape[0]):
            DERR = combinations[E_ind, -1] - combinations[P_ind, -1]
            if DERR > 0:
                rankingPairs.append(dict())
                tmp_pair = rankingPairs[-1]
                tmp_pair["P"] = combinations[P_ind, 0:k]
                tmp_pair["P_docID"] = combinations[P_ind, k:k + k]
                tmp_pair["E"] = combinations[E_ind, 0:k]
                tmp_pair["E_docID"] = combinations[E_ind, k:k + k]
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


def teamDraftInterleave(rankP, rankE, docP, docE):
    '''Interleave the ranking pairs for online evaluation.

    Arguments:
        rankP {ndarray} -- 
        rankE {ndarray} -- 
        docP {ndarray} -- 
        docE {ndarray} -- 

    Returns:
        list -- interleaved result, pattern: ['0.0P' '1.0E' '0.0E']
    '''

    P = rankP.tolist()
    E = rankE.tolist()
    dP = docP.tolist()
    dE = docE.tolist()
    interleavedList = []
    docIDs = []
    while len(interleavedList) < 3:
        order = rn.random()
        if order > 0.5:  # P goes first
            if P:
                while dP[0] in docIDs:
                    P.pop(0)
                    dP.pop(0)
                    if not P:
                        break
                if P:
                    interleavedList.append(str(P[0])+'P')
                    docIDs.append(dP[0])
                    P.pop(0)
                    dP.pop(0)
            if E:
                if len(interleavedList) == 3:
                    break
                while dE[0] in docIDs:
                    E.pop(0)
                    dE.pop(0)
                    if not E:
                        break
                if E:
                    interleavedList.append(str(E[0])+'E')
                    docIDs.append(dE[0])
                    E.pop(0)
                    dE.pop(0)
        else:  # E goes first
            if E:
                while dE[0] in docIDs:
                    E.pop(0)
                    dE.pop(0)
                    if not E:
                        break
                if E:
                    interleavedList.append(str(E[0])+'E')
                    docIDs.append(dE[0])
                    E.pop(0)
                    dE.pop(0)
            if P:
                if len(interleavedList) == 3:
                    break
                while dP[0] in docIDs:
                    P.pop(0)
                    dP.pop(0)
                    if not P:
                        break
                if P:
                    interleavedList.append(str(P[0])+'P')
                    docIDs.append(dP[0])
                    P.pop(0)
                    dP.pop(0)
    return interleavedList


def computeProbDist(listLength):
    tau = 3
    denominator = 0
    probList = []
    for index in range(listLength):
        rank = index + 1
        probList.append(1 / np.power(rank, tau))
        denominator += 1 / np.power(rank, tau)
    for index, item in enumerate(probList):
        item /= denominator
        probList[index] = round(item, 2)
    return probList

def probInterleave(rankP, rankE, docP, docE):
    P = rankP.tolist()
    E = rankE.tolist()
    dP = docP.tolist()
    dE = docE.tolist()
    interleavedList = []
    probDistP = computeProbDist(len(P))
    probDistE = computeProbDist(len(E))
    while len(interleavedList) < 3:
        turn = rn.random()
        if turn > 0.5:  # P picks
            if P:
                pick = rn.random()
                temp = 0
                for index in range(len(probDistP)):
                    temp += probDistP[index]
                    if pick < temp:
                        interleavedList.append(str(P[index])+'P')
                        if dP[index] in dE:
                            tIndex =  dE.index(dP[index])
                            E.pop(tIndex)
                            dE.pop(tIndex)
                            probDistE = computeProbDist(len(E))
                        P.pop(index)
                        dP.pop(index)
                        probDistP = computeProbDist(len(P))
                        break
                        
        else:  # E picks
            if E:
                pick = rn.random()
                temp = 0
                for index in range(len(probDistE)):
                    temp += probDistE[index]
                    if pick < temp:
                        interleavedList.append(str(E[index])+'E')
                        if dE[index] in dP:
                            tIndex = dP.index(dE[index])
                            P.pop(tIndex)
                            dP.pop(tIndex)
                            probDistP = computeProbDist(len(P))
                        E.pop(index)
                        dE.pop(index)
                        probDistE = computeProbDist(len(E))
                        break
    return interleavedList


class RandomClickModel:
    '''Class for Random Click Model.
    '''

    def __init__(self, docPerPage):
        self.docPerPage = docPerPage

    def estimateParameters(self, clickLog):
        '''Estimate the probability of random click.

        Arguments:
            clickLog {string} -- path to the log file

        '''

        q = 0
        c = 0
        with open(clickLog) as log:
            for line in log:
                tmp = line.split()
                if 'Q' in tmp:
                    q += 1
                elif 'C' in tmp:
                    c += 1
        self.rho = c / (q * self.docPerPage)

    def simulate(self, int_res):
        '''One user click simulation with RCM.

        Arguments:
            int_res {list or ndarray} -- interleaved result

        Returns:
            int -- result of comparison between E and P (win, tie, lose)
        '''

        if type(int_res) == list:
            int_len = len(int_res)
        elif type(int_res) == np.ndarray:
            int_len = int_res.shape[0]
            int_res = int_res.reshape(int_len, -1)

        P = 0
        E = 0
        for i in np.arange(int_len):
            if np.random.rand(1) < self.rho:
                if 'P' in int_res[i]:
                    P += 1
                else:
                    E += 1
        if E > P:
            return 1
        elif E == P:
            return 0
        else:
            return -1

    def computeSampleSize(self, alpha, beta, p0, repetition, int_res):
        '''Compute the sample size for the interleaving result (DERR).

        Arguments:
            alpha {float} -- Type I error rate
            beta {float} -- Type II error rate
            p0 {float} -- proportion for comparison
            repetition {int} -- number of repetitions for user click simulation
            int_res {list or ndarray} -- interleaved result

        Returns:
            int -- computed sample size
        '''

        P = 0
        E = 0
        for _ in np.arange(repetition):
            sim_res = self.simulate(int_res)
            if sim_res == 1:
                E += 1
            elif sim_res == -1:
                P += 1

        p1 = E / (E + P)
        delta = np.abs(p1 - p0)
        z_alpha = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (1 - alpha) ** 2)
        z_beta = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (1 - beta) ** 2)
        if delta == 0.0:
            N = 0
        else:
            N = ((z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1))) / delta) ** 2 + 1 / delta
        return np.ceil(N)


def getStatistics(groups, clickModel, alpha=0.05, beta=0.1, p0=0.5, repetition=50):
    '''Calculate statistics for each groups of ranking pairs.

    Arguments:
        groups {list} -- list of groups
        clickModel {RandomClickModel or PositionBasedModel} -- instance of either click model

    Keyword Arguments:
        alpha {float} -- Type I error rate (default: {0.05})
        beta {float} -- Type II error rate (default: {0.1})
        p0 {float} -- comparison proportion (default: {0.5})
        repetition {int} -- number of repetitions (default: {50})

    Returns:
        list -- list of dictionaries containing statistics for each group
    '''

    groupStatistics = []
    for group in groups:
        tmpN = np.empty(shape=(len(group)), dtype=np.float32)
        for i, pair in enumerate(group):
            int_res = teamDraftInterleave(pair['P'], pair['E'], pair['P_docID'], pair['E_docID'])
            N = clickModel.computeSampleSize(alpha, beta, p0, repetition, int_res)
            if N == 0.0:
                continue
            tmpN[i] = N
        groupStatistics.append(dict())
        tmp = groupStatistics[-1]
        if len(group) == 0:
            tmp['min'] = 'Nan'
            tmp['max'] = 'Nan'
            tmp['mean'] = 'Nan'
            tmp['median'] = 'Nan'
        else:
            tmp['min'] = np.min(tmpN)
            tmp['max'] = np.max(tmpN)
            tmp['mean'] = np.mean(tmpN)
            tmp['median'] = np.median(tmpN)
    return groupStatistics


def main():
    DEBUG = True
    k = 3
    max_rel = 1
    docPerPage = 10
    clickLog = './YandexRelPredChallenge.txt'
    alpha = 0.05
    beta = 0.1
    p0 = 0.5
    repetition = 50
    docs = 6

    # docIDs = np.arange(docs)
    # rels = np.arange(max_rel + 1)
    # combinations = np.array(np.meshgrid(rels, rels, rels, docIDs, docIDs, docIDs)).T.reshape(-1, 6)
    combinations = getCombinations(docs, k)
    combinations = appendERR(combinations, k, max_rel)
    rankingPairs = getRankingPairs(combinations, k)
    groups = getBins(rankingPairs)

    if DEBUG:
        count = 0
        for i in np.arange(len(groups)):
            print('Group {} has {} pairs.'.format(i + 1, len(groups[i])))
            count += len(groups[i])
        print("In total {} pairs:".format(count))
        print("Example pair:", groups[0][4])
        testTDI = teamDraftInterleave(groups[0][4].get('P'), groups[0][4].get('E'), groups[0][4].get('P_docID'), groups[0][4].get('E_docID'))
        print("Interleaved result with TeamDraftInterleaving:", testTDI)
        testPI = probInterleave(groups[0][4].get('P'), groups[0][4].get('E'), groups[0][4].get('P_docID'), groups[0][4].get('E_docID'))
        print("Interleaved result with ProbInterleaving:", testPI)
        print(combinations.shape)
    start = time()
    print('Random Click Model: ')
    rcm = RandomClickModel(docPerPage)
    rcm.estimateParameters(clickLog)
    groupStatistics = getStatistics(groups, rcm)
    print('Group statistics: ')
    print(groupStatistics)
    print('Time elapsed: {:.3f}s'.format(time() - start))


if __name__ == "__main__":
    main()
