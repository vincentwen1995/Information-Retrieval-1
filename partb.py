from itertools import permutations
import numpy as np


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
    # print(ERRs)
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


class RandomClickModel:
    '''Class for Random Click Model.
    '''

    def __init__(self, docPerPage):
        self.docPerPage = docPerPage

    def estimateRho(self, clickLog):
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
                if int_res[i] == 0:
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
        N = ((z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1))) / delta) ** 2 + 1 / delta
        return np.ceil(N)


def main():
    DEBUG = True
    k = 3
    max_rel = 1
    docPerPage = 10
    clickLog = './YandexRelPredChallenge.txt'
    alpha = 0.05
    beta = 0.1
    p0 = 0.5
    repetition = 100
    docs = 6

    # docIDs = np.arange(docs)
    # rels = np.arange(max_rel + 1)
    # combinations = np.array(np.meshgrid(rels, rels, rels, docIDs, docIDs, docIDs)).T.reshape(-1, 6)
    combinations = getCombinations(docs, k)
    combinations = appendERR(combinations, k, max_rel)
    rankingPairs = getRankingPairs(combinations, k)
    groups = getBins(rankingPairs)

    if DEBUG:
        for i in np.arange(len(groups)):
            print('Group {} has {} pairs.'.format(i + 1, len(groups[i])))
        print(groups[0][4])
        print(combinations.shape)
        rcm = RandomClickModel(docPerPage)
        rcm.estimateRho(clickLog)
        int_res = [1, 1, 0, 0, 1, 1, 0, 1, 1, 0]
        N = rcm.computeSampleSize(alpha, beta, p0, repetition, int_res)
        print(N)


if __name__ == "__main__":
    main()
