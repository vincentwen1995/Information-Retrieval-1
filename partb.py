from itertools import permutations
from time import time
from scipy.stats import norm
from tqdm import tqdm
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
    temp = []
    for P_ind in np.arange(combinations.shape[0]):
        for E_ind in np.arange(combinations.shape[0]):
            DERR = combinations[E_ind, -1] - combinations[P_ind, -1]
            diff = []
            for i in range(len(combinations[P_ind, 0:k])):
                if combinations[P_ind, k:k + k][i] == combinations[E_ind, k:k + k][i]:
                    diff.append('T')
                else:
                    diff.append('E')
            inList = str(combinations[P_ind, 0:k]) + str(combinations[E_ind, 0:k]) + str(diff)
            if (inList not in temp) and DERR > 0:
                temp.append(inList)
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
    '''Interleave the ranking pairs for online evaluation with team draft method.

    Arguments:
        rankP {ndarray} -- rank of P
        rankE {ndarray} -- rank of E
        docP {ndarray} -- docID of P
        docE {ndarray} -- docID of E

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
                    interleavedList.append(str(int(P[0])) + 'P')
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
                    interleavedList.append(str(int(E[0])) + 'E')
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
                    interleavedList.append(str(int(E[0])) + 'E')
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
                    interleavedList.append(str(int(P[0])) + 'P')
                    docIDs.append(dP[0])
                    P.pop(0)
                    dP.pop(0)
    return interleavedList


def computeProbDist(listLength):
    '''[summary]

    Arguments:
        listLength {int} -- length of ranking result

    Returns:
        list -- list containing probabilities of each rank to be sampled
    '''

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
    '''Interleave the ranking pairs for online evaluation with probabilistic method.

    Arguments:
        rankP {ndarray} -- rank of P
        rankE {ndarray} -- rank of E
        docP {ndarray} -- docID of P
        docE {ndarray} -- docID of E

    Returns:
        list -- interleaved result, pattern: ['0.0P' '1.0E' '0.0E']
    '''

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
                        interleavedList.append(str(int(P[index])) + 'P')
                        if dP[index] in dE:
                            tIndex = dE.index(dP[index])
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
                        interleavedList.append(str(int(E[index])) + 'E')
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

    def computeSampleSize(self, alpha, beta, p0, repetition, int_func, int_args):
        '''Compute the sample size for the interleaving result (DERR).

        Arguments:
            alpha {float} -- Type I error rate
            beta {float} -- Type II error rate
            p0 {float} -- proportion for comparison
            repetition {int} -- number of repetitions for user click simulation
            int_func {list or ndarray} -- interleaved function to use
            int_args {tuple} -- arguments for interleaving function
        Returns:
            int -- computed sample size
        '''

        P = 0
        E = 0
        for _ in np.arange(repetition):
            int_res = int_func(*int_args)
            sim_res = self.simulate(int_res)
            if sim_res == 1:
                E += 1
            elif sim_res == -1:
                P += 1

        p1 = E / (E + P)
        delta = np.abs(p1 - p0)
        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(1 - beta)
        if delta == 0.0:
            N = 0
        else:
            N = ((z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1))) / delta) ** 2 + 1 / delta
        return np.ceil(N)


class PositionBasedModel:
    '''Class for Position Based Model.
    '''

    def __init__(self):
        self.gamma = 0

    def preprocessing(self, max_rank=3, file_path="YandexRelPredChallenge.txt"):
        '''Preprocessing Yandex log.

        Preprocess Yandex log to get convient data structures S_uq and S for
        calcuting gamma and alpha using EM algorithm.

        Args:
            max_rank: Max rank, by default 3 in this case.
            Path: File path for Yandex log.

        Returns:
            S_uq: A dictionary, used for updating alpha. In S_uq, each key is a
            (document_id, query_id) pair, corresponding value is a list, with each
            element a tuple (rank, observed click) recording information about each
            s in S_uq. With this format:

            S_uq: {[(document_id, query_id)]:[(rank, observed click),...]}

            S: A dictionary, used for updating gamma. In S, each key is a
            session id(the row number of row containing query in Yandex log, not
            for row containing click), corresponding value is a list, with each
            element recording information about (document_id, query_id) and
            observed click on rank 1-3 for every s in S. With this format:

            S: {[document_id]:[((document_id, query_id),observed click),...]}
        '''

        # initialization
        S_uq = {}
        S = {}

        # read Yandex log
        with open(file_path) as f:
            data = []
            for line in f:
                data.append(line.rstrip('\n').split('\t'))

        for i in range(len(data)):  # for every row in Yandex log
            record = data[i]
            action = record[2]

            # if the record is a query
            if action == "Q":
                S[i] = []  # initialize session i
                docs_last_q = []  # docs in the last query, for examining click
                id_last_s = i  # id of the last session, for examining click

                for r in range(max_rank):  # for rank from 1 until max_rank
                    query_id = record[3]
                    doc_id = record[r + 5]
                    key = (doc_id, query_id)

                    rank = r
                    c_us = 0

                    docs_last_q.append(doc_id)

                    # update S_uq
                    if key not in S_uq:
                        S_uq[key] = [(rank, c_us)]
                    else:
                        S_uq[key].append((rank, c_us))

                    # update S
                    S[i].append((key, c_us))

            # if the record is a click
            else:
                doc_id = record[-1]
                key = (doc_id, query_id)  # session of a click is the last query

                if doc_id in docs_last_q:  # if the doc clicked is in the last query
                    rank = S_uq[key][-1][0]  # rank of the doc clicked in the last session
                    c_us = 1

                    # update S_uq
                    S_uq[key][-1] = (rank, c_us)

                    # update S
                    S[id_last_s][rank] = (key, c_us)

        return S_uq, S

    def learn_by_EM(self, gamma_0=0.5, alpha_0=0.5, max_rank=3):
        '''Learning gamma and alpha using EM.

        Args:
            gamma_0: A float, initialized value for gamma.
            alpha_0: A float, initialized value for alpha.
            max_rank: Max rank, by default 3 in this case.

        Returns:
            gamma_t1: A np.arrayupdated, with max_rank = 3.
        '''
        # get data structure
        S_uq, S = self.preprocessing()

        # retrieve all uq pairs
        all_uq = list(S_uq.keys())

        # initialize gamma
        gamma_t = np.array([gamma_0] * max_rank)
        gamma_t1 = np.array([0.9] * max_rank)

        # init alpha
        alpha_t = {}

        for uq in all_uq:
            alpha_t[uq] = alpha_0
        alpha_t1 = alpha_t.copy()

        # beging learing until converge
        total_diff = 1
        while total_diff > 0.0001:

            # update alpha
            for uq in all_uq:
                s_uq = S_uq[uq]  # retrieve all sessions s in S_uq
                alpha_sum = 0  # for summation
                length_s_uq = len(s_uq)
                for s in s_uq:  # for every session s in S_uq
                    rank = s[0]
                    c_us = s[1]
                    alpha_sum += (c_us + (1 - c_us) * (1 - gamma_t[rank]) * alpha_t[uq] / (1 - gamma_t[rank] * alpha_t[uq]))
                alpha_t1[uq] = (alpha_sum + 1) / (length_s_uq + 2)

            # update gamma
            length_S = len(S)
            for rank in range(max_rank):
                gamma_sum = 0  # for summation
                for i in S.keys():  # for every session s in S
                    s_i = S[i]
                    uq = s_i[rank][0]
                    c_us = s_i[rank][1]
                    gamma_sum += (c_us + (1 - c_us) * gamma_t[rank] * (1 - alpha_t[uq]) / (1 - gamma_t[rank] * alpha_t[uq]))
                gamma_t1[rank] = (gamma_sum + 1) / (length_S + 2)

            total_diff = np.sum(gamma_t1 - gamma_t)

            # update values at t using values at t+1
            gamma_t = gamma_t1.copy()
            alpha_t = alpha_t1.copy()

        self.gamma = gamma_t1

    def simulate(self, int_res, epsilon=0.3):
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
            rel = int_res[i]
            if '1' in rel:
                P_click = self.gamma[i] * (1 - epsilon)
                if np.random.rand(1) < P_click:
                    if 'P' in int_res[i]:
                        P += 1
                    else:
                        E += 1
            elif '0' in rel:
                P_click = self.gamma[i] * epsilon
                if np.random.rand(1) < P_click:
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

    def computeSampleSize(self, alpha, beta, p0, repetition, int_func, int_args):
        '''Compute the sample size for the interleaving result (DERR).
        Arguments:
            alpha {float} -- Type I error rate
            beta {float} -- Type II error rate
            p0 {float} -- proportion for comparison
            repetition {int} -- number of repetitions for user click simulation
            int_func {list or ndarray} -- interleaved function to use
            int_args {tuple} -- arguments for interleaving function
        Returns:
            int -- computed sample size
        '''

        P = 0
        E = 0
        for _ in np.arange(repetition):
            int_res = int_func(*int_args)
            sim_res = self.simulate(int_res)
            if sim_res == 1:
                E += 1
            elif sim_res == -1:
                P += 1

        p1 = E / (E + P)
        delta = np.abs(p1 - p0)
        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(1 - beta)
        if delta == 0.0:
            N = 0
        else:
            N = ((z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1))) / delta) ** 2 + 1 / delta
        return np.ceil(N)


def getStatistics(groups, clickModel, interleaving, alpha=0.05, beta=0.1, p0=0.5, repetition=50):
    '''Calculate statistics for each groups of ranking pairs.

    Arguments:
        groups {list} -- list of groups
        clickModel {RandomClickModel or PositionBasedModel} -- instance of either click model
        interleaving {string} -- interleaving method, 'teamdraft' or 'prob' 

    Keyword Arguments:
        alpha {float} -- Type I error rate (default: {0.05})
        beta {float} -- Type II error rate (default: {0.1})
        p0 {float} -- comparison proportion (default: {0.5})
        repetition {int} -- number of repetitions (default: {50})

    Returns:
        list -- list of dictionaries containing statistics for each group
    '''
    if 'teamdraft' in interleaving.lower():
        int_func = teamDraftInterleave
    else:
        int_func = probInterleave

    groupStatistics = []
    for group in tqdm(groups, desc='Computing statistics for each group...', ascii=True):
        groupStatistics.append(dict())
        tmp = groupStatistics[-1]
        if len(group) == 0:
            tmp['min'] = 'N/A'
            tmp['median'] = 'N/A'
            tmp['max'] = 'N/A'
            continue
        tmpN = []
        for pair in group:
            int_args = (pair['P'], pair['E'], pair['P_docID'], pair['E_docID'])
            N = clickModel.computeSampleSize(alpha, beta, p0, repetition, int_func, int_args)
            if N == 0.0:
                continue
            tmpN.append(N)
        tmpN = np.array(tmpN)
        tmp['min'] = np.min(tmpN)
        tmp['median'] = np.median(tmpN)
        tmp['max'] = np.max(tmpN)
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

    start = time()
    print('Random Click Model: ')
    print('Processing...')
    rcm = RandomClickModel(docPerPage)
    print('Parameter estimation...')
    rcm.estimateParameters(clickLog)

    print('Team-draft interleaving: ')
    groupStatisticsRCM = getStatistics(groups, rcm, 'teamdraft')
    print('Done!')
    print('Group statistics: ')
    for i in np.arange(len(groupStatisticsRCM)):
        print('Sample size estimation of group', i + 1, ':')
        print(groupStatisticsRCM[i])

    print('Probabilistic interleaving: ')
    groupStatisticsRCM = getStatistics(groups, rcm, 'prob')
    print('Done!')
    print('Group statistics: ')
    for i in np.arange(len(groupStatisticsRCM)):
        print('Sample size estimation of group', i + 1, ':')
        print(groupStatisticsRCM[i])

    print('Position Based Model: ')
    print('Processing...')
    pbm = PositionBasedModel()
    print('Parameter estimation...')
    pbm.learn_by_EM()

    print('Team-draft interleaving: ')
    groupStatisticsPBM = getStatistics(groups, pbm, 'teamdraft')
    print('Done!')
    print('Group statistics: ')
    for i in np.arange(len(groupStatisticsPBM)):
        print('Sample size estimation of group', i + 1, ':')
        print(groupStatisticsPBM[i])

    print('Probabilistic interleaving: ')
    groupStatisticsPBM = getStatistics(groups, pbm, 'prob')
    print('Done!')
    print('Group statistics: ')
    for i in np.arange(len(groupStatisticsPBM)):
        print('Sample size estimation of group', i + 1, ':')
        print(groupStatisticsPBM[i])

    print('Time elapsed: {:.3f}s'.format(time() - start))


if __name__ == "__main__":
    main()
