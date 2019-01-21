import numpy as np

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

        for i in range(len(data)): # for every row in Yandex log
            record = data[i]
            action = record[2]

            # if the record is a query
            if action == "Q":
                S[i] = [] # initialize session i
                docs_last_q = [] # docs in the last query, for examining click
                id_last_s = i # id of the last session, for examining click

                for r in range(max_rank): # for rank from 1 until max_rank
                    query_id = record[3]
                    doc_id = record[r+5]
                    key = (doc_id, query_id)

                    rank = r
                    c_us = 0

                    docs_last_q.append(doc_id)

                    # update S_uq
                    if key not in S_uq:
                        S_uq[key] = [(rank,c_us)]
                    else:
                        S_uq[key].append((rank,c_us))

                    # update S
                    S[i].append((key, c_us))

            # if the record is a click
            else:
                doc_id = record[-1]
                key = (doc_id, query_id) # session of a click is the last query

                if doc_id in docs_last_q: # if the doc clicked is in the last query
                    rank = S_uq[key][-1][0] # rank of the doc clicked in the last session
                    c_us = 1

                    # update S_uq
                    S_uq[key][-1] = (rank,c_us)

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
        gamma_t = np.array([gamma_0]*max_rank)
        gamma_t1 = np.array([0.9]*max_rank)

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
                s_uq = S_uq[uq] # retrieve all sessions s in S_uq
                alpha_sum = 0 # for summation
                length_s_uq = len(s_uq)
                for s in s_uq: # for every session s in S_uq
                    rank = s[0]
                    c_us = s[1]
                    alpha_sum += (c_us + (1 - c_us) * (1 - gamma_t[rank]) * alpha_t[uq] / (1 - gamma_t[rank] * alpha_t[uq]))
                alpha_t1[uq] = (alpha_sum + 1) / (length_s_uq + 2)

            # update gamma
            length_S = len(S)
            for rank in range(max_rank):
                gamma_sum = 0 # for summation
                for i in S.keys(): # for every session s in S
                    s_i = S[i]
                    uq = s_i[rank][0]
                    c_us = s_i[rank][1]
                    gamma_sum += (c_us + (1 - c_us) * gamma_t[rank] * (1 - alpha_t[uq]) / (1 - gamma_t[rank] * alpha_t[uq]))
                gamma_t1[rank] = (gamma_sum + 1) / (length_S + 2)

            print(gamma_t1)

            total_diff = np.sum(gamma_t1 - gamma_t)

            # update values at t using values at t+1
            gamma_t = gamma_t1.copy()
            alpha_t = alpha_t1.copy()

        self.gamma = gamma_t1

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
            if np.random.rand(1) < self.gamma[i]:
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

a = PositionBasedModel()
a.learn_by_EM()