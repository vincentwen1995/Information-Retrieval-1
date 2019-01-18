import random as rn
import numpy as np


class rankingPair:
    '''Class for pair of ranking of relevance of P and E.
    '''

    def __init__(self, P, E, k, scale):
        '''Constructor.

        Arguments:
            P {list} -- ranking of relevance for P
            E {list} -- ranking of relevance for E
            k {int} -- cut-off rank for calculating ERR
            scale {int} -- relevance grading scale
        '''
        self.P = P
        self.E = E
        self.k = k
        self.scale = scale

    def computeThetas(self):
        '''Compute the theta parameters for PBM model.
        '''
        self.thetas = np.array(shape=3, dtype=np.float32)

    def computeDERR(self, parameter_list):
        pass
