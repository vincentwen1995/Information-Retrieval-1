#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:14:38 2019

@author: YIJIE
"""
import numpy as np 

def data_preprocessing(max_rank=3, file_path="YandexRelPredChallenge.txt"):
    '''Preprocessing Yandex log to get convient data structures 
    calcuting gamma and alpha using EM.
    
    Description for S_uq(updating alpha): 
        S_uq is a dict, each key is a (document_id, query_id) pair, 
        corresponding value is a list, with each element a tuple 
        (rank, observed click) recording information about each s in S_uq. For
        example, S_uq: {[(document_id, query_id)]:[(rank, observed click),...]}
    
    Description for S(updating gamma): 
        S_uq is a dict, each key is a session id(the row number of row containing
        query in Yandex log, not for row containing click), corresponding value 
        is a list, with each element recording information about (document_id, query_id)
        and observed click on rank 1-3 for every s in S. For example, 
        S: {[document_id]:[((document_id, query_id),observed click),...]}
    
    Arguments:
        max_rank {int} -- by default 3
        path {str} -- file path
    Returns:
        S_uq {dict} -- data structure for calculating gamma using EM
        S {dict} -- data structure for calculating gamma using EM
    '''
    
    # init
    S_uq = {} 
    S = {} 
        
    # read Yandex log    
    with open(file_path) as f:
        data = []
        for line in f:
            data.append(line.rstrip('\n').split('\t'))

    for i in range(len(data)): # for every row in Yandex log
        record = data[i]
        
        # if the record is a query
        if record[2] == "Q":
            
            S[i] = [] # init S for session s
            docs_last_q = [] # docs in the last query, for examining click
            id_last_s = i # id of the last session, for examining click
            
            for r in range(max_rank): # for rank from 1-3
                query_id = record[3]
                doc_id = record[r+5]
                docs_last_q.append(doc_id)
                
                key = (doc_id, query_id)
                rank = r
                c_us = 0
                
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


def learn_by_EM(gamma_0=0.5, alpha_0=0.5):
    '''Learning gamma and alpha using EM 
    calcuting gamma and alpha using EM.
    
    Arguments:
        gamma_0 {float} -- initialized value for gamma 
        alpha_0 {float} -- initialized value for alpha
    Returns:
        gamma_t1 {np.array} -- updated gamma
    '''
    
    # get data structure 
    S_uq, S = preprocessing()
    
    # retrieve all uq pairs
    all_uq = list(S_uq.keys()) 
    
    # initialize gamma
    gamma_t = np.array([gamma_0]*3)
    gamma_t1 = gamma_t.copy()
    
    # init alpha
    alpha_t = {}
    
    for uq in all_uq:
        alpha_t[uq] = alpha_0
    alpha_t1 = alpha_t.copy()
    
    # beging learning 
    for t in range(100): 
        
        # update alpha
        for uq in all_uq: 
            alpha_sum = 0 # for summation
            s_uq = S_uq[uq] # retrieve all sessions s in S_uq
            length_s_uq = len(s_uq) 
            for s in s_uq: # for every session s in S_uq
                rank = s[0]
                c_us = s[1] 
                alpha_sum += (c_us + (1 - c_us) * (1 - gamma_t[rank]) * alpha_t[uq] / (1 - gamma_t[rank] * alpha_t[uq]))
            
            alpha_t1[uq] = alpha_sum / length_s_uq 
        
        # update gamma
        length_S = len(S) 
        for r in range(3): # rank from 1-3
            gamma_sum = 0 # for summation
            for i in S.keys(): # for every session s in S
                s_i = S[i] 
                uq = s_i[r][0]
                c_us = s_i[r][1]
                gamma_sum += (c_us + (1 - c_us) * gamma_t[r] * (1 - alpha_t[uq]) / (1 - gamma_t[r] * alpha_t[uq]))
        
            gamma_t1[r] = gamma_sum / length_S
        
        # update values at t using values at t+1
        gamma_t = gamma_t1.copy()
        alpha_t = alpha_t1.copy()
        
        if t % 5 == 0:
            print(gamma_t1)
    
    return gamma_t1

gamma_t1 = learn_by_EM()


            
            
    
    
    
    
    
    
    
    
    
    
    
        
    