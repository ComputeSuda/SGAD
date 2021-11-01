#coding=UTF-8
'''
Created on 2016��11��27��

@author: ZWT
'''
import numpy as np
import time

def HPI(MatrixAdjacency_Train):
    similarity_StartTime = time.clock()
    
    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
    
    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (deg_row.shape[0],1)
    deg_row_T = deg_row.T
    tempdeg = np.minimum(deg_row,deg_row_T)
    
    Matrix_similarity = Matrix_similarity / tempdeg
    
    similarity_EndTime = time.clock()

    return Matrix_similarity
