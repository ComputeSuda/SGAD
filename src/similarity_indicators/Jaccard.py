# coding=UTF-8
'''
Created on 2016��11��22��

@author: ZWT
'''
import numpy as np
import time
import torch


def Jaccard(MatrixAdjacency_Train):
    similarity_StartTime = time.clock()
    # print('1111111111')
    # print(MatrixAdjacency_Train.shape)
    MatrixAdjacency_Train = torch.FloatTensor(MatrixAdjacency_Train).cuda()
    # print(MatrixAdjacency_Train)
    # Matrix_similarity = np.dot(MatrixAdjacency_Train, MatrixAdjacency_Train)
    Matrix_similarity = torch.mm(MatrixAdjacency_Train, MatrixAdjacency_Train)
    # print('----------------')
    MatrixAdjacency_Train = MatrixAdjacency_Train.cpu()
    # print(MatrixAdjacency_Train.shape)
    MatrixAdjacency_Train = MatrixAdjacency_Train.numpy()
    deg_row = sum(MatrixAdjacency_Train)
    # print(type(deg_row.shape[0]))
    # num_shape = deg_row.shape[0]
    deg_row.shape = (deg_row.shape[0], 1)
    deg_row_T = deg_row.T
    tempdeg = deg_row + deg_row_T
    Matrix_similarity = Matrix_similarity.cpu().numpy()
    temp = tempdeg - Matrix_similarity

    Matrix_similarity = Matrix_similarity / temp
    print(Matrix_similarity.shape)

    return Matrix_similarity

# import numpy.matlib as matlab
# def spones(Matrix):
#     MatrixIndex = np.argwhere(Matrix != 0)
#     for index in range(len(MatrixIndex)):
#         Matrix[MatrixIndex[index,0],MatrixIndex[index,1]] = 1
#     return Matrix
#
# def Jaccavrd(MatrixAdjacency_Train):
#     Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)
#     
#     deg_row = matlab.repmat(matlab.sum(MatrixAdjacency_Train,1), matlab.size(MatrixAdjacency_Train, 1), 1)
#     deg_row = deg_row * spones(Matrix_similarity)
#     deg_row = np.triu(deg_row) + np.triu(deg_row.T)
#     temp = (deg_row * spones(Matrix_similarity)) - Matrix_similarity
#     
#     Matrix_similarity = Matrix_similarity / temp
#     np.seterr(divide='ignore', invalid='ignore')
#     print np.isnan(Matrix_similarity)
#     Matrix_similarity = np.nan_to_num(Matrix_similarity)
#     print np.isnan(Matrix_similarity)
#     return Matrix_similarity
