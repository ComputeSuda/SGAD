import numpy as np
import os
import csv
import math
import networkx as nx

import similarity_indicators.CN
import similarity_indicators.Salton
import similarity_indicators.Jaccard
import similarity_indicators.Sorenson
import similarity_indicators.HPI
import similarity_indicators.Cos
import similarity_indicators.AA
import similarity_indicators.ACT
import similarity_indicators.HDI
import similarity_indicators.Katz
import similarity_indicators.LHN_I
import similarity_indicators.LP
import similarity_indicators.PA
import similarity_indicators.RA
import similarity_indicators.RWR


# AC
def Auto_Covariance(seq):
    lg = 30  # will affect 'ac_array' down below
    AC_array = [[0 for u in range(lg)] for v in range(7)]
    mean_feature = [0, 0, 0, 0, 0, 0, 0]
    locate_feature = transfer_feature()  # 提取蛋白质序列中每个氨基酸残基的标准化后的特征
    for j in range(len(mean_feature)):
        for i in range(len(seq)):
            if seq[i] == 'X' or seq[i] == 'U' or seq[i] == ' ' or seq[i] == 'B':
                continue
            mean_feature[j] += locate_feature[seq[i]][j]
    for k in range(len(mean_feature)):
        mean_feature[k] /= len(seq)
    for lag in range(lg):
        for ac_fea in range(len(mean_feature)):
            AC_array[ac_fea][lag] = acsum(seq, lag, mean_feature, ac_fea, locate_feature)
    Auto_Covariance_feature = []
    for o in range(len(AC_array)):
        for p in range(len(AC_array[0])):
            Auto_Covariance_feature += [AC_array[o][p]]
    Auto_Covariance_feature = np.array(Auto_Covariance_feature)
    return Auto_Covariance_feature


def acsum(protein_array, lag, mean_feature, ac_fea, locate_feature):
    phychem_sum = 0
    for i in range(len(protein_array) - lag):
        if (protein_array[i] == 'X' or protein_array[i + lag] == 'X'
                or protein_array[i] == 'U' or protein_array[i + lag] == 'U'
                or protein_array[i] == ' ' or protein_array[i + lag] == ' '
                or protein_array[i] == 'B' or protein_array[i + lag] == 'B'):
            continue
        phychem_sum += (locate_feature[protein_array[i]][ac_fea] - mean_feature[ac_fea]) * (
                    locate_feature[protein_array[i + lag]][ac_fea] - mean_feature[ac_fea])
    phychem_sum /= (len(protein_array) - lag + 0.0000001)
    return phychem_sum


def CodingAC(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinAC = dict()
    for p_key in protein.keys():
        if len(protein[p_key]) == 0:
            print(p_key)
        Ac = Auto_Covariance(protein[p_key])
        # print(embeddings[p_key])
        proteinAC[p_key] = np.array(list(Ac))
        count = count + 1
        if count % 200 == 0:
            print('AC:', count)
        # print(proteinAC[p_key])
    return proteinAC


# CT
def conjoint_triad(seq):
    local_operate_array = aac_7_number_description(seq)
    vector_3_matrix = [[a, b, c, 0] for a in range(1, 8) for b in range(1, 8) for c in range(1, 8)]
    for m in range(len(local_operate_array) - 2):
        vector_3_matrix[(local_operate_array[m] - 1) * 49 + (local_operate_array[m + 1] - 1) * 7 + (
                    local_operate_array[m + 2] - 1)][3] += 1
    CT_array = []
    for q in range(343):
        CT_array += [vector_3_matrix[q][3]]
    CT_array = np.array(CT_array)
    return CT_array


def CodingCT(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinCT = dict()
    for p_key in protein.keys():
        if len(protein[p_key]) == 0:
            print(p_key)
        Ct = conjoint_triad(protein[p_key])
        proteinCT[p_key] = np.array(list(Ct))
        count = count + 1
        if count % 200 == 0:
            print('CT:', count)
    return proteinCT


# LD
def local_descriptors(seq):
    local_operate_array = aac_7_number_description(seq)
    A_point = math.floor(len(seq) / 4) - 1
    B_point = A_point * 2 + 1
    C_point = A_point * 3 + 2
    part_vector = []
    part_vector += construct_63_vector(local_operate_array[0:A_point])
    part_vector += construct_63_vector(local_operate_array[A_point:B_point])
    part_vector += construct_63_vector(local_operate_array[B_point:C_point])
    part_vector += construct_63_vector(local_operate_array[C_point:])
    part_vector += construct_63_vector(local_operate_array[0:B_point])
    part_vector += construct_63_vector(local_operate_array[B_point:])
    part_vector += construct_63_vector(local_operate_array[A_point:C_point])
    part_vector += construct_63_vector(local_operate_array[0:C_point])
    part_vector += construct_63_vector(local_operate_array[A_point:])
    part_vector += construct_63_vector(local_operate_array[math.floor(A_point / 2):math.floor(C_point - A_point / 2)])
    part_vector = np.array(part_vector)
    return part_vector


def construct_63_vector(part_array):
    simple_7 = [0 for n in range(7)]
    marix_7_7 = [[0 for n in range(7)] for m in range(7)]
    simple_21 = [0 for n in range(21)]
    simple_35 = [0 for n in range(35)]
    for i in range(len(part_array)):
        simple_7[part_array[i] - 1] += 1
        if i < (len(part_array) - 1) and part_array[i] != part_array[i + 1]:
            if part_array[i] > part_array[i + 1]:
                j, k = part_array[i + 1], part_array[i]
            else:
                j, k = part_array[i], part_array[i + 1]
            marix_7_7[j - 1][k - 1] += 1
    i = 0
    for j in range(7):
        for k in range(j + 1, 7):
            simple_21[i] = marix_7_7[j][k]
            i += 1
    residue_count = [0, 0, 0, 0, 0, 0, 0]
    for q in range(len(part_array)):
        residue_count[part_array[q] - 1] += 1
        if residue_count[part_array[q] - 1] == 1:
            simple_35[5 * (part_array[q] - 1)] = q + 1
        elif residue_count[part_array[q] - 1] == math.floor(simple_7[part_array[q] - 1] / 4):
            simple_35[5 * (part_array[q] - 1) + 1] = q + 1
        elif residue_count[part_array[q] - 1] == math.floor(simple_7[part_array[q] - 1] / 2):
            simple_35[5 * (part_array[q] - 1) + 2] = q + 1
        elif residue_count[part_array[q] - 1] == math.floor(simple_7[part_array[q] - 1] * 0.75):
            simple_35[5 * (part_array[q] - 1) + 3] = q + 1
        elif residue_count[part_array[q] - 1] == simple_7[part_array[q] - 1]:
            simple_35[5 * (part_array[q] - 1) + 4] = q + 1
    for o in range(7):
        simple_7[o] /= len(part_array)
    for p in range(21):
        simple_21[p] /= len(part_array)
    for m in range(35):
        simple_35[m] /= len(part_array)
    simple_63_vector = simple_7 + simple_21 + simple_35
    return simple_63_vector


def aac_7_number_description(protein_array):
    # 将蛋白质序列中的氨基酸进行分类
    local_operate_array = []
    for i in range(len(protein_array)):
        if protein_array[i] in 'AGV':
            local_operate_array += [1]
        elif protein_array[i] in 'ILFP':
            local_operate_array += [2]
        elif protein_array[i] in 'YMTS':
            local_operate_array += [3]
        elif protein_array[i] in 'HNQW':
            local_operate_array += [4]
        elif protein_array[i] in 'RK':
            local_operate_array += [5]
        elif protein_array[i] in 'DE':
            local_operate_array += [6]
        elif protein_array[i] == 'C':
            local_operate_array += [7]
        else:
            local_operate_array += [7]
    return local_operate_array


def CodingLD(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinLD = dict()
    for p_key in protein.keys():
        if len(protein[p_key]) == 0:
            print(p_key)
        Ld = local_descriptors(protein[p_key])
        # print(Ld)
        # print(type(Ld))
        proteinLD[p_key] = np.array(list(Ld))
        count = count + 1
        if count % 200 == 0:
            print('LD:', count)
    return proteinLD


# PseAAC
def PseAAC(seq):
    nambda = 15  #
    omega = 0.05  #
    locate_feature = transfer_feature()  # 提取蛋白质序列中每个氨基酸残基的标准化后的特征
    AA_frequency = {'A': [0], 'C': [0], 'D': [0], 'E': [0], 'F': [0], 'G': [0], 'H': [0], 'I': [0], 'K': [0], 'L': [0],
                    'M': [0], 'N': [0], 'P': [0], 'Q': [0], 'R': [0], 'S': [0], 'T': [0], 'V': [0], 'W': [0], 'Y': [0]}
    A_class_feature = [0 for v in range(20)]  # 全为0的列表
    B_class_feature = []
    sum_frequency = 0
    sum_occurrence_frequency = 0
    for i in range(len(seq)):
        if seq[i] == 'X' or seq[i] == 'U' or seq[i] == 'B':
            continue
        AA_frequency[seq[i]][0] += 1
    for j in AA_frequency:
        sum_frequency += AA_frequency[j][0]
    for m in AA_frequency:
        if sum_frequency == 0:
            s = [0 for b in range(35)]
            return s
        else:
            AA_frequency[m][0] /= sum_frequency
    for o in AA_frequency:
        sum_occurrence_frequency += AA_frequency[o][0]

    for k in range(1, nambda + 1):
        B_class_feature += [thet(seq, locate_feature, k)]
    Pu_under = sum_occurrence_frequency + omega * sum(B_class_feature)
    for l in range(nambda):
        B_class_feature[l] = (B_class_feature[l] * omega / Pu_under) * 100
    number_range = range(len(AA_frequency))
    for charater, number in zip(AA_frequency, number_range):
        A_class_feature[number] = AA_frequency[charater][0] / Pu_under * 100
    class_feature = A_class_feature + B_class_feature
    class_feature = np.array(class_feature)
    return class_feature


def thet(seq, locate_feature, t):
    sum_comp = 0
    for i in range(len(seq) - t):
        sum_comp += comp(seq[i], seq[i + t], locate_feature)
    if (len(seq) - t) == 0:
        sum_comp /= (len(seq) - t + 1)
    else:
        sum_comp /= (len(seq) - t)
    return sum_comp


def comp(Ri, Rj, locate_feature):
    theth = 0
    if Ri == 'X' or Rj == 'X' or Ri == 'U' or Rj == 'U' or Ri == 'B' or Rj == 'B':
        return 0
    else:
        for i in range(3):
            theth += pow(locate_feature[Ri][i] - locate_feature[Rj][i], 2)
        theth = theth / 3
        return theth


def transfer_feature():
    opposite_path = os.path.abspath('')
    with open(os.path.join(opposite_path, '../normalized_feature.csv')) as C:
        normalized_feature = csv.reader(C)
        feature_hash = {}
        amino_acid = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                      'Y']
        for charater in amino_acid:
            feature_hash[charater] = []
        for row in normalized_feature:
            i = 0
            for charater in amino_acid:
                feature_hash[charater] += [float(row[i])]
                i += 1
    return feature_hash


def CodingPseAAC(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinPseAAC = dict()
    for p_key in protein.keys():
        PseA = PseAAC(protein[p_key])
        # print(type(PseA))
        # print(PseA)
        proteinPseAAC[p_key] = np.array(list(PseA))
        count = count + 1
        if count % 200 == 0:
            print('PseAAC:', count)
    return proteinPseAAC


def getnewList(newlist):
    d = []
    for element in newlist:
        if not isinstance(element, list):
            d.append(element)
        else:
            d.extend(getnewList(element))
    return d


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    count = 0
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
        count = count + 1
    file.close()
    print("save = %d" % count)
    print("success")


def loadtt(fileName):
    data = []
    # numFeat = len(open(fileName).readline().split('\t'))
    # print(numFeat)
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        lineArr = [curLine[0], curLine[1], int(curLine[2])]
        data.append(lineArr)
    return data


def loadGdata(fileName):
    data = []
    # numFeat = len(open(fileName).readline().split('\t'))
    # print(numFeat)
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        if curLine[2] == '1':
            lineArr = [curLine[0], curLine[1]]
            data.append(lineArr)
    return data


def Init22(Test_File, Train_File, proteinID):
    print("DataShape......")
    g_train = loadGdata(Train_File)
    g_test = loadGdata(Test_File)
    g_data = g_train + g_test
    g = nx.Graph()
    g.add_nodes_from(proteinID)
    g.add_edges_from(g_data)
    nodelist = list(g.nodes())
    Nodecount = len(nodelist)
    nodedic = dict(zip(list(g.nodes()), range(Nodecount)))
    MatrixAdjacency_Train = np.zeros([Nodecount, Nodecount], dtype=np.int32)
    print("Nodecount  " + str(Nodecount))
    nodedic = dict(zip(list(g.nodes()), range(Nodecount)))
    for i in range(len(g_train)):
        # print(i)
        MatrixAdjacency_Train[nodedic[g_train[i][0]], nodedic[g_train[i][1]]] = 1
        MatrixAdjacency_Train[nodedic[g_train[i][1]], nodedic[g_train[i][0]]] = 1
    # print('------------------------------')
    Matrix_similarity = similarity_indicators.Jaccard.Jaccard(MatrixAdjacency_Train)
    # print('----------------')
    # print(Matrix_similarity.shape)
    where_are_nan = np.isnan(Matrix_similarity)
    Matrix_similarity[where_are_nan] = -1
    # print('---------------------------')
    return Matrix_similarity, nodedic, nodelist


def one_hot(seq):
    amino_acid = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # print(len(amino_acid))
    OneHots = dict()
    for a in amino_acid:
        OneHots[a] = 0
    for s in seq:
        if s.upper() in amino_acid:
            OneHots[s] += 1
        else:
            print('error----------')
            print(s)
            continue

    list_frequent = []
    for value in OneHots.values():
        list_frequent.append(value)
    # print(list(OneHots.items()))
    # print(np.array(list(OneHots.values())).shape)
    return list_frequent


def CodingOH(protein):
    # 获取字典形式后根据序列进行编码，用字典进行保存所有的蛋白质的编码形式并返回
    count = 0
    proteinOH = dict()
    for p_key in protein.keys():
        if len(protein[p_key]) == 0:
            print(p_key)
        Oh = one_hot(protein[p_key])
        # print(Oh)
        # print(type(Oh))
        proteinOH[p_key] = np.array(list(Oh)).astype(float)
        count = count + 1
        if count % 200 == 0:
            print(count)
    return proteinOH


# print(CodingOH({'111': 'MAETVWSTDTGEAVYRSRDPVRNLRLRVHLQRITSSNFLHYQPAAELGKDLIDLATFRPQPTASGHRPEEDEEEEIVIGWQEKLFSQFEVDLYQNETACQSPLDYQYRQEILKLENSGGKKNRRIFTYTDSDRYTNLEEHCQRMTTAASEVPSFLVERMANVRRRRQDRRGMEGGILKSR'}))

# test1 = CodingOH({'111': 'MAETVWSTDTGEAVYRSRDDRRGMEGGILKSR'})
# test2 = CodingCT({'111': 'MAETVWSTDTGEAVYRSRDPVRNLRLRVHLQRITSSNFLHYQPAAELGKDLIDLATFRPQPTASGHRPEEDEEEEIVIGWQEKLFSQFEVDLYQNETACQSPLDYQYRQEILKLENSGGKKNRRIFTYTDSDRYTNLEEHCQRMTTAASEVPSFLVERMANVRRRRQDRRGMEGGILKSR'})
# print(test1)
# print(test2)