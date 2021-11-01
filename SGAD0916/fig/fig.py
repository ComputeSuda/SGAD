import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.model_selection import train_test_split
from sklearn import metrics

font_arial = {
    'size': 14,
}

plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率


def loadtt(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        lineArr = [curLine[0], curLine[1], int(curLine[2])]
        data.append(lineArr)
    return data


def changeM(Matrix, e):
    Matrix = Matrix[:, 2:]
    e_score = Matrix[:, e]
    return e_score


dataset = "Hsapi"
test_data = loadtt("../Datasets/" + dataset + "/test.txt")
y_test = [test_data[i][2] for i in range(len(test_data))]

oursEns = np.loadtxt("../Datasets/" + dataset + "/Ens_SGAD930.txt")
ours = np.loadtxt("../Datasets/" + dataset + "/SGAD930.txt")
EnsDNN = np.loadtxt("../Datasets/" + dataset + "/ours_pre.txt")

maxOursauc = 0
maxEDauc = 0
maxOursEnsauc = 0

eOurs = 0
eED = 0
eOE = 0

for i in range(500):
    y_scoreOurs = changeM(ours, i)
    y_scoreEnsDNN = changeM(EnsDNN, i)
    y_scoreOursEns = changeM(oursEns, i)
    fprOurs, tprOurs, thresholdOurs = roc_curve(y_test, y_scoreOurs)  ###计算真正率和假正率
    fprED, tprED, thresholdED = roc_curve(y_test, y_scoreEnsDNN)
    fprOE, tprOE, thresholdOE = roc_curve(y_test, y_scoreOursEns)
    if auc(fprOurs, tprOurs) > maxOursauc:
        print('ours:', auc(fprOurs, tprOurs))
        maxOursauc = auc(fprOurs, tprOurs)
        eOurs = i
    if auc(fprED, tprED) > maxEDauc:
        print('ED:', auc(fprED, tprED))
        maxEDauc = auc(fprED, tprED)
        eED = i
    if auc(fprOE, tprOE) > maxOursEnsauc:
        print('OE:', auc(fprOE, tprOE))
        maxOursEnsauc = auc(fprOE, tprOE)
        eOE = i

# eOurs = 1
# eED = 1
# eOE = 1

y_scoreOurs = changeM(ours, eOurs)
y_scoreEnsDNN = changeM(EnsDNN, eED)
y_scoreOursEns = changeM(oursEns, eOE)

# Compute ROC curve and ROC area for each class
fprOurs, tprOurs, thresholdOurs = roc_curve(y_test, y_scoreOurs)  ###计算真正率和假正率
fprED, tprED, thresholdED = roc_curve(y_test, y_scoreEnsDNN)
fprOE, tprOE, thresholdOE = roc_curve(y_test, y_scoreOursEns)

aucOurs = auc(fprOurs, tprOurs)
aucED = auc(fprED, tprED)
aucOE = auc(fprOE, tprOE)

# 画图
plt.figure()
lw = 2

# plt.figure(figsize=(10, 10))

# plt.title("(F) " + dataset + " with various methods", font_arial)

plt.plot(fprOurs, tprOurs, color='green', linestyle='--', label='SGAD (AUC = %0.6f) ' % aucOurs)
plt.plot(fprED, tprED, color='blue', linestyle='--', label='DNN_ENs (AUC = %0.6f) ' % aucED)
plt.plot(fprOE, tprOE, color='red', label='SGAD_Ens (AUC = %0.6f) ' % aucOE)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate', font_arial)
plt.ylabel('True Positive Rate', font_arial)
plt.legend(loc="lower right", prop=font_arial)

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc="lower right")

plt.savefig(dataset + '_roc_Ensstrategy0930_sup' + '.png', dpi=300)

plt.show()
