def readSeq(ID):
    Path = "../fasta/"
    filename = Path + ID + ".fasta"
    fr = open(filename)
    next(fr)
    Seq = fr.read().replace("\n", "")
    fr.close()
    return Seq


filepath = "../Datasets/"
# filepath = "../1-3Datasets/"
# filepath = "../1-5Datasets/"


def load(fileName):
    fileName = filepath + fileName
    data = []
    # numFeat = len(open(fileName).readline().split('\t'))
    # print(numFeat)
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        # print(curLine)
        lineArr = (curLine[0], curLine[1])
        data.append(lineArr)
    fr.close()
    return data
