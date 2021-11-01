from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
from itertools import chain
from sklearn import metrics
import torch.nn.functional as F
from sklearn import preprocessing
import warnings
from features import *
from Data_loader import *

warnings.filterwarnings("ignore")

from Evaluation_Indicators.AUC import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def Indd_stacking2(clf, train_data, test_data, OneEncoded, n_folds=5):
    train_num, test_num = len(train_data), len(
        test_data)  # The number of training sets is 6204, the number of test sets is 1552
    second_level_train_set = np.zeros((
        train_num,))  # np.zeros((n,)) is an n-dimensional array, in the form of 1*n, corresponding to the second-level training set
    second_level_test_set = np.zeros((
        test_num,))  # np.zeros((n,)) is an n-dimensional array in the form of 1*n, corresponding to the second layer test set
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)  # 5-fold crossover experiment

    train_data_ = [(np.array([np.concatenate((OneEncoded[train_data[i][0]], train_data[i][3][0]), axis=0),
                              np.concatenate((OneEncoded[train_data[i][1]], train_data[i][3][1]), axis=0)]),
                    train_data[i][2]) for i in range(len(train_data))]
    test_data_ = [(np.array([np.concatenate((OneEncoded[test_data[i][0]], test_data[i][3][0]), axis=0),
                             np.concatenate((OneEncoded[test_data[i][1]], test_data[i][3][1]), axis=0)]),
                   test_data[i][2]) for i in range(len(test_data))]

    # Set hyperparameters
    # num_epochs = 2     # Number of iterations
    net = clf(Nodecount, args.nhid0, args.nhid1, args.dropout, args.alpha)
    criterion = nn.CrossEntropyLoss()  # Define the loss function as cross entropy
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # Use gradient descent to optimize
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for i, (train_index, test_index) in enumerate(
            kf.split(train_data_)):  # Divide the 6204 pieces of data in the training set into 5 parts
        print("n_folds = %d" % i)
        tra = np.array(train_data_)[train_index]
        tst = np.array(train_data_)[test_index]
        tra_ = [(torch.from_numpy(tra[i][0]), torch.from_numpy(np.array(tra[i][1]))) for i in range(len(tra))]
        tst_ = [(torch.from_numpy(tst[i][0]), torch.from_numpy(np.array(tst[i][1]))) for i in range(len(tst))]
        test_ = [(torch.from_numpy(test_data_[i][0]), torch.from_numpy(np.array(test_data_[i][1]))) for i in
                 range(len(test_data_))]
        tra_ = DataLoader(tra_, batch_size=256, shuffle=True)
        tst_ = DataLoader(tst_, batch_size=256, shuffle=False)
        test_ = DataLoader(test_, batch_size=128, shuffle=False)
        second_level_train_set[test_index], test_nfolds_sets[:, i] = train(net, tra_, tst_, test_, num_epochs,
                                                                           optimizer, criterion, scheduler)
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)

    return second_level_train_set, second_level_test_set


def train(net, train_data, valid_data, x_test, num_epochs, optimizer, criterion, scheduler):
    if torch.cuda.is_available():
        net = net.cuda()
    output_valid_data = []
    output_x_test = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for a, a_label in train_data:
            a_label = a_label.long()
            a = torch.tensor(a, dtype=torch.float32)
            if torch.cuda.is_available():
                a = Variable(a.cuda())  # (64, 2, 128, 128)
                a_label = Variable(a_label.cuda())  # (64, 128, 128)
            else:
                a = Variable(a)  # Ideally, the dimension of a is 2*128*128
                a_label = Variable(a_label)
            # forward
            output, output_label, L_1st, L_2nd, L_all = net(a)
            L_reg = 0
            for param in net.parameters():
                L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
            loss = criterion(output, a_label) + L_all + L_reg
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(epoch)
        if valid_data is not None:
            Acc = 0
            Recall = 0
            Specificity = 0
            Precision = 0
            MCC = 0
            F1 = 0
            AUC = 0
            net = net.eval()
            for a, a_label in valid_data:
                a_label = a_label.long()
                a = torch.tensor(a, dtype=torch.float32)
                if torch.cuda.is_available():
                    a = Variable(a.cuda(), volatile=True)  # (64, 2, 128, 128)
                    a_label = Variable(a_label.cuda(), volatile=True)  # (64, 128, 128)
                else:
                    a = Variable(a, volatile=True)
                    a_label = Variable(a_label, volatile=True)
                output, output_label, L_1st, L_2nd, L_all = net(a)
                L_reg = 0
                for param in net.parameters():
                    L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
                loss = criterion(output, a_label) + L_all + L_reg
                _, pred_label_valid_data = output.max(1)
                loss.item()
                if epoch == num_epochs - 1:
                    output_valid_data.append(pred_label_valid_data)
                y_pred = output.max(1)[1]
                Acc += metrics.accuracy_score(a_label.cpu(), y_pred.cpu())
                Recall += metrics.recall_score(a_label.cpu(), y_pred.cpu())
                Precision += metrics.precision_score(a_label.cpu(), y_pred.cpu())
                F1 += metrics.f1_score(a_label.cpu(), y_pred.cpu())
                AUC += metrics.roc_auc_score(a_label.cpu(), y_pred.cpu())
            epoch_str = (
                    "Epoch %d. Accuracy: %f, Recall: %f, Precision: %f, F1: %f, AUC: %f"
                    % (epoch, Acc / len(valid_data), Recall / len(valid_data),
                       Precision / len(valid_data), F1 / len(valid_data),
                       AUC / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        print(epoch_str)
        output_x_test = []
        if x_test is not None:
            x_test_loss = 0
            x_test_acc = 0
            net = net.eval()
            for a, a_label in x_test:
                a_label = a_label.long()
                a = torch.tensor(a, dtype=torch.float32)
                if torch.cuda.is_available():
                    a = Variable(a.cuda(), volatile=True)  # (128, 2, 128, 128)
                    a_label = Variable(a_label.cuda(), volatile=True)  # (64, 128, 128)
                else:
                    a = Variable(a, volatile=True)
                    a_label = Variable(a_label, volatile=True)
                output, output_label, L_1st, L_2nd, L_all = net(a)
                _, pred_label_x_test = output.max(1)
                if epoch == num_epochs - 1:
                    output_x_test.append(pred_label_x_test)
    output_valid_data = [output_valid_data[i] for i in range(len(output_valid_data))]
    output_valid_data = np.array(list(chain(*output_valid_data)))
    output_x_test = [output_x_test[i] for i in range(len(output_x_test))]
    output_x_test = np.array(list(chain(*output_x_test)))
    return output_valid_data, output_x_test


# Models
class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class multihead_attention(nn.Module):
    def __init__(self, num_units, num_heads=4, dropout_rate=0, causality=False):
        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.output_dropout = nn.Dropout(p=self.dropout_rate)
        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        if torch.cuda.is_available():
            queries = queries.cuda()
            keys = keys.cuda()
            values = values.cuda()
        Q = self.Q_proj(queries)  # (N, T_q, C)     torch.Size([512, 64, 64])
        K = self.K_proj(keys)  # (N, T_q, C)        torch.Size([512, 64, 64])
        V = self.V_proj(values)  # (N, T_q, C)      torch.Size([512, 64, 64])
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     torch.Size([2048, 64, 16])
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     torch.Size([2048, 64, 16])
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)     torch.Size([2048, 64, 16])
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)
        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)
        if torch.cuda.is_available():
            padding = Variable(torch.ones(*outputs.size()) * (-2 ** 32 + 1)).cuda()
        else:
            padding = Variable(torch.ones(*outputs.size()) * (-2 ** 32 + 1))
        # print(padding.device)
        condition = key_masks.eq(0.).float()
        # print(condition.device)
        outputs = padding * condition + outputs * (1. - condition)
        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size())  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
            # print(tril)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)
            padding = Variable(torch.ones(*masks.size()) * (-2 ** 32 + 1))
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1. - condition)
        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = self.normalization(outputs)  # (N, T_q, C)
        return outputs


def block(in_channel, out_channel, stride=1):
    layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                          nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False))
    return layer


class cnn_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(cnn_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        self.conv1 = block(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = block(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # if not self.same_shape:
        self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), True)
        x = self.conv3(x)
        return x


class cnn2_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(cnn2_block, self).__init__()
        self.conv1 = block(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = block(out_channel, out_channel)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = self.conv2(x)
        # x = self.bn1(x)
        return x


class Model4(nn.Module):
    """210"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model4, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(210 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3),
                                       nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(210, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[210, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[210, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model41(nn.Module):
    """210"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model41, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256

        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount

        self.clf_layer = nn.Sequential(nn.Linear(210 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.BatchNorm1d(self.dense_dim2), nn.Dropout(0.2),
                                       nn.Linear(self.dense_dim2, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(210, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[210, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[210, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model42(nn.Module):
    """210"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model42, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(210 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.Dropout(0.3), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),
                                       nn.Linear(self.dense_dim3, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(210, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[210, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[210, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model43(nn.Module):
    """210"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model43, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(210 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.Linear(self.dense_dim1, self.dense_dim2),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(210, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[210, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[210, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model44(nn.Module):
    """210"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model44, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(210 * 2 + 4 * 16 + 3, self.dense_dim1), nn.Linear(self.dense_dim1, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(210, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[210, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[210, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model45(nn.Module):
    """210"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model45, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 64
        self.dense_dim3 = 16
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(210 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.BatchNorm1d(self.dense_dim2),
                                       nn.Linear(self.dense_dim2, self.dense_dim3),
                                       nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(210, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[210, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[210, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model46(nn.Module):
    """210"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model46, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(210 * 2 + 4 * 16 + 3, self.dense_dim1), nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.BatchNorm1d(self.dense_dim2),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(210, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[210, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[210, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model5(nn.Module):
    """630"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model5, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 64, 2 * 64, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 64, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(630 * 2 + 4 * 64 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3),
                                       nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(630, 64 * 64)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[630, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[630, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model51(nn.Module):
    """630"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model51, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 64, 2 * 64, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 64, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(630 * 2 + 4 * 64 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1),
                                       nn.Linear(self.dense_dim1, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(630, 64 * 64)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[630, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[630, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model52(nn.Module):
    """630"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model52, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 64, 2 * 64, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 64, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Linear(630 * 2 + 4 * 64 + 3, 2)

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(630, 64 * 64)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[630, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[630, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model53(nn.Module):
    """630"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model53, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 64, 2 * 64, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 512
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(630 * 2 + 4 * 64 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3), nn.Dropout(0.2),
                                       nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(630, 64 * 64)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[630, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[630, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model54(nn.Module):
    """630"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model54, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 64, 2 * 64, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 512
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(630 * 2 + 4 * 64 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),
                                       nn.BatchNorm1d(self.dense_dim3),
                                       nn.Linear(self.dense_dim3, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(630, 64 * 64)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[630, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[630, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model55(nn.Module):
    """630"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model55, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 64, 2 * 64, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 256
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(630 * 2 + 4 * 64 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(630, 64 * 64)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[630, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[630, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model56(nn.Module):
    """630"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model56, self).__init__()
        self.attention = multihead_attention(64)
        self.pro_len = 64
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 64, 2 * 64, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 64, 1)
        self.dense_dim1 = 1024
        self.dense_dim2 = 512
        self.dense_dim3 = 64
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(630 * 2 + 4 * 64 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.BatchNorm1d(self.dense_dim2),
                                       nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),
                                       nn.Linear(self.dense_dim3, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(630, 64 * 64)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 64, 64)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[630, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[630, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model6(nn.Module):
    """343"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model6, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(343 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3),
                                       nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(343, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[343, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[343, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model61(nn.Module):
    """343"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model61, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(343 * 2 + 4 * 16 + 3, self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.BatchNorm1d(self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, self.dense_dim3), nn.ELU(),
                                       nn.Linear(self.dense_dim3, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(343, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[343, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[343, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)
        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model62(nn.Module):
    """343"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model62, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.dense_dim3 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(343 * 2 + 4 * 16 + 3, self.dense_dim1), nn.Dropout(0.2),
                                       nn.Linear(self.dense_dim1, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(343, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[343, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[343, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model63(nn.Module):
    """343"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model63, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(343 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(343, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[343, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[343, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model64(nn.Module):
    """343"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model64, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(343 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(), nn.Dropout(0.2),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(343, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[343, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[343, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model65(nn.Module):
    """343"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model65, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 256
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(343 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.Dropout(0.2), nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(), nn.Dropout(0.2),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(343, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[343, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[343, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model66(nn.Module):
    """343"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model66, self).__init__()
        self.attention = multihead_attention(16)
        self.pro_len = 16
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 16, 2 * 16, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 16, 1)
        self.dense_dim1 = 512
        self.dense_dim2 = 128
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(343 * 2 + 4 * 16 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1),  # nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(), nn.Dropout(0.2),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(343, 1 * 16 * 16)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 16, 16)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[343, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[343, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model7(nn.Module):
    """35"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model7, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 8, 2 * 8, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 8, 1)
        self.clf_layer = nn.Linear(4 * 8 + 3, 2)
        self.dense_dim1 = 64
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(35 * 2 + 4 * 8 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.ELU(),
                                       nn.Linear(self.dense_dim1, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(35, 8 * 8)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[35, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[35, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model71(nn.Module):
    """35"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model71, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 8, 2 * 8, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 8, 1)
        self.clf_layer = nn.Linear(4 * 8 + 3, 2)
        self.dense_dim1 = 32
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(35 * 2 + 4 * 8 + 3, self.dense_dim1), nn.ELU(),
                                       nn.Linear(self.dense_dim1, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(35, 8 * 8)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[35, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[35, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model72(nn.Module):
    """35"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model72, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 8, 2 * 8, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 8, 1)
        self.clf_layer = nn.Linear(4 * 8 + 3, 2)
        self.dense_dim1 = 64
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Linear(35 * 2 + 4 * 8 + 3, 2)

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(35, 8 * 8)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[35, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[35, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model73(nn.Module):
    """35"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model73, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 8, 2 * 8, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 8, 1)
        self.clf_layer = nn.Linear(4 * 8 + 3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 32
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(35 * 2 + 4 * 8 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(35, 8 * 8)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[35, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[35, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model74(nn.Module):
    """35"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model74, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 8, 2 * 8, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 8, 1)
        self.clf_layer = nn.Linear(4 * 8 + 3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 16
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(35 * 2 + 4 * 8 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1), nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(35, 8 * 8)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[35, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[35, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model75(nn.Module):
    """35"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model75, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 8, 2 * 8, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 8, 1)
        self.clf_layer = nn.Linear(4 * 8 + 3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 16
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(35 * 2 + 4 * 8 + 3, self.dense_dim1),
                                       nn.BatchNorm1d(self.dense_dim1),
                                       nn.Linear(self.dense_dim1, self.dense_dim2),
                                       nn.BatchNorm1d(self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, 2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(35, 8 * 8)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[35, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[35, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


class Model76(nn.Module):
    """35"""

    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(Model76, self).__init__()
        self.attention = multihead_attention(8)
        self.pro_len = 8
        self.cos_sim = nn.CosineSimilarity()
        self.Bilinear_sim = nn.Bilinear(2 * 8, 2 * 8, 1, bias=False)
        self.linear_sim = nn.Linear(4 * 8, 1)
        self.clf_layer = nn.Linear(4 * 8 + 3, 2)
        self.dense_dim1 = 64
        self.dense_dim2 = 16
        self.encode0 = nn.Linear(node_size * 2, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size * 2)
        self.droput = droput
        self.alpha = alpha
        self.c = Nodecount
        self.clf_layer = nn.Sequential(nn.Linear(35 * 2 + 4 * 8 + 3, self.dense_dim1), nn.ELU(),
                                       nn.Linear(self.dense_dim1, self.dense_dim2), nn.ELU(),
                                       nn.Linear(self.dense_dim2, 2), nn.BatchNorm1d(2))

    def GatedCNN(self, input_):
        input_ = torch.tensor(input_, dtype=torch.float32).cpu()
        netliearn = nn.Linear(35, 8 * 8)
        input_0 = netliearn(input_)  # input_0.shape = (512, 1, 256)
        input_0 = input_0.reshape(len(input_0), 1, 8, 8)  # input_0: torch.Size([512, 1, 16, 16])
        net = cnn2_block(1, 1)
        out = net(input_0)  # [512, 1, 16, 16] -- > GatedCNN--out: torch.Size([512, 1, 16, 16])
        return out

    def aggregate_layers(self, input_):
        "Apply layers to input then concatenate result"
        avglayer = nn.AvgPool1d(self.pro_len)
        maxlayer = nn.MaxPool1d(self.pro_len)
        # print('aggregate_layers--input_: {}'.format(input_.shape))
        avg_ = avglayer(input_.transpose(1, 2)).squeeze()
        max_ = maxlayer(input_.transpose(1, 2)).squeeze()
        out_ = torch.cat((avg_, max_), 1)
        return out_

    def forward(self, X):
        X1, X2 = X.split(1, 1)  # proteinEncoded1: torch.Size([512, 1, 1218])

        proteinEncoded1, adj1 = torch.split(X1, split_size_or_sections=[35, self.c], dim=2)
        proteinEncoded2, adj2 = torch.split(X2, split_size_or_sections=[35, self.c], dim=2)

        proGatedCNN_1 = self.GatedCNN(proteinEncoded1).squeeze(1)  # proGatedCNN_1: torch.Size([512, 32, 32])
        proGatedCNN_2 = self.GatedCNN(proteinEncoded2).squeeze(1)
        conc_pro12 = torch.cat([proteinEncoded1, proteinEncoded2], dim=2).squeeze(1)
        sonsen = conc_pro12.cuda().data.cpu().numpy()
        sonsen = preprocessing.normalize(sonsen)
        sonsen = preprocessing.scale(sonsen)
        sonsen = torch.from_numpy(np.array(sonsen))
        sonsen = sonsen.float()

        # network
        adj_batch = torch.cat([adj1, adj2], axis=2).squeeze(1)  # [256, nodenum]
        b_mat = torch.ones_like(adj_batch)  # [256, 242]
        b_mat[adj_batch != 0] = args.beta

        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0  # [256, 128]
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)  # [256, 1]

        L_1st = torch.sum((embedding_norm -
                           2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                           + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))

        # attention
        pro1_att = self.attention(proGatedCNN_1, proGatedCNN_2, proGatedCNN_1)  # pro1_att: torch.Size([512, 16, 16])
        pro2_att = self.attention(proGatedCNN_2, proGatedCNN_1, proGatedCNN_2)
        pro1_self = torch.mul(proGatedCNN_1.cuda(), pro1_att.cuda())  # pro1_self: torch.Size([512, 16, 16])
        pro2_self = torch.mul(proGatedCNN_2.cuda(), pro2_att.cuda())
        pro1_rep = self.aggregate_layers(pro1_self)  # pro1_rep: torch.Size([512, 16*2])
        pro2_rep = self.aggregate_layers(pro2_self)
        sen_cos_sim = self.cos_sim(pro1_rep, pro2_rep).view(-1, 1).float()  # Cosine similarity torch.Size([512, 1])
        sen_bilinear_sim = self.Bilinear_sim(pro1_rep, pro2_rep)  # Bilinear torch.Size([512, 1])
        sen_linear_sim = torch.tanh(self.linear_sim(torch.cat((pro1_rep, pro2_rep), -1)))  # torch.Size([512, 1])
        merged = torch.cat((
            pro1_rep.cuda(), pro2_rep.cuda(), sonsen.cuda(), sen_cos_sim.cuda(), sen_bilinear_sim.cuda(),
            sen_linear_sim.cuda()), 1)  # merged:torch.Size([128, 64*4+3])
        merged = merged.cuda().data.cpu().numpy()
        merged = preprocessing.normalize(merged)
        merged = preprocessing.scale(merged)
        merged = torch.from_numpy(np.array(merged))
        merged = merged.float()
        output = self.clf_layer(merged.cuda())  # torch.Size([128, 2]
        pred = torch.sigmoid(output).squeeze()
        _, m_label = pred.max(1)

        return pred, m_label, L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd


def getnewList(newlist):
    d = []
    for element in newlist:
        if not isinstance(element, list):
            d.append(element)
        else:
            d.extend(getnewList(element))

    return d


def text_save(filename, data):  # filename is the path to write the CSV file, data is the list of data to be written.
    count = 0
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']',
                                                  '')  # Remove [], the two lines are different according to the data, you can choose
        s = s.replace("'", '').replace(',',
                                       '') + '\n'  # Remove single quotes, commas, and append a newline at the end of each line
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


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default='./data/cora/cora_edgelist.txt',
                        help='Input graph file')
    parser.add_argument('--output', default='./data/cora/Vec.emb',
                        help='Output representation file')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--weighted', action='store_true', default=False,
                        help='Treat graph as weighted')
    parser.add_argument('--epochs', default=100, type=int,
                        help='The training epochs of SDNE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-2, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=100, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--nhid0', default=1000, type=int,
                        help='The first dim')
    parser.add_argument('--nhid1', default=128, type=int,
                        help='The second dim')
    parser.add_argument('--step_size', default=10, type=int,
                        help='The step size for lr')
    parser.add_argument('--gamma', default=0.9, type=int,
                        help='The gamma for lr')
    args = parser.parse_args()
    return args


def train_stacking0323(net, train_data, valid_data, num_epochs, optimizer, criterion, scheduler):
    if torch.cuda.is_available():
        net = net.cuda()
    output_valid_data = []
    epochspred = []
    epochspredlabel = []
    for epoch in range(num_epochs):
        start_onebs = time.time()
        train_loss = 0
        train_acc = 0
        net = net.train()
        for a, a_label in train_data:
            a_label = a_label.long()
            a = torch.tensor(a, dtype=torch.float32)
            if torch.cuda.is_available():
                a = Variable(a.cuda())  # (64, 2, 128, 128)
                a_label = Variable(a_label.cuda())  # (64, 128, 128)
            else:
                a = Variable(a)  # Ideally, the dimension of a is 2*128*128
                a_label = Variable(a_label)
            # forward
            # print(net(a).shape)
            # output, output_label, L_1st, L_2nd, L_all = net(a)
            # output_label = net(a).argmax(dim=1).long()
            # print(output_label, output_label.shape)
            # print(a_label, a_label.shape)
            L_reg = 0
            for param in net.parameters():
                L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
            # print(L_reg)
            # loss = criterion(output, a_label) + L_all + L_reg
            # output_label=torch.unsqueeze(output_label, 0)
            loss = criterion(net(a), a_label) + L_reg
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(epoch)
        if valid_data is not None:
            Acc = 0
            Recall = 0
            Specificity = 0
            Precision = 0
            MCC = 0
            F1 = 0
            AUC = 0
            net = net.eval()
            loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
            onepred = []
            for a, a_label in valid_data:
                a_label = a_label.long()
                a = torch.tensor(a, dtype=torch.float32)
                if torch.cuda.is_available():
                    a = Variable(a.cuda(), volatile=True)  # (64, 2, 128, 128)
                    a_label = Variable(a_label.cuda(), volatile=True)  # (64, 128, 128)
                else:
                    a = Variable(a, volatile=True)
                    a_label = Variable(a_label, volatile=True)
                # output, output_label, L_1st, L_2nd, L_all = net(a)
                output = net(a)
                pred_label_valid_data = output.max(1)[1]
                if epoch == num_epochs - 1:
                    output_valid_data.append(pred_label_valid_data)
                L_reg = 0
                for param in net.parameters():
                    L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
                # loss = criterion(output, a_label) + L_all + L_reg
                loss = criterion(output, a_label) + L_reg
                loss.item()
                y_pred = output.max(1)[1]

                onepred.append(output[:, 1].tolist())
                onepred = getnewList(onepred)

                Acc += metrics.accuracy_score(a_label.cpu(), y_pred.cpu())
                Recall += metrics.recall_score(a_label.cpu(), y_pred.cpu())
                Precision += metrics.precision_score(a_label.cpu(), y_pred.cpu())
                F1 += metrics.f1_score(a_label.cpu(), y_pred.cpu())
                AUC += metrics.roc_auc_score(a_label.cpu(), y_pred.cpu())

                loss_sum += loss
                # loss_L1 += L_1st
                # loss_L2 += L_2nd
                loss_reg += L_reg

            epoch_str = (
                    "Epoch %d. Accuracy: %f, Recall: %f, Precision: %f, F1: %f, AUC: %f"
                    % (epoch, Acc / len(valid_data), Recall / len(valid_data),
                       Precision / len(valid_data), F1 / len(valid_data),
                       AUC / len(valid_data)))
            end_onebs = time.time()
            epoch_time = int(end_onebs - start_onebs)
            # epoch_loss = ('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f} - loss_reg: {4: .4f}'
            #              .format(epoch_time, loss_sum / len(valid_data), loss_L2 / len(valid_data),
            #                     loss_L1 / len(valid_data), loss_reg / len(valid_data)))
            epoch_loss = ('{0}s - loss: {1: .4f} - loss_reg: {2: .4f}'
                          .format(epoch_time, loss_sum / len(valid_data), loss_reg / len(valid_data)))

        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        print(epoch_str)
        print(epoch_loss)
        epochspred.append(onepred)

    output_valid_data = [output_valid_data[i] for i in range(len(output_valid_data))]
    output_valid_data = np.array(list(chain(*output_valid_data)))

    epochspred = np.mat(epochspred).T

    return output_valid_data, epochspred


# 1
if __name__ == '__main__':
    disease = "All_six"
    Ensnum_epoch = 500
    num_epochs = 200
    # print(num_epochs)
    args = parse_args()
    Po_pairs = load(disease + "/Positive_pairs.txt")
    Ne_pairs = load(disease + "/Negative_pairs.txt")
    # Po_pairs = [(Po_pairs[i][0], Po_pairs[i][1], 1) for i in range(len(Po_pairs))]  # If the interaction is marked as 1
    # Ne_pairs = [(Ne_pairs[i][0], Ne_pairs[i][1], 0) for i in range(len(Ne_pairs))]  # No interaction is marked as 0
    # Allpairs = Po_pairs + Ne_pairs  # All data
    # X = [[Allpairs[i][0], Allpairs[i][1], Allpairs[i][2]] for i in range(len(Allpairs))]
    # train_data, test_data = train_test_split(X, test_size=0.2)
    train_data = loadtt("../Datasets/" + disease + "/train.txt")
    test_data = loadtt("../Datasets/" + disease + "/test.txt")

    # train_data = loadtt("../1-3Datasets/" + disease + "/train.txt")
    # test_data = loadtt("../1-3Datasets/" + disease + "/test.txt")

    # train_data = loadtt("../1-5Datasets/" + disease + "/train.txt")
    # test_data = loadtt("../1-5Datasets/" + disease + "/test.txt")
    save_figdata = [[i, test_data[i][2]] for i in range(len(test_data))]

    # 2
    # Keep the proteins involved
    proteinID1 = [Po_pairs[i][0] for i in range(len(Po_pairs))]
    proteinID2 = [Po_pairs[i][1] for i in range(len(Po_pairs))]
    proteinID3 = [Ne_pairs[i][0] for i in range(len(Ne_pairs))]
    proteinID4 = [Ne_pairs[i][1] for i in range(len(Ne_pairs))]
    proteinID = list(set(proteinID1 + proteinID2 + proteinID3 + proteinID4))
    add_nodes = list(set(proteinID3 + proteinID4) - set(proteinID1 + proteinID1))
    # print(add_nodes)
    # Get the sequence of each protein and store it in the dictionary
    proteinSeq = []
    # print(proteinID[0])
    # print(readSeq(proteinID[0]))
    for ID in proteinID:
        Seq = readSeq(ID)
        proteinSeq.append(Seq)
    protein = dict(zip(proteinID, proteinSeq))  # Such as {ID: Seq}

    # Construct adjacency matrix and similarity matrix
    Train_File_g = "../Datasets/" + disease + "/train.txt"
    Test_File_g = "../Datasets/" + disease + "/test.txt"

    # Train_File_g = "../1-3Datasets/" + disease + "/train.txt"
    # Test_File_g = "../1-3Datasets/" + disease + "/test.txt"

    # Train_File_g = "../1-5Datasets/" + disease + "/train.txt"
    # Test_File_g = "../1-5Datasets/" + disease + "/test.txt"
    Adjsi, nodedic, nodelist = Init22(Test_File_g, Train_File_g, proteinID)
    Nodecount = len(nodelist)
    for i in range(len(train_data)):
        temp = [Adjsi[nodelist.index(train_data[i][0])], Adjsi[nodelist.index(train_data[i][1])]]
        train_data[i].append(np.array(temp))
    for i in range(len(test_data)):
        test_data[i].append(np.array([Adjsi[nodelist.index(test_data[i][0])], Adjsi[nodelist.index(test_data[i][1])]]))

    # After obtaining the dictionary form, encode according to the sequence and save it with the dictionary
    proteinAC = CodingAC(protein)  # M4,M41,M42
    proteinLD = CodingLD(protein)  # M5,M51,M52
    proteinCT = CodingCT(protein)  # M6,M61,M62
    proteinPseAAC = CodingPseAAC(protein)  # M7,M71,M72

    proteindict = proteinAC, proteinLD, proteinCT, proteinPseAAC
    # print(proteindict)

    # 3
    """The following is the general framework of the entire program, each learner needs to be filled"""
    """The get_stacking function needs to be changed because the original classifier is different from pytorch"""

    import time

    start = time.clock()

    train_sets = []
    test_sets = []
    models4list = [Model4, Model41, Model42, Model43]
    models5list = [Model5, Model51, Model52, Model53]
    models6list = [Model6, Model61, Model62, Model63]
    models7list = [Model7, Model71, Model72, Model73]
    num_model = 0
    for i in range(len(proteindict)):  # There are four different encoding forms
        if i == 0:
            # print(i)
            for clf in models4list:  # models are represented in the form of a list, and three models are assumed
                print("================AC  num_model = %d  ================" % num_model)
                train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
                train_sets.append(train_set)
                test_sets.append(test_set)
                num_model = num_model + 1
        elif i == 1:
            for clf in models5list:  # models are represented in the form of a list, and three models are assumed
                print("================LD  num_model = %d  ================" % num_model)
                train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
                train_sets.append(train_set)
                test_sets.append(test_set)
                num_model = num_model + 1
        elif i == 2:
            for clf in models6list:  # models are represented in the form of a list, and three models are assumed
                print("================CT  num_model = %d  ================" % num_model)
                train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
                train_sets.append(train_set)
                test_sets.append(test_set)
                num_model = num_model + 1
        else:
            for clf in models7list:  # models are represented in the form of a list, and three models are assumed
                print("================PseAAC  num_model = %d  ================" % num_model)
                train_set, test_set = Indd_stacking2(clf, train_data, test_data, proteindict[i])
                train_sets.append(train_set)
                test_sets.append(test_set)
                num_model = num_model + 1

    # Feature is the result of each model output
    meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

    # Modify data type
    meta_train = torch.tensor(meta_train, dtype=torch.float32)
    meta_test = torch.tensor(meta_test, dtype=torch.float32)

    print(meta_train.shape)  # Ideal (6204, 9)
    print(meta_test.shape)  # Ideal (1552,9)

    meta_train = [(meta_train[i], torch.from_numpy(np.array(train_data[i][2]))) for i in range(len(meta_train))]
    meta_test = [(meta_test[i], torch.from_numpy(np.array(test_data[i][2]))) for i in range(len(meta_test))]

    # Save the training set and test set of the second layer trainer
    metatrain = []
    for i in range(len(meta_train)):
        a = meta_train[i][0].numpy().tolist()
        al = meta_train[i][1].numpy().tolist()
        a.append(al)
        metatrain.append(a)

    metatest = []
    for i in range(len(meta_test)):
        a = meta_test[i][0].numpy().tolist()
        al = meta_test[i][1].numpy().tolist()
        a.append(al)
        metatest.append(a)

    metatrain = np.mat(metatrain)
    metatest = np.mat(metatest)

    np.savetxt("../Datasets/" + disease + "/_meta_train.txt", metatrain, fmt='%0.4f')
    np.savetxt("../Datasets/" + disease + "/_meta_test.txt", metatest, fmt='%0.4f')

    # np.savetxt("../1-3Datasets/" + disease + "/_meta_train.txt", metatrain, fmt='%0.4f')
    # np.savetxt("../1-3Datasets/" + disease + "/_meta_test.txt", metatest, fmt='%0.4f')

    # np.savetxt("../1-5Datasets/" + disease + "/_meta_train.txt", metatrain, fmt='%0.4f')
    # np.savetxt("../1-5Datasets/" + disease + "/_meta_test.txt", metatest, fmt='%0.4f')

    # Iterator
    meta_train = DataLoader(meta_train, batch_size=512, shuffle=True)
    meta_test = DataLoader(meta_test, batch_size=256, shuffle=True)

    # Use Sequential to define a 4-layer neural network
    # Because the input dimension is the number of models, the output classification is binary classification
    in_netClaDNN = 16


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = nn.Linear(in_netClaDNN, 16)
            self.l2 = nn.Linear(16, 2)

        def forward(self, x):
            x = F.relu(self.l1(x))
            x = F.softmax(self.l2(x))
            return x


    netClaDNN = Net()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()  # Define the loss function as cross entropy
    optimizer = torch.optim.SGD(netClaDNN.parameters(),
                                1e-1)  # Optimal use of stochastic gradient descent, learning rate 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Start training
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    print("================Ensemble=================")
    _, epochspred = train_stacking0323(netClaDNN, meta_train, meta_test, Ensnum_epoch, optimizer, criterion, scheduler)

    save_figdata = np.mat(save_figdata)
    save_figdata = np.hstack((save_figdata, epochspred))

    np.savetxt("../Datasets/" + disease + "/Ens_SGAD930_all_six.txt", save_figdata, fmt='%0.6f')
    # np.savetxt("../1-3Datasets/" + disease + "/Ens_SGAD917.txt", save_figdata, fmt='%0.6f')
    # np.savetxt("../1-5Datasets/" + disease + "/Ens_SGAD917.txt", save_figdata, fmt='%0.6f')

    end = time.clock()
    print(str(end - start))
