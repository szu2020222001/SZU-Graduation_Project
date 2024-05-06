import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from utils import normalize_A, generate_cheby_adj


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        self.dp = nn.Dropout(dropout)
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x, L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class Attentionadj(nn.Module):
    def __init__(self, in_size, hidden_size=21):
        super(Attentionadj, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class SFGCN1(nn.Module):
    def __init__(self, xdim, kadj, num_out, nclass, dropout):
        super(SFGCN1, self).__init__()
        self.SGCN1 = Chebynet(xdim, kadj, num_out, dropout)  # SGC(k=kadj,nfeat=5,nclass=num_out)#GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = Chebynet(xdim, kadj, num_out, dropout)  # SGC(k=kadj,nfeat=5,nclass=num_out)# GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = Chebynet(xdim, kadj, num_out, dropout)  # SGC(k=kadj,nfeat=5,nclass=num_out)#GCN(nfeat, nhid1, nhid2, dropout)
        self.BN1 = nn.BatchNorm1d(xdim[2])
        self.attention = Attention(num_out)
        self.attentionadj = Attentionadj(21)
        self.bn = nn.BatchNorm1d(21)
        self.mp = nn.AvgPool2d(2)
        self.MLP = nn.Sequential(
            nn.Linear(80, 8),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.MLP_special = nn.Sequential(  # new
            nn.Linear(80, 8),
            # nn.BatchNorm1d(8),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.MLP_LAST = nn.Linear(8, nclass)
        self.A = nn.Parameter(torch.FloatTensor(21, 21).cuda())
        self.A = nn.init.kaiming_normal_(self.A,)

    def forward(self, x, fadj):
        fadj = normalize_A(fadj, symmetry=False, gaowei=True)       # [B,C,C]
        sadj = normalize_A(self.A, symmetry=False, gaowei=False)    # [C,C]
        emb1 = self.SGCN1(x, sadj)  # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj)  # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        # output = (emb1 + emb2 + Xcom) / 3
        output, att = self.attention(emb)
        # emb = None
        output = F.relu(self.bn(output))
        output = self.mp(output)    # [B, C, num_out]->[B,C//2=10,num_out//2=8]
        feat = output.reshape(output.shape[0], -1)  # [B,C//2=10,num_out//2=8]->[B,80]
        if output.shape[0]==1:
            output1 = self.MLP_special(feat)
        else:
            output1 = self.MLP(feat)
        output = self.MLP_LAST(output1)
        # return output, att, emb1, com1, com2, emb2, emb, output1
        return feat, output

