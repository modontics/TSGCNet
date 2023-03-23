import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def knn(x, k):
    # use float16 to save GPU memory, HalfTensor is enough here; this reduced lots mem usage, so we can use 10GB GPU
    # (batch_size, dim, num_points)
    x = x.half()

    # bmm does not help on GPU memory here: inner = -2 * torch.bmm(x.transpose(2, 1), x)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    # use two steps for pairwise_distance, to save GPU memory; this reduced lots mem usage, so we can use 10GB GPU
    pairwise_distance = -xx - inner
    xx = xx.transpose(2, 1)
    pairwise_distance -= xx
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]  # (batch_size, num_points, k)
    return idx


# TODO: we do not really exact K and exact distance, so we can look for approximation of knn solutions (eg Faiss library developed by Facebook)
# adapted the following code from GPT-4, but it is >3x slower than matmul/pairwise_distance, and correctness not verified
def knn_balltree(x: torch.Tensor, k):
    from sklearn.neighbors import BallTree

    x = x.transpose(2, 1)
    batch_size, num_points, dim = x.shape

    # Flatten points to a 2D tensor for ball tree calculation
    x_flat = x.view(batch_size * num_points, dim)

    with torch.no_grad():
        # Create ball tree
        tree = BallTree(x_flat.cpu().numpy(), leaf_size=k)

        # Query ball tree
        dist, idx = tree.query(x_flat.cpu().numpy(), k=k+1)

    # Convert indices to PyTorch tensor
    idx = torch.from_numpy(idx[:, 1:]).to(x.device)

    # Reshape indices to match original tensor shape
    idx = idx.view(batch_size, num_points, k)

    return idx


def knn_faiss_gpu():
    """an example of using Faiss GPU for k-nearest neighbors (knn) search. NOT TESTED!
    In this example, we generate some random data and a query tensor. We then initialize a Faiss index and train it on the data. Finally, we perform knn search on the query tensor using the trained Faiss GPU index, and print the resulting indices. Note that we convert the PyTorch tensors to numpy arrays before passing them to Faiss, and also convert the resulting indices back to PyTorch tensors if needed.

    https://github.com/facebookresearch/faiss
    https://anaconda.org/pytorch/faiss-gpu
    """
    import faiss

    # Generate some random data
    batch_size = 32
    num_points = 1000
    embedding_size = 128
    data = torch.randn(batch_size, num_points, embedding_size)

    # Initialize Faiss GPU index
    d = embedding_size
    index = faiss.IndexFlatL2(d)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

    # Train the index
    gpu_index.add(data.cpu().numpy())

    # Perform knn search
    k = 10
    query = torch.randn(batch_size, k, embedding_size)
    gpu_query = query.cpu().numpy()
    _, knn_indices = gpu_index.search(gpu_query, k)

    print(knn_indices)


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def get_graph_feature(coor, nor, k=10):
    batch_size, num_dims, num_points  = coor.shape
    coor = coor.view(batch_size, -1, num_points)

    idx = knn(coor, k=k)
    index = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()

    coor = coor.transpose(2,1).contiguous()
    nor = nor.transpose(2,1).contiguous()

    # coordinate
    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)
    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()
    return coor_feature, nor_feature, index



class GraphAttention(nn.Module):
    def __init__(self,feature_dim,out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.K=K

    def forward(self, Graph_index, x, feature):

        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0,2,3,1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre-neighbor_feature, neighbor_feature], dim=3).permute(0,3,2,1)
        e = self.conv(delta_f)
        e = e.permute(0,3,2,1)
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature),dim = 2) .permute(0,2,1)
        return graph_feature


class TSGCNet(nn.Module):
    def __init__(self, k=16, in_channels=12, output_channels=2):
        super(TSGCNet, self).__init__()
        self.k = k
        ''' coordinate stream '''
        # NOTE: we reduced to half size of original network;
        # we already use 4 points and 4 normals per face, that is enough input data, maybe it is too redundant, we could reduce data and increase network size
        out1 = 32 # in_channels*2 * 2
        out2 = out1 * 2
        out3 = out2 * 2
        out4 = out3 * 2
        self.bn1_c = nn.BatchNorm2d(out1)
        self.bn2_c = nn.BatchNorm2d(out2)
        self.bn3_c = nn.BatchNorm2d(out3)
        self.bn4_c = nn.BatchNorm1d(out4)
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels*2, out1, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2_c = nn.Sequential(nn.Conv2d(out1*2, out2, kernel_size=1, bias=False),
                                   self.bn2_c,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3_c = nn.Sequential(nn.Conv2d(out2*2, out3, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4_c = nn.Sequential(nn.Conv1d(out1+out2+out3, out4, kernel_size=1, bias=False),
                                     self.bn4_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        # SY 12 -> 3
        self.attention_layer1_c = GraphAttention(feature_dim=12, out_dim=out1, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=out1, out_dim=out2, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=out2, out_dim=out3, K=self.k)
        # SY 12 -> 3
        self.FTM_c1 = STNkd(k=12)
        ''' normal stream '''
        self.bn1_n = nn.BatchNorm2d(out1)
        self.bn2_n = nn.BatchNorm2d(out2)
        self.bn3_n = nn.BatchNorm2d(out3)
        self.bn4_n = nn.BatchNorm1d(out4)
        self.conv1_n = nn.Sequential(nn.Conv2d((in_channels)*2, out1, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv2_n = nn.Sequential(nn.Conv2d(out1*2, out2, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv3_n = nn.Sequential(nn.Conv2d(out2*2, out3, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv4_n = nn.Sequential(nn.Conv1d(out1+out2+out3, out4, kernel_size=1, bias=False),
                                     self.bn4_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        # SY 12 -> 3
        self.FTM_n1 = STNkd(k=12)

        '''feature-wise attention'''

        self.fa = nn.Sequential(nn.Conv1d(out4*2, out4*2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(out4*2),
                                nn.LeakyReLU(0.2))

        ''' feature fusion '''
        self.pred1 = nn.Sequential(nn.Conv1d(out4*2, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))

        # nn.Dropout(p): p, probability of an element to be zeroed
        # NOTE: original p=0.6, it may converge too slow, we change to smaller prob (as we dropout 3 times) so it will converge faster
        p = 0.03
        self.dp1 = nn.Dropout(p)
        self.dp2 = nn.Dropout(p)
        self.dp3 = nn.Dropout(p)

    def forward(self, x):
        # SY 12 -> 3
        coor = x[:, :12, :]
        # SY 12 -> 3
        nor = x[:, 12:, :]

        # transform
        trans_c = self.FTM_c1(coor)
        coor = coor.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)
        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)

        coor1, nor1, index = get_graph_feature(coor, nor, k=self.k)
        coor1 = self.conv1_c(coor1)
        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]
        # del and release intermediate variables, so we can relieve GPU mem fragment problem
        del coor, nor
        # NOTE: calling torch.cuda.empty_cache() too often will slow down the training process
        # torch.cuda.empty_cache()

        coor2, nor2, index = get_graph_feature(coor1, nor1, k=self.k)
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]

        coor3, nor3, index = get_graph_feature(coor2, nor2, k=self.k)
        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]

        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor = self.conv4_c(coor)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor = self.conv4_n(nor)
        del coor1, coor2, coor3, nor1, nor2, nor3

        # TODO: do we really need average? we already normalized coordinates and normals in dataloader; and, "512" seems very arbitrary!
        # avgSum_coor = coor.sum(1)/512
        # avgSum_nor = nor.sum(1)/512
        # avgSum = avgSum_coor+avgSum_nor
        # coor *= (avgSum_coor / avgSum).reshape(1, 1, self.num_faces)
        # nor *= (avgSum_nor / avgSum).reshape(1, 1, self.num_faces)
        x = torch.cat((coor, nor), dim=1)
        weight = self.fa(x)
        x = weight * x
        del coor, nor, weight

        x = self.pred1(x)
        # TODO: original TSGCNet code did not reassign x after dropout, and Dropout Module use default inplace=False, so effectively it did not perform dropout!
        # see https://jamesmccaffrey.wordpress.com/2019/01/23/pytorch-train-vs-eval-mode/, we need to use model.train() model.eval() correctly!
        self.dp1(x)
        x = self.pred2(x)
        self.dp2(x)
        x = self.pred3(x)
        self.dp3(x)
        score = self.pred4(x)
        score = F.log_softmax(score, dim=1)
        score = score.permute(0, 2, 1)

        # print(f"forward: id {torch.cuda.current_device()} allocated {torch.cuda.memory_allocated()//(1024**2)} reserved {torch.cuda.memory_reserved()//(1024**2)}")
        return score


if __name__ == "__main__":
    num_faces = 16000
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input size: [batch_size, C, N], where C is number of dimension, N is the number of mesh faces.
    x = torch.rand(1,24,num_faces)
    x = x.cuda()
    model = TSGCNet(in_channels=12, output_channels=17, k=32)
    model = model.cuda()
    y = model(x)
    print(y.shape)
