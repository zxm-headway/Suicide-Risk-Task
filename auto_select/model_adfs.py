from torchfm.layer import CrossNetwork, FeaturesEmbedding, MultiLayerPerceptron
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np




class EarlyStopping(object):

    def __init__(self, num_trials,path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = path

    def is_continuable(self, model, accuracy):


        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save({'state_dict': model.state_dict()},self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
        


class MultiLayerPerceptron1(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        self.mlps = nn.ModuleList()
        self.out_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            # layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
            self.mlps.append(nn.Sequential(*layers))
            layers = list()
        if self.out_layer:
            self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        for layer in self.mlps:
            x = layer(x)
        if self.out_layer:
            x = self.out(x)
        return x


# 预测模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embed_dim = input_dim
        # self.cn = CrossNetwork(input_dim=self.embed_dim,num_layers=2)
        self.mlp = MultiLayerPerceptron1(input_dim=self.embed_dim,embed_dims=[self.embed_dim//2,5], output_layer=False, dropout=0.2)
        # self.linear = torch.nn.Linear(self.embed_dim*2, 5)
        self.BN = nn.BatchNorm1d(self.embed_dim)
        self.weight_init(self.mlp)

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self,input):
        # field = self.BN(input)
        # field = self.cn(input)
        
        
        res = self.mlp(input)
        # res = torch.cat([field, res], dim=1)
        # res = self.linear(res)


 
        # res = self.linear(res)
        return res


class controller_mlp(nn.Module):
    def __init__(self,input_dim, nums):
        super().__init__()
        self.inputdim = input_dim
        self.mlp = MultiLayerPerceptron1(input_dim=self.inputdim, embed_dims=[64,32,nums], output_layer=False, dropout=0.2)
        # self.weight_init(self.mlp)
    
    def forward(self, emb_fields):
        # input_mlp = emb_fields.flatten(start_dim=1).float()
        output_layer = self.mlp(emb_fields)
        return torch.softmax(output_layer, dim=1)

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)



# 适应性特征选择方法
class AdaFS_soft(nn.Module): 
    def __init__(self,input_dims,inputs_dim,num):
        super().__init__()
        self.num = num
        self.embed_dim = input_dims
        # self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
        self.mlp = MultiLayerPerceptron1(input_dim=self.embed_dim, embed_dims=[self.embed_dim//2,5] , output_layer=False, dropout=0.2)
        self.controller = controller_mlp(input_dim=self.embed_dim, nums=self.num)
        self.weight = 0
        self.useBN = False
        self.inputs_dim = inputs_dim
        self.UseController = True
        # self.BN = nn.BatchNorm1d(self.embed_dim)
        self.stage = 1


    def forward(self, field):
        if self.useBN == True:
            field = self.BN(field)
        if self.UseController and self.stage == 1:
            self.weight = self.controller(field)
            self.dims = []
            for k,v in self.inputs_dim.items():
                self.dims.append(v)
            offsets = np.array((0, *np.cumsum(self.dims)[:-1]), dtype=np.int64)
            field1 = field.clone()
            for i in range(len(offsets)-1):
                field1[:, offsets[i]:offsets[i+1]] = field[:, offsets[i]:offsets[i+1]] * self.weight[:,i].unsqueeze(1)
        res = self.mlp(field1)
        return res


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0] 
    return index, 
# x.gather(dim, index)

class AdaFS_hard(nn.Module): 
    def __init__(self,input_dims,inputs_dim):
        super().__init__()
        self.num = 5
        self.embed_dim = input_dims
        # self.emb = EMB(args.field_dims[:self.num],self.embed_dim)
        self.mlp = MultiLayerPerceptron1(input_dim=self.embed_dim,
                                        embed_dims=[self.embed_dim//2,5], output_layer=True, dropout=0.2)
        self.controller = controller_mlp(input_dim=self.embed_dim, nums=self.num)
        self.UseController = True
        self.BN = nn.BatchNorm1d(self.embed_dim)
        self.k = 3
        self.useWeight = True
        self.reWeight = True
        self.inputs_dim = inputs_dim
        self.useBN = False
        self.stage = -1

    def forward(self, field):
        # field = self.emb(field)
        #对每个feature进行batchnorm
        if self.useBN == True:
            field = self.BN(field)
        if self.UseController and self.stage == 1:
            weight = self.controller(field)
            kmax_index, kmax_weight = kmax_pooling(weight,1,self.k)
            if self.reWeight == True:
                kmax_weight = kmax_weight/torch.sum(kmax_weight,dim=1).unsqueeze(1) #reweight, 使结果和为1
            #创建跟weight同维度的mask，index位赋予值，其余为0
            mask = torch.zeros(weight.shape[0],weight.shape[1]).to(self.device)
            if self.useWeight:
                mask = mask.scatter_(1,kmax_index,kmax_weight) #填充对应索引位置为weight值
            else:
                mask = mask.scatter_(1,kmax_index,torch.ones(kmax_weight.shape[0],kmax_weight.shape[1])) #对应索引位置填充1

            field = field * torch.unsqueeze(mask,1)      
        input_mlp = field.flatten(start_dim=1).float()
        res = self.mlp(input_mlp)
        return res


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = np.array(data)  # 将数据转换为 NumPy 数组
        self.labels = np.array(labels)  # 将标签转换为 NumPy 数组
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)

# 基于mvfs的特征选择方法，混合专家模型

class MVFS(nn.Module):
    pass