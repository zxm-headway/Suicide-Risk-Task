from torchfm.layer import CrossNetwork, FeaturesEmbedding, MultiLayerPerceptron
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np


# 选择的网络层
class SelectionNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SelectionNetwork, self).__init__()
        self.mlp =  MultiLayerPerceptron(input_dim=input_dims,embed_dims=[output_dims], output_layer=False, dropout=0.0)
        self.weight_init(self.mlp)
                                        
    def forward(self, input_mlp):
        output_layer = self.mlp(input_mlp)
        return torch.softmax(output_layer, dim=1)
      

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)



class MvFS_Controller(nn.Module):
    # embed_dims是字段的长度，也就是有多少字段，每个元素表示对应字段的重要性
    def __init__(self, input_dim, embed_dims, num_selections):
        super().__init__()
        self.inputdim = input_dim
        self.num_selections = num_selections
        self.T = 1
        # 决定每个子网络的重要性
        self.gate = nn.Sequential(nn.Linear(embed_dims * num_selections , num_selections))
        
        # 子网络
        self.SelectionNetworks = nn.ModuleList(
            [SelectionNetwork(input_dim, embed_dims) for i in range(num_selections)]
        )

    def forward(self, input_mlp):

        # input_mlp = emb_fields.flatten(start_dim=1).float()

        importance_list= []
        for i in range(self.num_selections):
            importance_vector = self.SelectionNetworks[i](input_mlp)
            importance_list.append(importance_vector)

        
        gate_input = torch.cat(importance_list, 1)
        selection_influence = self.gate(gate_input)
        selection_influence = torch.sigmoid(selection_influence) # 0-1之间
        
        scores = None
        for i in range(self.num_selections):
            score = torch.mul(importance_list[i], selection_influence[:,i].unsqueeze(1))
            if i == 0 :
                scores = score
            else:
                scores = torch.add(scores, score)
                        
        scores = 0.5 * (1+ torch.tanh(self.T*(scores-0.1)))
        
        if self.T < 5:
            self.T += 0.001
        return scores


class MultiLayerPerceptron(nn.Module):
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
        self.mlp = MultiLayerPerceptron(input_dim=self.embed_dim,embed_dims=[self.embed_dim//2,5], output_layer=False, dropout=0.2)
        # self.linear = torch.nn.Linear(16 + self.embed_dim, 5)
        self.BN = nn.BatchNorm1d(self.embed_dim)

    def forward(self,input):
        # field = self.BN(input)
        # field = self.cn(input)
        res = self.mlp(input)

        # res = torch.cat([field, res], dim=1)
        # res = self.linear(res)
        return res

class MvFS_DCN(nn.Module): 
    def __init__(self,inputs, num_selections):
        super().__init__()

        self.dcn = MLP(input_dim=inputs)
        
        self.controller = MvFS_Controller(input_dim=inputs, embed_dims=5, num_selections= num_selections)
  
        self.weight = 0
        self.stage = -1
       

    def forward(self, field,):
        # field = self.emb(field)
        if self.stage == 1: # use controller
 
            self.weight = self.controller(field)
            selected_field = field * torch.unsqueeze(self.weight,1)
    
            input_mlp = selected_field
        else: # only backbone 
            input_mlp = field

        input_mlp = input_mlp.flatten(start_dim=1).float()
        
        res = self.dcn(input_mlp)

        return torch.sigmoid(res.squeeze(1))