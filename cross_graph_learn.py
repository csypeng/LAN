
import dgl
from dgl.data import DGLDataset
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv
from dgl.udf import EdgeBatch
from dgl.heterograph import DGLHeteroGraph
import networkx as nx
from networkx.classes.graph import Graph as NXGraph
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List
from wl_labelling import compress_graph
import time

GPUID=0
torch.cuda.set_device(GPUID)



def read_and_split_to_individual_graph(fname):
    
    f = open(fname)
    lines = f.read()
    f.close()

    lines2 = lines.split("t # ")

    lines3 = [g.strip().split("\n") for g in lines2]

    glist = []
    max_node_label = 0
    max_edge_label = 0
    for idx in range(1, len(lines3)):
        cur_g = lines3[idx]
        
        gid_line = cur_g[0].strip().split(' ')
        gid = gid_line[0]
        
        g = nx.Graph(id = gid)

        for idy in range(1, len(cur_g)):
            tmp = cur_g[idy].split(' ')
            if tmp[0] == 'v':
                g.add_node(tmp[1], label=int(tmp[2]))
                max_node_label = max(max_node_label, int(tmp[2]))

            if tmp[0] == 'e':
                g.add_edge(tmp[1], tmp[2])
                max_edge_label = max(max_edge_label, 0)
    
        glist.append(g)
    
    return glist, max_node_label, max_edge_label
 


class GINDataset(DGLDataset):

    def __init__(self, name, gid2gmap, self_loop=False, degree_as_nlabel=False,
                 raw_dir=None, force_reload=False, verbose=False):

        self._name = name 
        gin_url = ""

        self.gid2gmap = gid2gmap    # key: graph id, value: DGLHeteroGraph

        self.g1List = []    # list of DGLHeteroGraph
        self.g2List = []    # list of DGLHeteroGraph
        self.ground_truth = []    # list of ground truth
        

        super(GINDataset, self).__init__(name=name, url=gin_url, hash_key=(name, self_loop, degree_as_nlabel),
                                         raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    @property
    def raw_path(self):
        return os.path.join(".", self.raw_dir)


    def download(self):
        pass


    def __len__(self):
        return len(self.g1List)


    def __getitem__(self, idx):
        return self.g1List[idx], self.g2List[idx], self.ground_truth[idx]
     

    def _file_path(self):
        return self.file


    def process(self):
        
        for k in self.gid2gmap.keys():
            g1 = self.gid2gmap[k]

            # randomly delete an edge from g1
            g2 = nx.Graph(g1)
            rand_edge = random.sample(g1.edges(), 1)
            g2.remove_edge(rand_edge[0][0], rand_edge[0][1])

            self.g1List.append(g1)
            self.g2List.append(g2)
            self.ground_truth.append(1)

            

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass





class myGINConv(nn.Module):
    
    def __init__(self):
        super(myGINConv, self).__init__()

    def forward(self, graph):
        graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
     
        
        self.RELU = torch.nn.ReLU(inplace=True)
  
        # self.conv1_for_g = myGINConv()
        # self.conv2_for_g = myGINConv()
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')
        
        
        self.fc = nn.Linear(64, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True) 
        self.fc3 = nn.Linear(256, 1, bias=True) 
        
        # max_node_label: 60
        self.fc_init_node = nn.Linear(60, 32, bias=True)

        self.fc_att = nn.Linear(64, 1, bias=True)


    def forward(self, g1, g2):

        h0_g1 = self.fc_init_node(g1.ndata['h'])
        h0_g2 = self.fc_init_node(g2.ndata['h'])

        g1.ndata['h'] = h0_g1
        g2.ndata['h'] = h0_g2
       
        # h1_g1 = self.conv1_for_g(g1, 'h')
        # h1_g2 = self.conv1_for_g(g2, 'h')
        g1.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        g2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))

        t1_g1 = g1.ndata['h']
        t1_g2 = g2.ndata['h']

        t1_g1_rep = t1_g1.repeat(1, g2.number_of_nodes()).view(-1, 32)
        t1_g2_rep = t1_g2.repeat(g1.number_of_nodes(), 1)
        t1_g1_cat_t1_g2 = torch.cat([t1_g1_rep, t1_g2_rep], 1)

        alpha = self.fc_att(t1_g1_cat_t1_g2)

        # compute attention weights for g1
        alpha_for_g1 = alpha.view(g1.number_of_nodes(), -1)
        alpha_for_g1_sum = alpha_for_g1.sum(1).view(-1, 1)
        alpha_for_g1 = alpha_for_g1 / alpha_for_g1_sum
        alpha_for_g1 = alpha_for_g1.view(-1,1)
        mu_g1 = alpha_for_g1 * t1_g2_rep
        
        mu_g1 = torch.split(mu_g1, g2.number_of_nodes())
        mu_g1 = torch.stack(mu_g1)
        mu_g1 = mu_g1.sum(1)
        h1_g1 = t1_g1 + mu_g1

        g1.ndata['h'] = h1_g1

        # compute attention weights for g2
        alpha_for_g2 = alpha.view(g1.number_of_nodes(), -1).transpose(0,1)
        alpha_for_g2_sum = alpha_for_g2.sum(1).view(-1,1)
        alpha_for_g2 = alpha_for_g2 / alpha_for_g2_sum
        alpha_for_g2 = alpha_for_g2.contiguous().view(-1, 1)
        
        t1_g1_rep_v2 = t1_g1.repeat(g2.number_of_nodes(), 1)
        mu_g2 = alpha_for_g2 * t1_g1_rep_v2

        mu_g2 = torch.split(mu_g2, g1.number_of_nodes())
        mu_g2 = torch.stack(mu_g2)
        mu_g2 = mu_g2.sum(1)
        h1_g2 = t1_g2 + mu_g2

        g2.ndata['h'] = h1_g2

        g1_emb = dgl.mean_nodes(g1, 'h')
        g2_emb = dgl.mean_nodes(g2, 'h')

        g1_emb_cat_g2_emb = torch.cat([g1_emb, g2_emb], 1)
        

        H = self.fc(g1_emb_cat_g2_emb)
        H2 = self.fc2(H) 
        pred = self.fc3(H2)


        return pred  

   


mseLoss = nn.MSELoss()


def myloss(preds, gts):
    return mseLoss(preds.view(-1,1), gts.view(-1,1).float())
    

def make_a_dglgraph(g: NXGraph) -> DGLHeteroGraph:
    g = g.to_directed()
    dg = dgl.from_networkx(g, node_attrs=["label"])
    return dg


def make_a_dglgraph(g):
    max_deg = 60 # largest node label 
    ones = torch.eye(max_deg)
   
    g = g.to_directed()
    dg = dgl.from_networkx(g, node_attrs=["label"])
    h0 = dg.ndata['label'].view(1,-1).squeeze()
    h0 = ones.index_select(0, h0).float()
    dg.ndata['h'] = h0

    return dg


def collate(samples):
    g1List, g2List, gtList = map(list, zip(*samples))
    dg_g1 = make_a_dglgraph(g1List[0])
    dg_g2 = make_a_dglgraph(g2List[0])
    return dg_g1, dg_g2, torch.tensor(gtList)



#####################################################################################################
### do the job as follows
#####################################################################################################




entire_dataset, max_node_label, max_edge_label = read_and_split_to_individual_graph("aids.txt")

print('entire_dataset len: ', len(entire_dataset))
print("max_node_label", max_node_label)    # max_node_label 59
print("max_edge_label", max_edge_label)    # max_edge_label 3


gid2gmap = {}    # key: graph id, value: DGLHeteroGraph
for g in entire_dataset:
    gid2gmap[g.graph.get('id')] = g


gid2dgmap = {}    # key: graph id, value: DGLHeteroGraph
for g in entire_dataset:
    dg = make_a_dglgraph(g)
    dg = dgl.add_self_loop(dg)
    gid2dgmap[g.graph.get('id')] = dg


train_data = GINDataset("aids", gid2gmap)


dataloader = DataLoader(
    train_data,
    batch_size=1,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)



n_epochs = 100000 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient

# create model
model = Model()
model#.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2norm)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.9)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

with torch.autograd.set_detect_anomaly(True):
    for epoch in range(n_epochs):
        model.train()
        batch_count = 0
        for g1, g2, gt in dataloader:
            print('='*40+" epoch ", epoch, "batch ", batch_count)
            preds = model(g1, g2)
            loss = myloss(preds, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            torch.cuda.empty_cache() 
             
            batch_count += 1   

        # do a test after an epoch
        print("do test ....")
        model.eval()      
        with torch.no_grad():
            for g1, g2, gt in dataloader:
                preds = model(g1, g2)
                torch.cuda.empty_cache() 
                print("+++++++"*5, 'test finish')
                
                 
