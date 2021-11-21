# neighborhood prediction model


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import dgl
from torch.utils.data import DataLoader
import numpy as np
import time
import random
from dgl.nn.pytorch.conv import GINConv, SGConv, SAGEConv
import os
import networkx as nx
from dgl.data import DGLDataset


GPUID=1
torch.cuda.set_device(GPUID)


def read_and_split_to_individual_graph(fname, gsizeNoLessThan=15, gsizeLessThan=30, writeTo=None, prefix=None, fileformat=None, removeEdgeLabel=True, removeNodeLabel=True, graph_num=100000000):
    '''
    aids graphs are in one file. write each graph into a single file
    :parm fname: the file storing all graphs in the format as follows:
        t # 1
        v 1
        v 2
        e 1 2
    :parm gsize: some graphs are too large to compute GED, just skip them at the current stage
    :parm writeTo: path to write
    :parm prefix: give a prefix to each file, e.g., "g" or "q4", etc
    :parm fileformat: None means following the original format of aids; 
    :parm removeEdgeLabel: all edges are given label "0"
    :parm removeNodeLabel: all nodes are given label "0"
    '''
    if writeTo is not None:
        if prefix is None:
            print("You want to write each graph into a single file.")
            print("You need to give the prefix of the filename to store each graph. For example, g, q4, q8")
            exit(-1)
        else:
            if writeTo[-1] == '/':
                writeTo = writeTo+prefix
            else:
                writeTo = writeTo+"/"+prefix
        if fileformat is None:
            print("please specify fileformat: aids, gexf")
            exit(-1)


    f = open(fname)
    lines = f.read()
    f.close()

    lines2 = lines.split("t # ")

    lines3 = [g.strip().split("\n") for g in lines2]

    glist = []
    for idx in range(1, len(lines3)):
        cur_g = lines3[idx]
        
        gid_line = cur_g[0].strip().split(' ')
        gid = gid_line[0]
        if len(gid_line) == 4:
            glabel = gid_line[3]
            g = nx.Graph(id = gid, label = glabel)
        elif len(gid_line) == 6:
            glabel = gid_line[3]
            g = nx.Graph(id = gid, label = glabel)
        else:
            g = nx.Graph(id = gid)

        
        for idy in range(1, len(cur_g)):
            tmp = cur_g[idy].split(' ')
            if tmp[0] == 'v':
                if removeNodeLabel == False:
                    g.add_node(tmp[1], att=tmp[2])
                else:
                    g.add_node(tmp[1], att="0")
            if tmp[0] == 'e':
                if removeEdgeLabel == False:
                    g.add_edge(tmp[1], tmp[2], att=tmp[3])
                else:
                    g.add_edge(tmp[1], tmp[2], att="0")
        

        if g.number_of_nodes() >= gsizeNoLessThan and g.number_of_nodes() < gsizeLessThan:          
            if writeTo is not None:
                if fileformat == "aids": 
                    f2 = open(writeTo+g.graph.get('id')+".txt", "w")              
                    f2.write("t # "+g.graph.get('id')+"\n")

                    if removeNodeLabel:
                        for i in range(0, len(g.nodes())):
                            f2.write("v "+str(i)+" 0\n")
                    else:
                        for i in range(0, len(g.nodes())):
                            f2.write("v "+str(i)+" "+g.nodes[str(i)].get("att")+"\n")

                    if removeEdgeLabel:
                        for e in g.edges():
                            f2.write("e "+e[0]+" "+e[1]+" 0\n")
                    else:
                        for e in g.edges():
                            f2.write("e "+e[0]+" "+e[1]+" "+g[e[0]][e[1]].get("att")+"\n")
                    f2.close()
                if fileformat == "gexf":
                    nx.write_gexf(g, writeTo+g.graph.get("id")+".gexf")
                

            glist.append(g)
            if len(glist) > graph_num:
                return glist
    
    return glist
 





class GINDataset(DGLDataset):

    def __init__(self, name, database, queries, exact_ans, isTrain=True, self_loop=False, degree_as_nlabel=False,
                 raw_dir=None, force_reload=False, verbose=False):

        self._name = name 
        gin_url = ""
        self.database = database
        self.queries = queries
        self.exact_ans = exact_ans
        self.isTrain = isTrain


        self.qList = []
        self.gPosList = []
        self.ground_truth = []

        super(GINDataset, self).__init__(name=name, url=gin_url, hash_key=(name, self_loop, degree_as_nlabel),
                                         raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    @property
    def raw_path(self):
        return os.path.join(".", self.raw_dir)


    def download(self):
        pass


    def __len__(self):
        return len(self.queries)


    def __getitem__(self, idx):
        return self.qList[idx], self.gPosList[idx], self.ground_truth[idx]
     

    def _file_path(self):
        return self.file


    def get_topkAll_in_a_list(self, topk, x):
        kth = x[topk-1]
        res = x[0:topk]
        for i in range(topk, len(x)):
            if x[i][1] == kth[1]:
                res.append(x[i])
        return res


    def process(self):
        for q in self.queries:
            self.qList.append(q)

            exact_ans_of_q = self.get_topkAll_in_a_list(200, self.exact_ans[q])
            exact_ans_of_q_IDSet = set()
            
            gt_label = []
            gPos = []
            for cur_ans in exact_ans_of_q:
                exact_ans_of_q_IDSet.add(cur_ans[0])

            for idx in range(0, len(self.database)):
                ele = self.database[idx]
                if ele.graph.get('id') in exact_ans_of_q_IDSet:
                    gt_label.append(1.0)
                    gPos.append(idx)
                else:
                    if self.isTrain:
                        rand = np.random.randint(10)
                        if rand > 8:
                            gt_label.append(0.0)
                            gPos.append(idx)
                    else:
                        gt_label.append(0.0)
                        gPos.append(idx)


            if len(gt_label) != len(gPos):
                print("len(gt_label) != len(gPos)")
                exit(-1)

            self.ground_truth.append(gt_label)
            self.gPosList.append(gPos)

            


    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass







class Model(nn.Module):
    def __init__(self, gInitEmbMap, gid2dgmap, allDBGEmb):
        super(Model, self).__init__()

        self.him = 1024
        self.allDBGEmb = allDBGEmb

        # use gin
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')

        self.gnn_bn = torch.nn.BatchNorm1d(self.him)
        self.gnn_bn2 = torch.nn.BatchNorm1d(self.him)

        self.gInitEmbMap = gInitEmbMap
        self.gid2dgMap = gid2dgmap


        self.fc = nn.Linear(self.him*2, 128, bias=True)
        self.fc2 = nn.Linear(128, 1, bias=True)
   
        self.bn = torch.nn.BatchNorm1d(128)
    
        self.RELU = torch.nn.ReLU(inplace=True)

        self.fc_init = nn.Linear(20, self.him, bias=True)
        

    def forward(self, qids, gPosList):
        preds = []
        

        for idx in range(0, len(qids)):
            qid = qids[idx]
            gPos = gPosList[idx]
            
            dg_of_q = self.gid2dgMap[qid]
            
            dg_of_q.ndata['h2'] = self.fc_init(dg_of_q.ndata['h'])       
            dg_of_q.ndata['h2'] = self.RELU(self.gnn_bn(self.conv1_for_g(dg_of_q, dg_of_q.ndata['h2'])))
            dg_of_q.ndata['h2'] = self.RELU(self.gnn_bn2(self.conv2_for_g(dg_of_q, dg_of_q.ndata['h2'])))

            qemb = dgl.mean_nodes(dg_of_q, 'h2').squeeze()
            gEmbList = self.allDBGEmb.index_select(0, torch.tensor(gPos).cuda())
            qemb = qemb.repeat(gEmbList.shape[0]).view(-1, self.him)
            
            H = torch.cat([qemb, gEmbList], 1)
            H2 = self.RELU(self.bn(self.fc(H)))
            probs = torch.sigmoid(self.fc2(H2)).view(1,-1).squeeze()
            preds.append(probs)
        
         
        return preds


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, min=1e-6, max=1-1e-6)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))




bceLoss = nn.BCELoss()



def myloss(preds, gts):
    
    loss = []
    for i in range(0, len(preds)):
        pred = preds[i]
        gt = torch.tensor(gts[i]).cuda()
        cur_loss= weighted_binary_cross_entropy(pred, gt, [1.0, 10.0])
        loss.append(cur_loss)
    loss = torch.stack(loss)

    return torch.mean(loss)


from sklearn.metrics import roc_auc_score 

def my_loss_for_test(preds, gts):
    avg_auc = 0
    for i in range(0, len(preds)):
        pred = preds[i]
        gt = gts[i]
        avg_auc += roc_auc_score(gt, pred.cpu().detach().numpy())
    avg_auc /= len(preds)

    return avg_auc


def check_recall(preds, gts):
    avg_precision = 0
    for i in range(0, len(preds)):
        pred = preds[i].cpu().detach().numpy().tolist()
        gt = gts[i]
        abc = []
        for idx in range(0, len(pred)):
            abc.append( (gt[idx], pred[idx]) )
        abc.sort(key = lambda x: -x[1])
        
        precision = 0
        top_perc10 = 200 
        for idx in range(0, top_perc10):
            if abc[idx][0] == 1:
                precision += 1
        precision = precision / top_perc10
        
        avg_precision += precision
    
    avg_precision = avg_precision / len(preds)

    print('avg precision', avg_precision)

    return avg_precision


def read_initial_gemb(addr):
    gEmbMap = {}
    gfileList = os.listdir(addr)
    for gfile in gfileList:
        gID = gfile[1:-4]
        f = open(addr+"/"+gfile)
        lines = f.read()
        f.close()
        lines = lines.strip().split('\n')
        lines = lines[1:]
        nodeEmbList = []
        for line in lines:
            tmp = line.strip().split(' ')
            tmp2 = [float(ele) for ele in tmp[1:]]
            nodeEmbList.append(tmp2)
        nodeEmbList = torch.tensor(nodeEmbList)
        gEmb = torch.mean(nodeEmbList, 0)
        gEmbMap[gID] = gEmb
    return gEmbMap



def make_a_dglgraph(g):
    
    max_deg = 20 # largest node degree of q
    ones = torch.eye(max_deg)
   
    edges = [[],[]]
    for edge in g.edges():
        end1 = edge[0]
        end2 = edge[1]
        edges[0].append(int(end1))
        edges[1].append(int(end2))
        edges[0].append(int(end2))
        edges[1].append(int(end1))
    dg = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])))
    
    h0 = dg.in_degrees().view(1,-1).squeeze()
    h0 = ones.index_select(0, h0).float()
    dg.ndata['h'] = h0

    return dg.to(torch.device('cuda:'+str(GPUID)))




def readQ2GDistBook(fname, validNodeIDSet=None):
    '''
    store distance from the query to a data graph
    '''
    f = open(fname)
    lines = f.read()
    f.close()
    lines = lines.strip()
    lines = lines.split('\n')
    distBook = {}
    for line in lines:
        tmp = line.split(" ")
        if validNodeIDSet != None and tmp[1] in validNodeIDSet:
            if tmp[0] in distBook:
                distBook[tmp[0]][tmp[1]] = float(tmp[2])
            else:
                distBook[tmp[0]] = {}
                distBook[tmp[0]][tmp[1]] = float(tmp[2])
    return distBook




def get_exact_answer(topk, Q2GDistBook):
    answer = {}
    for query in Q2GDistBook.keys():
        distToGList = list(Q2GDistBook[query].items())
        distToGList.sort(key=lambda x: x[1])
        dist_thr = -1
        if topk-1 < len(distToGList):
            dist_thr = distToGList[topk-1][1]
        else:
            dist_thr = 1000000.0
        a = []
        for ele in distToGList:
            if ele[1] <= dist_thr:
                a.append(ele)
            else:
                break
        answer[query] = a
    return answer




def collate(samples):
    qids, gPosList, gtlabels = map(list, zip(*samples))
 
    return qids, gPosList, gtlabels


#####################################################################################################
### do the job as follows
#####################################################################################################

# read in proximity graph
pgTmp = read_and_split_to_individual_graph("PG.aids.nx", 0, 10000000000)
pgTmp = pgTmp[0]


gInitEmbMap = read_initial_gemb('data/AIDS/emb/aids.emb')
print('read g init emb done.')

entire_dataset = read_and_split_to_individual_graph("aids.txt", 0, 10000000)
print('entire_dataset len: ', len(entire_dataset))
gid2gmap = {}
for g in entire_dataset:
    gid2gmap[g.graph.get("id")] = g
gid2dgmap = {}
for g in entire_dataset:
    dg = make_a_dglgraph(g)
    dg = dgl.add_self_loop(dg)
    gid2dgmap[g.graph.get('id')] = dg.to(torch.device('cuda:'+str(GPUID)))



database = entire_dataset[0:40000]
database_ids = set([ele.graph.get('id') for ele in database])


databaseGEmb = []
for g in database:
    gEmb = gInitEmbMap[g.graph.get('id')]
    databaseGEmb.append(gEmb)
databaseGEmb = torch.stack(databaseGEmb).cuda()

train_queries_ids = []
f = open('data/AIDS/query_train.txt')
lines = f.read()
f.close()
lines = lines.strip().split('\n')
for line in lines:
    qid = line.strip()
    train_queries_ids.append(qid)


test_queries_ids = []
f = open('data/AIDS/query_test.txt')
lines = f.read()
f.close()
lines = lines.strip().split('\n')
for line in lines:
    qid = line.strip()
    test_queries_ids.append(qid)




q2GDistBook = readQ2GDistBook("data/AIDS/aids.txt", database_ids)
print('readQ2GDistBook done')
exact_ans = get_exact_answer(100000000, q2GDistBook)


train_data = GINDataset("aids", database, train_queries_ids, exact_ans, isTrain=True)
test_data = GINDataset("aids", database, test_queries_ids, exact_ans, isTrain=False)




dataloader = DataLoader(
    train_data,
    batch_size=200,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)

testdataloader = DataLoader(
    test_data,
    batch_size=20,
    collate_fn=collate,
    drop_last=False,
    shuffle=True)


n_epochs = 8000 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient


model = Model(gInitEmbMap, gid2dgmap, databaseGEmb)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2norm)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.9)




with torch.autograd.set_detect_anomaly(True):   
    for epoch in range(n_epochs):
        model.train()
        start_train_time = time.time()
        batch_count = 0
        for qids, gPosList, gts in dataloader:
            print('='*40+" epoch ", epoch, "batch ", batch_count)
            preds = model(qids, gPosList)
            loss = myloss(preds, gts)
            print('loss ', loss)
            auc = my_loss_for_test(preds, gts)
            print('auc ', auc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            batch_count += 1     


        # do a test after an epoch
        print("do test ....")
        model.eval()      
        with torch.no_grad():
            # do valid
            # print("start valid ................")
            # for valid_graph, valid_label, graphIDs, nodeMaps in validdataloader:
            #     valid_pred = model(valid_graph, valid_label, graphIDs, nodeMaps)
            #     tmp = my_loss_for_test(valid_pred, valid_label)
            #     valid_loss = tmp[0]
            #     print('valid loss ', valid_loss)
            #     avg_mse_of_valid = avg_mse_of_valid + valid_loss.item()
            #     valid_count = valid_count + 1
            #     print("+++++++"*5, "valid finish")


            # do test
            print("start test ................")
            print("epoch", epoch)
            for qids, gPosList, gts in testdataloader:
                preds = model(qids, gPosList)
                auc = my_loss_for_test(preds, gts)
                check_recall(preds, gts)
                print('auc ', auc)
            
                print("+++++++"*5, 'test finish')



