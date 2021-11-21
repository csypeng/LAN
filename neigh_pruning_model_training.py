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
import jpype




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

    def __init__(self, name, trainFileName, gid2dgmap, gID2InitEmbMap, gID2InitTensorIndexMap, neighNum, margin, isTrain, self_loop=False, degree_as_nlabel=False,
                 raw_dir=None, force_reload=False, verbose=False):

        self._name = name 
        gin_url = ""

        self.gid2dgmap = gid2dgmap
        self.gID2InitEmbMap = gID2InitEmbMap
        self.gID2InitTensorIndexMap = gID2InitTensorIndexMap
        self.isTrain = isTrain
        self.neighNum = neighNum
        self.margin = margin
        self.trainFileName = trainFileName

        self.qList = []
        self.pgNodeEmbList = []
        self.ground_truth = []
        self.neighInitEmbIndexList = []
        self.mask_of_1_list = []
        self.mask_of_0_list = []
        self.class_weight_list = []

        super(GINDataset, self).__init__(name=name, url=gin_url, hash_key=(name, self_loop, degree_as_nlabel),
                                         raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    @property
    def raw_path(self):
        return os.path.join(".", self.raw_dir)


    def download(self):
        pass


    def __len__(self):
        return len(self.qList)


    def __getitem__(self, idx):
        return self.qList[idx], self.pgNodeEmbList[idx], self.neighInitEmbIndexList[idx], self.ground_truth[idx], self.mask_of_1_list[idx], self.mask_of_0_list[idx], self.class_weight_list[idx]
     

    def _file_path(self):
        return self.file


    def process(self):
        if self.isTrain:
            f = open(self.trainFileName) # do not forget to set this address 
        else:
            f = open('test.data') # do not forget to set this address 
        
        lines = f.read()
        f.close()
        lines = lines.strip().split('\n')
        
        for line in lines:
            tmp = line.strip().split(' ')
           
            self.qList.append(self.gid2dgmap[tmp[0]])
            self.pgNodeEmbList.append(self.gID2InitEmbMap[tmp[1]])
            q_g_ged = float(tmp[2])
            gt = [int(ele) for ele in tmp[3:3+self.neighNum]]
            print(sum(gt)/self.neighNum)
            gt = gt[st:ed]
            print('st', st, 'ed', ed)
            self.ground_truth.append(gt)
            mask_of_1 = []
            mask_of_0 = []
            for i in range(0, len(gt)):
                if gt[i] == 1:
                    mask_of_1.append(1)
                    mask_of_0.append(0)
                else:
                    mask_of_0.append(1)
                    mask_of_1.append(0)

            mask_of_1 = torch.tensor(mask_of_1).bool()
            mask_of_0 = torch.tensor(mask_of_0).bool()
     
            self.mask_of_1_list.append(mask_of_1)
            self.mask_of_0_list.append(mask_of_0)

            neighIDs_and_neighDists = tmp[3+self.neighNum:]
            neighIDs = neighIDs_and_neighDists[0 : int(len(neighIDs_and_neighDists)/2)]
            neighDists = neighIDs_and_neighDists[int(len(neighIDs_and_neighDists)/2) : ]
            if len(neighIDs) != len(neighDists):
                print('ERROR! len(neighIDs) != len(neighDists)')
                print("neighIDs", len(neighIDs))
                print("neighDists", len(neighDists))
                exit(-1)


            neighIDs = neighIDs[st:ed]
            posInInitTensor = []
            for neighID in neighIDs:
                posInInitTensor.append(self.gID2InitTensorIndexMap[neighID])
            while len(posInInitTensor) < (ed-st):
                # pad zero 
                posInInitTensor.append(40000) # aids dataset size is 40000, do not forget to re-set it for different dataset
            
            class_weight = []
            neighDists = neighDists[st:ed]
            # print(neighDists)
            # print(q_g_ged)
            for idx in range(0, len(neighDists)):
                neighDist = float(neighDists[idx])
                dist_diff = neighDist - q_g_ged - self.margin # the self.margin here needs to be consistent with the margin in train.data generation
                class_weight.append(abs(dist_diff))
            while len(class_weight) < (ed-st):
                class_weight.append(0.0)
            class_weight = np.array(class_weight)
            # print(class_weight)
            # class_weight = class_weight/(np.max(class_weight)+0.00000001)
            class_weight = class_weight/(100.0)
            # print(class_weight)
            class_weight = np.exp(class_weight)
           

            self.class_weight_list.append(class_weight)
            self.neighInitEmbIndexList.append(posInInitTensor)
            print(len(self.ground_truth))
            print('-------')

        

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass



        
import torch.autograd.profiler as profiler

class Model(nn.Module):
    def __init__(self, hdim, outputNum):
        super(Model, self).__init__()
        
        self.hdim = hdim
        self.outputNum = outputNum

        self.RELU = torch.nn.ReLU(inplace=True)
  
        self.fc_init = nn.Linear(20, self.hdim, bias=True)
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')
        # self.conv1_for_g = GINConv(nn.Linear(hdim, hdim, bias=True), 'mean')
        # self.conv2_for_g = GINConv(nn.Linear(hdim, hdim, bias=True), 'mean')
        self.gnn_bn = torch.nn.BatchNorm1d(hdim)
        self.gnn_bn2 = torch.nn.BatchNorm1d(hdim)
        
        self.fc = nn.Linear(self.hdim*3, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True) 
        self.fc3 = nn.Linear(256, 256, bias=True)
        self.fc4 = nn.Linear(256, 1, bias=True)  
        self.bn = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(256)

        self.dp = torch.nn.Dropout(0.5)
 


    def forward(self, qList, pgNodeEmbList, neighEmbList, classWeightList):
    
        batch_size = len(pgNodeEmbList)
        number_of_outputs = self.outputNum   # do not forget to set it for different dataset

        qList.ndata['h'] = self.fc_init(qList.ndata['h'])
        qList.ndata['h'] = self.RELU(self.gnn_bn(self.conv1_for_g(qList, qList.ndata['h'])))
        qList.ndata['h'] = self.RELU(self.gnn_bn2(self.conv2_for_g(qList, qList.ndata['h'])))
        qemb = dgl.mean_nodes(qList, 'h')

        a = torch.cat([qemb, pgNodeEmbList], 1) 
        a = a.repeat(1, number_of_outputs).view(-1, self.hdim*2)
                
        b = torch.cat([a, neighEmbList], 1)
        
        H = self.RELU(self.bn(self.fc(b)))
        H2 = self.RELU(self.bn2(self.fc2(H)))
        H3 = self.RELU(self.bn3(self.fc3(H2)))
        preds = torch.sigmoid(self.fc4(H3))

        preds = preds.view(batch_size, number_of_outputs)

        return preds, H3  


bceLoss = nn.BCELoss()
mseLoss = nn.MSELoss()
  

def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, min=1e-6, max=1-1e-6)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def myloss(epoch, preds, gts, mask_of_1_list, mask_of_0_list, classWeightList):
    bce = weighted_binary_cross_entropy(preds, gts, [10.0, 1.0])
    return bce


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


from sklearn.metrics import confusion_matrix

def myloss_for_test(preds, gts, thr):
    fpList = []
    fnList = []
    tpList = []
    tnList = []
    fprList = []
    fnrList = []
    tprList = []
    for i in range(0, preds.shape[0]):
        pred = preds[i]
        gt = gts[i]
        pred = pred.view(-1,1).cpu().detach().numpy()
        gt = gt.view(-1,1).cpu().detach().numpy()
        y = (pred > thr)
        y = y.astype(int)
        
        TP, FP, TN, FN = perf_measure(gt, y)
        fpList.append(FP)
        fnList.append(FN)
        tpList.append(TP)
        tnList.append(TN)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN+0.000001)
        # Specificity or true negative rate
        TNR = TN/(TN+FP+0.000001) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP+0.000001)
        # Negative predictive value
        NPV = TN/(TN+FN+0.000001)
        # Fall out or false positive rate
        FPR = FP/(FP+TN+0.000001)
        # False negative rate
        FNR = FN/(TP+FN+0.000001)
        # False discovery rate
        FDR = FP/(TP+FP+0.000001)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        fprList.append(FPR)
        fnrList.append(FNR)
        tprList.append(TPR)
        
    fpList = np.array(fpList)
    fnList = np.array(fnList)
    tpList = np.array(tpList)
    tnList = np.array(tnList)
    fprList = np.array(fprList)
    fnrList = np.array(fnrList)
    tprList = np.array(tprList)

    return np.mean(fprList), np.mean(fnrList), np.mean(tprList), np.mean(fpList), np.mean(fnList), np.mean(tpList), np.mean(tnList)



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
        gEmbMap[gID] = gEmb#.cuda()
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

    return dg#.to(torch.device('cuda:'+str(GPUID)))



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


def make_big_init_emb_tensor(gID2InitEmbMap, hdim):
    gid2posMap = {}
    embList = []
    for k,v in gID2InitEmbMap.items():
        embList.append(v)
        gid2posMap[k] = len(embList)-1
    embList.append(torch.zeros(hdim))
    
    return torch.stack(embList), gid2posMap


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



from functools import partial


def my_collate_fn(samples, gInitEmbTensor):
    qEmbs, pgNodes, neighIndexLists, gts, mask_of_1_list, mask_of_0_list, classWeightList = map(list, zip(*samples))

    neighIndexLists = torch.tensor(neighIndexLists).view(1,-1).squeeze()
    neighInitEmbs = torch.index_select(gInitEmbTensor, 0, neighIndexLists)

    return dgl.batch(qEmbs), torch.stack(pgNodes), neighInitEmbs, torch.tensor(gts), torch.stack(mask_of_1_list), torch.stack(mask_of_0_list), torch.tensor(classWeightList)




if __name__ == '__main__':

    D_of_pg = 80 # this is the max degree of pg
    model_prediction_num = 10 # 80/8=20
    prune_margin = 1 

    GPUID=7
    st = GPUID*10 # 8 GPU cards
    ed = (GPUID+1)*10
    
    GPUID = GPUID % 4


    print("st ", st)
    print("ed ", ed)
    print('GPUID ', GPUID)

    if ed > 80:
        print('ERROR! ed > 80')
        exit(-1)

    torch.cuda.set_device(GPUID)


    #####################################################################################################
    ### do the job as follows
    #####################################################################################################


    emb_dim = 512 # dim of embedding
    gID2InitEmbMap = read_initial_gemb('data/AIDS/emb/aids'+str(emb_dim)) # it is pre-computed by node2vec on csr
    gInitEmbBigTensor, gID2InitTensorIndexMap = make_big_init_emb_tensor(gID2InitEmbMap, emb_dim) 
    print('read g init emb done.')
    print("gInitEmbBigTensor.shape", gInitEmbBigTensor.shape)
    


    entire_dataset = read_and_split_to_individual_graph("data/AIDS/aids.txt", 0, 10000000)
    print('entire_dataset len: ', len(entire_dataset))
    gid2dgmap = {}
    for g in entire_dataset:
        dg = make_a_dglgraph(g)
        dg = dgl.add_self_loop(dg)
        gid2dgmap[g.graph.get('id')] = dg


    train_data = GINDataset("aids", 'aids_train.perc20.data', gid2dgmap, gID2InitEmbMap, gID2InitTensorIndexMap, D_of_pg, prune_margin, isTrain=True)
    test_data = GINDataset("aids", 'aids_train.perc20.data', gid2dgmap, gID2InitEmbMap, gID2InitTensorIndexMap, D_of_pg, prune_margin, isTrain=False)


    dataloader = DataLoader(
        train_data,
        batch_size=1000,
        collate_fn=partial(my_collate_fn, gInitEmbTensor=gInitEmbBigTensor),
        num_workers=6,
        drop_last=False,
        shuffle=True)

    testdataloader = DataLoader(
        test_data,
        batch_size=1,
        collate_fn=partial(my_collate_fn, gInitEmbTensor=gInitEmbBigTensor),
        drop_last=False,
        shuffle=False)


    n_epochs = 1000 # epochs to train
    lr = 0.05 # learning rate
    l2norm = 0 # L2 norm coefficient


    # create model
    model = Model(hdim=emb_dim, outputNum=model_prediction_num)
    model.cuda()

   
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = l2norm)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.95)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    start_train_time = time.time()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(n_epochs):
            old_lr = optimizer.param_groups[0]['lr']
            print("cur lr: ", old_lr)
            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), "aids.D"+str(D_of_pg)+".perc20_model_save/prune_ged"+str(st)+"_"+str(ed)+'.e'+str(epoch)+".pkl")
            model.train()
            batch_count = 0
            for qids, pgNodes, neighEmbList, gts, mask_of_1_list, mask_of_0_list, classWeightList in dataloader:
                preds, H3 = model(qids.to(torch.device('cuda:'+str(GPUID))), pgNodes.cuda(), neighEmbList.cuda(), classWeightList.cuda())
                loss = myloss(epoch, preds, gts.float().cuda(), mask_of_1_list.cuda(), mask_of_0_list.cuda(), classWeightList.cuda())
                if batch_count % 10 == 0:
                    print('='*40+" epoch ", epoch, "batch ", batch_count)
                    print('loss ', loss.item())
                    print('st ', st, ' ed ', ed, ' GPUID ', GPUID)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()  
                batch_count += 1 


            # do a test after an epoch
            if epoch % 10 == 0:
                print("do test ....")
                model.eval()
                with torch.no_grad():
                    # do test
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm1d):
                            m.track_running_stats=False  
                    print("start test ................")
                    print("epoch", epoch)
                    # for qids, pgNodes, gts in testdataloader:
                    for qids, pgNodes, neighEmbList, gts, index_of_1_list, index_of_0_list in testdataloader:
                        preds = model(qids.to(torch.device('cuda:'+str(GPUID))), pgNodes.cuda(), neighEmbList)
                        fpr, fnr, tpr, FP, FN, TP, TN = myloss_for_test(preds, gts, 0.5)
                        print('fpr ', fpr, 'fnr ', fnr, 'tpr ', tpr)
                        print('preds', preds)
                        print('gts', gts)
                        print('index_of_1_list', index_of_1_list)
                        print('index_of_0_list', index_of_0_list)
                        pred_1_probs = torch.masked_select(preds, index_of_1_list)
                        pred_0_probs = torch.masked_select(preds, index_of_0_list)
                        print('pred_1_probs', pred_1_probs)
                        print('pred_0_probs', pred_0_probs)
                        fnList = (pred_1_probs < 0.5).int().view(1,-1).squeeze(dim=0).cpu().detach().numpy().tolist()
                        fpList = (pred_0_probs > 0.5).int().view(1,-1).squeeze(dim=0).cpu().detach().numpy().tolist()
                        print('fnList', fnList)
                        print('fpList', fpList)
                        fnRatio = sum(fnList)/(len(fnList)+0.0000001)
                        fpRatio = sum(fpList)/(len(fpList)+0.0000001)
                        print(' fpR ', fpRatio, 'fnR ', fnRatio)
                        print("+++++++"*5, 'test finish')


    # do a test after an epoch
    print("do test ....")
    model.eval()
    with torch.no_grad():
        # do test
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats=False
                  
        
        print("start test ................")
        print("epoch", epoch)
        for qids, pgNodes, neighEmbList, gts, index_of_1_list, index_of_0_list in testdataloader:
            preds = model(qids.to(torch.device('cuda:'+str(GPUID))), pgNodes.cuda(), neighEmbList.cuda())
            fpr, fnr, tpr, FP, FN, TP, TN = myloss_for_test(preds, gts, 0.5)
            print('fpr ', fpr, 'fnr ', fnr, 'tpr ', tpr)
            print('preds', preds)
            print('gts', gts)
            print('index_of_1_list', index_of_1_list)
            print('index_of_0_list', index_of_0_list)
            pred_1_probs = torch.masked_select(preds, index_of_1_list)
            pred_0_probs = torch.masked_select(preds, index_of_0_list)
            print('pred_1_probs', pred_1_probs)
            print('pred_0_probs', pred_0_probs)
            fnList = (pred_1_probs < 0.5).int().view(1,-1).squeeze(dim=0).cpu().detach().numpy().tolist()
            fpList = (pred_0_probs > 0.5).int().view(1,-1).squeeze(dim=0).cpu().detach().numpy().tolist()
            print('fnList', fnList)
            print('fpList', fpList)
            fnRatio = sum(fnList)/(len(fnList)+0.0000001)
            fpRatio = sum(fpList)/(len(fpList)+0.0000001)
            print(' fpR ', fpRatio, 'fnR ', fnRatio)
            print("+++++++"*5, 'test finish')        

    end_train_time = time.time()
    print('train time (s) ', (end_train_time - start_train_time))


    print("st ", st)
    print("ed ", ed)
    print('GPUID ', GPUID)


    torch.save(model.state_dict(), "aids.D"+str(D_of_pg)+".perc10_model_save/prune_ged"+str(st)+"_"+str(ed)+".pkl")
