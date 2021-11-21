import torch
import networkx as nx
import random
import time
import subprocess
import heapq
import logging
import os
import numpy as np
import jpype
# from Properties import *

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
from sklearn.cluster import KMeans
import heapq


logging.basicConfig(level=logging.ERROR)
DEBUG = False


jarpath = os.path.join(os.path.abspath('.'), 'graph-matching-toolkit/graph-matching-toolkit.jar')
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % jarpath)
javaClass = jpype.JClass('algorithms.GraphMatching')


GPUID = 2

modelMap = {}


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
 



def read_PG(fname):
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
        g = nx.Graph(id = gid)

        for idy in range(1, len(cur_g)):
            tmp = cur_g[idy].split(' ')
            if tmp[0] == 'v':
                g.add_node(tmp[1], att="0")
            if tmp[0] == 'e':
                g.add_edge(tmp[1], tmp[2], ged=float(tmp[3]))

        glist.append(g)

    return glist




def readG2GDistBook(fname):
    '''
    store distance between two graphs, from G1 to G2 and from G2 to G1
    '''
    f = open(fname)
    lines = f.read()
    f.close()
    lines = lines.strip()
    lines = lines.split('\n')
    distBook = {}
    for line in lines:
        tmp = line.split(" ")

        if tmp[0] in distBook:
            distBook[tmp[0]][tmp[1]] = float(tmp[2])
        else:
            distBook[tmp[0]] = {}
            distBook[tmp[0]][tmp[1]] = float(tmp[2])
        if tmp[1] in distBook:
            distBook[tmp[1]][tmp[0]] = float(tmp[2])
        else:
            distBook[tmp[1]] = {}
            distBook[tmp[1]][tmp[0]] = float(tmp[2])
        
        # dist from self to self is zero
        distBook[tmp[0]][tmp[0]] = 0
        distBook[tmp[1]][tmp[1]] = 0
    
    return distBook



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


def getSubG2GDistBookGivenQueries(G2GDistBookFileName, qGList): 
    # we only need the distance from qG in qGList to others
    f = open(G2GDistBookFileName+".sub4q.txt", 'w')
    allDistBook = readG2GDistBook(G2GDistBookFileName)
    for q in qGList:
        distList_of_q = allDistBook[q.graph.get('id')]
        for key,value in distList_of_q.items():
            f.write(q.graph.get('id')+' '+key+' '+str(value)+"\n")
    f.close()

    



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




def getProdGraphV4(g1, g2):
    small_g = None
    large_g = None
    if g1.number_of_nodes() <= g2.number_of_nodes():
        small_g = g1
        large_g = g2
    else:
        small_g = g2
        large_g = g1

    smallG_nodeList = list(small_g.nodes())
    largeG_nodeList = list(large_g.nodes())
    smallG_nodeList.extend([str(-ele) for ele in range(1, len(largeG_nodeList)-len(smallG_nodeList)+1)])
    
    # sampled_smallG_nodes = random.sample(smallG_nodeList, max([5, int(len(smallG_nodeList)/10)]))
    # sampled_largeG_nodes = random.sample(largeG_nodeList, max([5, int(len(largeG_nodeList)/10)]))
    sampled_smallG_nodes = random.sample(smallG_nodeList, max([min([len(smallG_nodeList), 10]), int(len(smallG_nodeList)/10)]))
    sampled_largeG_nodes = random.sample(largeG_nodeList, max([min([len(largeG_nodeList), 10]), int(len(largeG_nodeList)/10)]))

    prod_g = nx.Graph()

    for n1 in sampled_smallG_nodes:
        if int(n1) >= 0:
            n1_neighs = list(small_g[n1])
        else:
            n1_neighs = []
        for n2 in sampled_largeG_nodes:
            n2_nonNeighs = random.sample(largeG_nodeList, max([3, int(len(largeG_nodeList)/10)]))
            for n1_neigh in n1_neighs:
                for n2_nonNeigh in n2_nonNeighs:
                    if large_g.has_edge(n2,n2_nonNeigh) == False:
                        prod_g.add_edge(n1+"|"+n2, n1_neigh+"|"+n2_nonNeigh)
    

    for n2 in sampled_largeG_nodes:
        n2_neighs = list(large_g[n2])
        for n1 in sampled_smallG_nodes:
            n1_nonNeighs = random.sample(smallG_nodeList, max([3, int(len(smallG_nodeList)/10)]))
            for n2_neigh in n2_neighs:
                for n1_nonNeigh in n1_nonNeighs:
                    if small_g.has_edge(n1,n1_nonNeigh) == False:
                        prod_g.add_edge(n1+"|"+n2, n1_nonNeigh+"|"+n2_neigh)


    if prod_g.number_of_edges() == 0:
        print(smallG_nodeList)
        print(largeG_nodeList)
        exit(-1)

    # print("prod_g nodes: ", prod_g.number_of_nodes())
    # print("prod_g edges: ", prod_g.number_of_edges())
    largest_deg = -1
    for node in prod_g:
        if len(prod_g[node]) > largest_deg:
            largest_deg = len(prod_g[node])

    return prod_g, largest_deg



estGEDBuffer = {}

def getDist(q, g, distBook, isQuery=False):
    '''
    q: the query graph
    g: the data graph
    distBook: pre-computed exact GED
    GEDEstimator: GNN model
    estScoreBuffer_for_q: used in query processing, to store the predictions of GED just for q, key is g, values is estimated GED
    global_estScoreBuffer: used in pg construction to store the predictions of GED, key is a graph, value is {another graph : estimated GED}
    '''
    qid = q.graph.get("id")
    gid = g.graph.get("id")
    # print("qid: ", qid)
    # print("gid: ", gid)

    if qid == gid:
        return 0


    if isQuery is False:
        if qid not in distBook or gid not in distBook[qid]:
            
            if qid in estGEDBuffer:
                if gid in estGEDBuffer[qid]:
                    return estGEDBuffer[qid][gid]

            # estimate GED   
            distance = javaClass.runApp("data/AIDS/g"+qid+".txt", "data/AIDS/g"+gid+".txt")
            distance = distance * 2.0
            
            if qid in estGEDBuffer:
                if gid not in estGEDBuffer[qid]:
                    estGEDBuffer[qid][gid] = distance
            else:
                estGEDBuffer[qid] = {}
                estGEDBuffer[qid][gid] = distance
            
            if gid in estGEDBuffer:
                if qid not in estGEDBuffer[gid]:
                    estGEDBuffer[gid][qid] = distance
            else:
                estGEDBuffer[gid] = {}
                estGEDBuffer[gid][qid] = distance

            print(qid, gid, "no exact GED, estimate it", distance)
            

            return distance
        else:
            # have pre-computed the exact GED
            return distBook[qid][gid]
    else:
        # compute distance on-the-fly for runtime testing

        if qid in estGEDBuffer:
            if gid in estGEDBuffer[qid]:
                return estGEDBuffer[qid][gid]

        distance = getExactDist("data/AIDS/g"+qid+".txt", "data/AIDS/g"+gid+".txt", 10000000, 10) # 10 seconds limitation for exact GED computation
        if distance < 0:
            distance = javaClass.runApp("data/AIDS/g"+qid+".txt", "data/AIDS/g"+gid+".txt")
            distance = distance * 2.0
        
        if qid in estGEDBuffer:
            if gid not in estGEDBuffer[qid]:
                estGEDBuffer[qid][gid] = distance
        else:
            estGEDBuffer[qid] = {}
            estGEDBuffer[qid][gid] = distance
            
        if gid in estGEDBuffer:
            if qid not in estGEDBuffer[gid]:
                estGEDBuffer[gid][qid] = distance
        else:
            estGEDBuffer[gid] = {}
            estGEDBuffer[gid][qid] = distance
        
        return distance
       

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


def myloss_for_test(preds, gts):
 
        
    TP, FP, TN, FN = perf_measure(gts, preds)
    
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

    
    return FN, FP, FNR, FPR


def greedy_search(proximityGraph, queryGraph, k, ep, ef, distBook, gid2gmap):
    '''
    k: return top-k from ef candidates
    ep: list of start nodes
    ef: number of candidates
    '''
    cand, stat = search_layer(proximityGraph, queryGraph, ep, ef, distBook, gid2gmap)
    while len(cand) > k:
        heapq.heappop(cand)
    
    return cand, stat


exact_ans = None

def search_layer(proxG, q, ep, ef, G2GDistBook, gid2gmap):
    '''
    At each hop, use GED ranking neural network to rank the nodes 
    parm: proxG: current proximity graph. each node in proxG is a networkx graph
    parm: q: query, which is a networkx graph
    parm: ep: enter points for the search in proxG. should be a list
    parm: ef: number of results
    '''
    bar = 1000

    aaa = get_topkAll_in_a_list(200, exact_ans[qid])
    aaa = set([ele[0] for ele in aaa])
   
    if len(ep) == 0:
        logging.error("Error: no enter point")
        exit(-1)

    
    DCS = 0.0 # count of distance computation
    hop_count = 0.0 # the number of hops
    visited = set()
    C = [] # search frontier, stored by a min-heap
    idx_C = 0 # required by min-heap 
    W = [] # result to return. dynamic list of found nearest neighbors. a max-heap, which can be realized by storing the negative value in a min-heap.
    idx_W = 0 # required by max-heap 
    
    hop_enter_neighorhood = None
    time_used_before_enter_neighorhood = 0
    time_used_after_enter_neighorhood = 0
    DCS_before_enter_neighorhood = 0

    model_predicted_nodes = set()

    for ele in ep: 
        dist = getDist(q, ele, G2GDistBook)
        DCS = DCS + 1

        if hop_enter_neighorhood is None:
            DCS_before_enter_neighorhood += 1
                
        heapq.heappush(C, (dist, idx_C, ele))
        idx_C = idx_C + 1
        heapq.heappush(W, (-dist, idx_W, ele))
        idx_W = idx_W + 1
        visited.add(ele.graph.get("id"))   

    while len(C) > 0:
        if DEBUG:
            print("-"*60)
            print("C:")
            for ele in C:
                print(str(ele[0])+" g"+ele[2].graph.get("id"))
            print("+++++++"*3)
            print("W:")
            for ele in W:
                print(str(ele[0])+" g"+ele[2].graph.get("id"))
            print("+++++++"*3)
            print("visited:")
            print(visited)
            print("+++++++"*3)

        c = heapq.heappop(C) # extract the nearest in C to q
        f = min(W) # get the furthest from W to q
        c_dist = getDist(q, c[2], G2GDistBook)  
        f_dist = getDist(q, f[2], G2GDistBook)
        # print([c[0], c[2].graph.get('id')])
    
        logging.debug("c is g"+c[2].graph.get("id")+ " dist=" + str(c_dist))
        logging.debug("f is g"+f[2].graph.get("id")+ " dist=" + str(f_dist))
        
        
        if c_dist > f_dist:
            break


        # list_of_W = list(W)
        # list_of_W.sort(key = lambda x: -x[0])
        
        # if len(list_of_W) < bar:
        #     if c_dist > f_dist:
        #         break
        # else:
        #     if c_dist > -list_of_W[bar-1][0]:
        #         break

        
             
        
        logging.debug("ok to go on ...")
        hop_count = hop_count + 1

       
        if c[2].graph.get('id') in aaa:
            if hop_enter_neighorhood is None:
                hop_enter_neighorhood = hop_count
            

        neighbors_of_c = list(proxG[c[2]])
        logging.debug("neighbors of c: " + str(["g"+ele.graph.get("id") for ele in neighbors_of_c]))
        
       

    
        if hop_count >= 0:
            # print('c_dist', c_dist)
            neighIDs = [neigh.graph.get('id') for neigh in neighbors_of_c]
            neighIDs.sort()
            # print(neighIDs)
            # neighDists = [getDist(q, gid2gmap[neighID], G2GDistBook) for neighID in neighIDs]
            # print(neighDists)

            for neighID in neighIDs:
                model_predicted_nodes.add(neighID)

            preds = []
            for i in range(0, 8):
                curGEDPruneModel = modelMap[i]
                curGEDPruneModel.eval()      
                with torch.no_grad():
                    for m in curGEDPruneModel.modules():
                        if isinstance(m, nn.BatchNorm1d):
                            m.track_running_stats=False

                    subNeighIDs = neighIDs[i*10 : (i+1)*10]
                    preds.append(curGEDPruneModel([q.graph.get('id')], [c[2].graph.get('id')], subNeighIDs).view(1,-1).squeeze())
            
            preds = torch.stack(preds)
            # print(preds)
            prune_decision = (preds > 0.5).int().view(1,-1).squeeze().cpu().detach().numpy()

            preds = preds.view(1,-1).squeeze().cpu().detach().numpy().tolist()

            neighID_and_preds = []
            for idx in range(0, len(neighIDs)):
                neighID_and_preds.append( (neighIDs[idx], preds[idx]) )
            
            neighID_and_preds.sort(key=lambda x: x[1])
            
            topPercNeighIDs = set()
            for ele in neighID_and_preds[ 0 : int(len(neighID_and_preds)*0.2) ]:
                topPercNeighIDs.add(ele[0])
            

        else:
            curGEDPruneModel = None
            prune_decision = None



        neighbors_of_c.sort(key=lambda x: x.graph.get('id'))
        neighIDs = [neigh.graph.get('id') for neigh in neighbors_of_c]
     

       

        for iii in range(0, len(neighbors_of_c)):
            neigh = neighbors_of_c[iii]
    
            if prune_decision is not None:
                if neigh.graph.get('id') not in topPercNeighIDs:
                    continue
           
            logging.debug("cur_neigh: g"+neigh.graph.get("id"))
            neigh_dist = getDist(q, neigh, G2GDistBook)
   

            if neigh.graph.get("id") not in visited:
                logging.debug("= not visited")

                visited.add(neigh.graph.get("id"))
                f = min(W)
                f_dist = getDist(q, f[2], G2GDistBook)
                logging.debug("= f is g"+f[2].graph.get("id") + " dist=" + str(f_dist))
                neigh_dist = getDist(q, neigh, G2GDistBook)
                DCS = DCS + 1
                logging.debug("= cur_neigh g"+neigh.graph.get("id") + " dist=" + str(neigh_dist))


                # for PG construction 
                if neigh_dist < f_dist or len(W) < ef:
                    if neigh_dist < f_dist:
                        logging.debug("= neigh_dist < f_dist")
                    if len(W) < ef:
                        logging.debug("= len(W) < ef")
                    heapq.heappush(C, (neigh_dist, idx_C, neigh))
                    idx_C = idx_C + 1
                    heapq.heappush(W, (-neigh_dist, idx_W, neigh))
                    idx_W = idx_W + 1
            
                    logging.debug("= push " + "g"+neigh.graph.get("id") + " to C and W")
                    if len(W) > ef:
                        deleted = heapq.heappop(W)
                        logging.debug("= W's size "+ str(len(W)) + " is too large, delete " + "g"+deleted[2].graph.get("id"))

            logging.debug("********")


    stat = {}
    stat["hop_count"] = hop_count
    stat["DCS"] = DCS
    stat['visited'] = visited
    stat['model_pred_count'] = len(model_predicted_nodes)
    print("DCS", DCS)
    print("model_pred_count", len(model_predicted_nodes))
    print("hop_enter_neighorhood", hop_enter_neighorhood)
    print("time_used_before_enter_neighorhood", time_used_before_enter_neighorhood)
    print("time_used_after_enter_neighorhood", time_used_after_enter_neighorhood)
    print("DCS_before_enter_neighorhood", DCS_before_enter_neighorhood)

    return W, stat




def select_neighbors_simple(q, cand, M):
    deleted = []
    while len(cand) > M:
        x = heapq.heappop(cand)
        deleted.append(x[2])
    return [ele[2] for ele in cand], deleted



def insert(proxG, q, ep, M, maxDeg0, efConst, G2GDistBook):
    cand, _ = search_layer(proxG, q, ep, efConst, G2GDistBook)
    neighs, _ = select_neighbors_simple(q, cand, M)
    
    logging.debug("insert g"+q.graph.get("id"))
    logging.debug("insert edges: "+str([ele.graph.get("id") for ele in neighs]))
    # insert q to proxG
    proxG.add_node(q) 
    # insert edges for q
    for neigh in neighs:
        proxG.add_edge(q, neigh)

    # shrink connections if needed
    for neigh in neighs:
        eConn = list(proxG[neigh])
        if len(eConn) > maxDeg0:
            # print("shink")
            logging.debug("shink g"+neigh.graph.get("id")+ " degree " + str(len(eConn)) + " exceeds maxDeg0 " + str(maxDeg0))
            eConn_heap = []
            idx_eConn = 0
            for ele in eConn:
                ele_dist = getDist(neigh, ele, G2GDistBook)
                heapq.heappush(eConn_heap, (-ele_dist, idx_eConn, ele))
                idx_eConn = idx_eConn + 1
            _, deleted = select_neighbors_simple(neigh, eConn_heap, maxDeg0)
            for ele in deleted:
                logging.debug("delete edge "+ neigh.graph.get("id") + " " + ele.graph.get("id"))
                if len(proxG[neigh]) == 1 or len(proxG[ele]) == 1:
                    pass
                else:
                    proxG.remove_edge(neigh, ele)


            



def build_proximity_graph(graphList, M, maxDeg0, efConst, G2GDistBook):
    ''' 
    The construction algorithm of the paper Hierarchical navigable small world graph HNSW PAMI2018
    But, we just construct the buttom layer of HNSW.
    - graphList: the graphs to insert to the proximity graph. Each node in the proximity graph is a graph in graphList
    - maxDeg0: the max degree of node in proximity graph. '0' means the buttom layer of HNSW. 
    - M, efConst: suppose you are inserting g to the proximity graph, you first find efConst nodes in the proximity graph as the 
    candidate neighbors of g. Then, you pick M candidates to connect with g. Note that the M nodes maybe not in the efConst candidates, if
    you use the select_neighbors_heuristic function in the HNSW paper to pick the M nodes, as select_neighbors_heuristic may check the neighbors
    of the nodes in the efConst candidates.
    - G2GDistBook: all pair distance between graphs in graphList. Just for fast construction.
    '''
    proxG = nx.Graph()
    proxG.add_node(graphList[0])

    for i in range(1, len(graphList)):
        if i % 1 == 0:
            print(i)
        if DEBUG:
            print("====================================" + str(i))
            print("cur proxG is: ")
            print(str(["g"+ele.graph.get("id") for ele in proxG.nodes()]))
            for edge in proxG.edges():
                print(edge[0].graph.get("id"), ' ', edge[1].graph.get("id"))
            print("inserting ", "g"+graphList[i].graph.get("id"))
        # insert(proxG, graphList[i], [graphList[0]], M, maxDeg0, efConst, G2GDistBook)
        rand = np.random.randint(i)
        insert(proxG, graphList[i], [graphList[rand]], M, maxDeg0, efConst, G2GDistBook)

    return proxG





def hnsw_const(G2GDistBook, data_graphs, M, maxDeg0, efConst):
    distBook = G2GDistBook
    proxG = build_proximity_graph(data_graphs, M, maxDeg0, efConst, distBook)
    print("node has: ", proxG.number_of_nodes())
    print("edge has: ", proxG.number_of_edges())
    print("cc has: ", nx.number_connected_components(proxG))
    return proxG



def save_proxG(fname, proxG):
    # write proxG into file
    f = open(fname, "w")
    f.write("t # 0\n")
    for n in proxG.nodes():
        f.write("v "+n.graph.get("id")+"\n")
    for e in proxG.edges():
        f.write("e "+e[0].graph.get("id")+" "+e[1].graph.get("id")+"\n")
    f.close()


def scan_db_and_comp_ged(q, database, distBook):
    res = []
    for g in database:
        dist = getDist(q, g, distBook)
        res.append([ g.graph.get('id'), dist])
        #print(len(res))
    res.sort(key = lambda x: x[1])
    #print(res)
    return res


def get_topkAll_in_a_list(topk, x):
    kth = x[topk-1]
    res = x[0:topk]
    for i in range(topk, len(x)):
        if x[i][1] == kth[1]:
            res.append(x[i])
    return res



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


def pgBuild(database, G2GDistBookFileName, M, maxDeg0, efConst):
    G2GDistBook = readG2GDistBook(G2GDistBookFileName)
    pg = hnsw_const(G2GDistBook, database, M, maxDeg0, efConst)
    save_proxG("hnsw.aids.M"+str(M)+".D"+str(maxDeg0)+".ef"+str(efConst)+".nx", pg)
    


def reassignNodeID(graph, fname):
    oldID2newIDMap = {}
    newG = nx.Graph()
    newNodeID = 0
    for node in graph.nodes():
        oldID2newIDMap[node] = newNodeID
        newG.add_node(newNodeID)
        newNodeID += 1
    for edge in graph.edges():
        end1 = edge[0]
        end2 = edge[1]
        new_end1 = oldID2newIDMap[end1]
        new_end2 = oldID2newIDMap[end2]
        newG.add_edge(new_end1, new_end2)
    nx.write_edgelist(newG, fname, data=False)



def getExactDist(gfile, qfile, thr, timelimit):
    """
    invoke the code of Lijun Chang (ICDE’20 paper)
    :parm gfile: data graph file name
    :parm qfile: query graph file name
    :parm thr: check if GED <= thr
    :parm timelimit: stop if reach the time limit (in seconds)
    :return: if timeout, return -2.0; if GED > thr, return -1.0; if GED <= thr, return GED
    """
    dist = -2.0
    st = time.time()
    try:
        abc = subprocess.check_output([".~/Graph_Edit_Distance/ged_debian", gfile, qfile, "astar", "LSa", str(thr)], timeout=timelimit) # timeout is in seconds
        abc = abc.decode('utf-8')
        abc2 = abc.split('\n')
        abc3 = abc2[1]
        abc3 = abc3.strip()
        abc4 = abc3.split(',')
        abc5 = abc4[0]
        abc6 = abc5.split(' ')
        dist = float(abc6[2])
        # print(abc)
        # print(dist)
        # print('gfile ', gfile)
        # print('qfile ', qfile)
        # print(dist)
    except:
        # print('gfile ', gfile)
        # print('qfile ', qfile)
        # print("time out!")
        # print(dist)
        pass
    et = time.time()
    # print("clock time (sec.) ", (et-st))
    return dist



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
        gEmbMap[gID] = gEmb.cuda()
    return gEmbMap



########################################################################################################
#### GNN model 
########################################################################################################



class Model(nn.Module):
    def __init__(self, gID2InitEmbMap, gid2dgmap):
        super(Model, self).__init__()

        self.gid2dgMap = gid2dgmap
        self.gID2InitEmbMap = gID2InitEmbMap

      

        self.RELU = torch.nn.ReLU(inplace=True)
  
        self.fc_init = nn.Linear(20, 512, bias=True)
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')
        # self.conv1_for_g = GINConv(nn.Linear(hdim, hdim, bias=True), 'mean')
        # self.conv2_for_g = GINConv(nn.Linear(hdim, hdim, bias=True), 'mean')
        self.gnn_bn = torch.nn.BatchNorm1d(512)
        self.gnn_bn2 = torch.nn.BatchNorm1d(512)
        
        self.fc = nn.Linear(512*3, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True) 
        self.fc3 = nn.Linear(256, 256, bias=True)
        self.fc4 = nn.Linear(256, 1, bias=True)  
        self.bn = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        
        self.dp = torch.nn.Dropout(0.5)
        


    def forward(self, qIDs, pgNodeIDs, neighIDs):

        outputNum = 10

        batch_dg = dgl.batch([self.gid2dgMap[qid] for qid in qIDs])
        batch_dg.ndata['h'] = self.fc_init(batch_dg.ndata['h'])
        batch_dg.ndata['h'] = self.RELU(self.gnn_bn(self.conv1_for_g(batch_dg, batch_dg.ndata['h'])))
        batch_dg.ndata['h'] = self.RELU(self.gnn_bn2(self.conv2_for_g(batch_dg, batch_dg.ndata['h'])))
        qemb = dgl.mean_nodes(batch_dg, 'h')
        qemb = qemb.view(1,-1)
        

        neighEmbList = torch.zeros(outputNum, 512).cuda()
        for idx in range(0, len(neighIDs)):
            neighID = neighIDs[idx]
            neighEmbList[idx] = self.gID2InitEmbMap[neighID]

        pgNode_embList = [self.gID2InitEmbMap[pgNodeID] for pgNodeID in pgNodeIDs]
        pgNode_embList = torch.stack(pgNode_embList).cuda()
        pgNode_embList = pgNode_embList.view(1,-1)


        a = torch.cat([qemb, pgNode_embList], 1) 
        a = a.repeat(1, outputNum).view(-1, 512*2)
    
        b = torch.cat([a, neighEmbList], 1)
  

        H = self.RELU(self.bn(self.fc(b))) 
        H2 = self.RELU(self.bn2(self.fc2(H)))
        H3 = self.RELU(self.bn3(self.fc3(H2)))
        pred = torch.sigmoid(self.fc4(H3))
        pred = pred.view(1, outputNum)
        
        return pred



########################################################################################################

entire_dataset = read_and_split_to_individual_graph("data/AIDS/aids.txt", 0, 10000000)
print(len(entire_dataset))

gid2gmap = {}
for g in entire_dataset:
    gid2gmap[g.graph.get("id")] = g
gid2dgmap = {}
for g in entire_dataset:
    dg = make_a_dglgraph(g)
    dg = dgl.add_self_loop(dg)
    gid2dgmap[g.graph.get('id')] = dg#.to(torch.device('cuda:'+str(GPUID)))


gID2InitEmbMap = read_initial_gemb('data/AIDS/emb/aids.emb')
print('read g init emb done.')


database = entire_dataset[0:40000] # aids db size = 4000


pgTmp = read_and_split_to_individual_graph("PG.aids.nx", 0, 10000000000)
pgTmp = pgTmp[0]


pgNodeIDSet = pgTmp.nodes()
pg = nx.Graph()
for nID in pgTmp.nodes():
    pg.add_node(gid2gmap[nID], deg=len(pgTmp[nID]))
for edge in pgTmp.edges():
    edge_weight = pgTmp.get_edge_data(*edge)
    pg.add_edge(gid2gmap[edge[0]], gid2gmap[edge[1]])


queries = []
f = open('data/AIDS/query_test.txt')
lines = f.read()
f.close()
lines = lines.strip().split('\n')
for line in lines:
    qid = line.strip()
    queries.append(gid2gmap[qid])
queryIDs = set()
for q in queries:
    queryIDs.add(q.graph.get('id'))



q2GDistBook = readQ2GDistBook("data/AIDS/aids.txt", pgNodeIDSet)
exact_ans = get_exact_answer(100000000, q2GDistBook)




ep = 319 # model of which epoch you want to use

model0 = Model(gID2InitEmbMap, gid2dgmap)
model0.load_state_dict(torch.load("aids.perc20_model_save/prune_ged0_10.e"+str(ep)+".pkl"))
modelMap[0] = model0.cuda()

model10 = Model(gID2InitEmbMap, gid2dgmap)
model10.load_state_dict(torch.load("aids.perc20_model_save/prune_ged10_20.e"+str(ep)+".pkl"))
modelMap[1] = model10.cuda()

model20 = Model(gID2InitEmbMap, gid2dgmap)
model20.load_state_dict(torch.load("aids.perc20_model_save/prune_ged20_30.e"+str(ep)+".pkl"))
modelMap[2] = model20.cuda()

model30 = Model(gID2InitEmbMap, gid2dgmap)
model30.load_state_dict(torch.load("aids.perc20_model_save/prune_ged30_40.e"+str(ep)+".pkl"))
modelMap[3] = model30.cuda()

model40 = Model(gID2InitEmbMap, gid2dgmap)
model40.load_state_dict(torch.load("aids.perc20_model_save/prune_ged40_50.e"+str(ep)+".pkl"))
modelMap[4] = model40.cuda()

model50 = Model(gID2InitEmbMap, gid2dgmap)
model50.load_state_dict(torch.load("aids.perc20_model_save/prune_ged50_60.e"+str(ep)+".pkl"))
modelMap[5] = model50.cuda()

model60 = Model(gID2InitEmbMap, gid2dgmap)
model60.load_state_dict(torch.load("aids.perc20_model_save/prune_ged60_70.e"+str(ep)+".pkl"))
modelMap[6] = model60.cuda()

model70 = Model(gID2InitEmbMap, gid2dgmap)
model70.load_state_dict(torch.load("aids.perc20_model_save/prune_ged70_80.e"+str(ep)+".pkl"))
modelMap[7] = model70.cuda()


def set_bn_eval(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('BatchNorm') != -1:
        print('make it eval()')
        m.eval()

for i in range(0, 8):
    modelMap[i].apply(set_bn_eval)




topk = 50

avg_recall = 0.0
avg_precision = 0.0
avg_DCS = 0.0
avg_hops = 0.0
counter = 0
start_time = time.time()
for q in queries[0:10]:
    qid = q.graph.get("id")
    print("qid: ", qid)

    exact_ans_of_q = get_topkAll_in_a_list(topk, exact_ans[qid])
    print("exact_ans_of_q: ", exact_ans_of_q)

    
    message = 'random_initial_node'
    rand = np.random.randint(len(database))
    start_nodes = [database[rand]]
    cand, stat = greedy_search(pg, q, 50, start_nodes, 50, q2GDistBook, gid2gmap)


    pred_ans_of_q = [(ele[2].graph.get('id'), ele[0]) for ele in cand]
    pred_ans_of_q.sort(key = lambda x: -x[1])
    print("pred_ans_of_q", pred_ans_of_q)
    print("DCS: ", stat['DCS'])

    avg_DCS += stat['DCS']
    avg_hops += stat['hop_count']

    recall = set([ele[0] for ele in pred_ans_of_q]) & set([ele[0] for ele in exact_ans_of_q])
    recall_perc = min(1.0, len(recall)/topk)
    print("recall perc: ", recall_perc)
    avg_recall += recall_perc
    
    precision = len(recall)/len(exact_ans_of_q)
    print('precision', precision)
    avg_precision += precision

    f.write(qid+" "+str(recall_perc)+" "+str(precision)+" "+str(stat['DCS'])+" "+str(stat['hop_count'])+"\n")

    counter += 1
    print('---------------------------------------------')
end_time = time.time()

print("avg_recall: ", (avg_recall/counter))
print("avg_precision: ", (avg_precision/counter))
print("avg_DCS: ", (avg_DCS/counter))
print('avg_hops: ', (avg_hops/counter))
print('avg_time: (s)', (end_time - start_time)/counter)
print("counter: ", counter)
print('msg: ', message)
print("ep ", ep)





jpype.shutdownJVM()
