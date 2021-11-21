
from collections import namedtuple
import dgl
from dgl.heterograph import DGLHeteroGraph
import dgl.function as fn
from dgl.nn import GINConv
from dgl.udf import NodeBatch
import networkx as nx
import os
import torch
from torch import Tensor
from typing import List, Tuple
from visualize_graph import visualize_wl_label_graph

class gen_wl_label_udf:
    """
    A reduce function to generate wl label.

    src: source label, e.g. wl_label_0
    dst: destination label, e.g. wl_label_1
    src_collection_to_dst: {(0, 1, 2): 4, (1, 2, 3): 5, ...}
    dst_to_src_collection: {4: (0, 1, 2), 5: (1, 2, 3), ...}
    """
    def __init__(self, src: str, dst: str, start_idx: int):
        self.src: str = src
        self.dst: str = dst
        self.start_idx: int = start_idx
        self.src_collection_to_dst = {}
        self.dst_to_src_collection = {}

    def __call__(self, nodes: NodeBatch):
        mailbox: dict = nodes.mailbox    
        
        # shape: (number of nodes with same number of messages received, number of messages received)
        m: torch.tensor = mailbox[self.src]
        
        # source label, e.g. wl_label_0
        # after unsqueeze, shape should be (n, 1)
        src_label: torch.tensor = nodes.data[self.src].unsqueeze(-1)    

        # collect messages from neighbors and combine with src_label
        # shape: (number of nodes with same number of messages received, number of messages received + 1)
        # src_label_collection = torch.cat((src_label, m), 1).type(torch.int)   
        src_label_collection = m.type(torch.int)    

        # sort along column dimension
        sorted_src_label_collection, _ = torch.sort(src_label_collection, 1)   

        # relabel src_label_collection to new label to 0, 1, 2, ...
        dst_label = torch.zeros(len(sorted_src_label_collection))
        for i, l in enumerate(sorted_src_label_collection):
            l_tuple: Tuple[int] = tuple(l.tolist())
            if l_tuple not in self.src_collection_to_dst:
                # start from 1 to differentiate the ones who has no message received
                dst_label_idx = self.start_idx + len(self.src_collection_to_dst)
                self.src_collection_to_dst.update({ l_tuple : dst_label_idx })
                self.dst_to_src_collection.update({ dst_label_idx : l_tuple })
                dst_label[i] = dst_label_idx
            else:
                dst_label[i] = self.src_collection_to_dst[l_tuple]
        return {self.dst: dst_label}


def compress_graph(gid: int, g: DGLHeteroGraph, max_degree: int) -> DGLHeteroGraph:
    """
    1. Re-label all nodes with new wl_label_0
    2. Use update_all to generate wl_label_1 with wl_label_0
    3. Use update_all to generate wl_label_2 with wl_label_2
    4. Generate a new graph with wl_label_0, wl_label_1, wl_label_2 and hGx nodes
    5. Create edges
    """

    # Relabel all nodes with new wl_label_0. 
    # The number of wl_label_0 is the same as the number of unique labels in the graph.
    unique_label_tensor: Tensor = torch.unique(g.ndata["label"]).type(torch.long)
    unique_label_list: List = sorted(unique_label_tensor.tolist())
    n_wl_label_0 = len(unique_label_list)
    label_to_wl_label_0 = { l: idx for (idx, l) in enumerate(unique_label_list) }
    wl_label_0_to_label = { idx: l for (idx, l) in enumerate(unique_label_list) }
    n_data_wl_label_0 = g.ndata["label"].clone().detach()
    n_data_wl_label_0.apply_(lambda l: label_to_wl_label_0[int(l)])
    g.ndata["wl_label_0"] = n_data_wl_label_0

    # Calculate wl label 1 and the mapping from wl labe 0 to wl label 1
    wl_label_0_to_1 = gen_wl_label_udf("wl_label_0", "wl_label_1", n_wl_label_0)
    g.update_all(fn.copy_u('wl_label_0', 'wl_label_0'), wl_label_0_to_1)
    n_wl_label_1 = len(wl_label_0_to_1.dst_to_src_collection)

    # Calcualte wl label 2 and the mapping from wl label 1 to wl label 2
    wl_label_1_to_2 = gen_wl_label_udf("wl_label_1", "wl_label_2", n_wl_label_0 + n_wl_label_1)
    g.update_all(fn.copy_u('wl_label_1', 'wl_label_1'), wl_label_1_to_2)
    n_wl_label_2 = len(wl_label_1_to_2.dst_to_src_collection)

    # Final node for each graph
    hGx = n_wl_label_0 + n_wl_label_1 + n_wl_label_2

    # Create a new graph
    edges = []
    edges_to_ids = {}
    edge_weight = []
    wl_label_0_to_1_edge = []
    wl_label_1_to_2_edge = []
    wl_label_2_to_hGx_edge = []

    def upsert_edge(edge: Tuple[int], label_edge_list: List[int]):
        if edge in edges_to_ids:
            # If exists add edge weight by 1
            edge_id = edges_to_ids[edge]
            edge_weight[edge_id] += 1
        else:
            # If not exists, add a new edge
            edges_to_ids[edge] = len(edge_weight)
            label_edge_list.append(edge)
            edge_weight.append(1)
            edges.append(edge)

    processed_wl_label_1 = set()
    processed_wl_label_2 = set()
    # g.nodes() -> tensor([0, 1, 2, 3, 4])
    for node_idx in g.nodes():
        label = g.ndata["label"][node_idx]
        wl_label_0 = int(g.ndata["wl_label_0"][node_idx])
        wl_label_1 = int(g.ndata["wl_label_1"][node_idx])
        wl_label_2 = int(g.ndata["wl_label_2"][node_idx])

        # Add edges from wl_label_0 to wl_label_1
        wl_label_0_collection = wl_label_0_to_1.dst_to_src_collection[wl_label_1]
        if wl_label_1 not in processed_wl_label_1:
            for wl_label_0 in wl_label_0_collection:
                edge = (wl_label_0, wl_label_1)
                upsert_edge(edge, wl_label_0_to_1_edge)
            processed_wl_label_1.add(wl_label_1)
        
        # Add edges from wl_label_1 to wl_label_2
        wl_label_1_collection = wl_label_1_to_2.dst_to_src_collection[wl_label_2]
        if wl_label_2 not in processed_wl_label_2:
            for wl_label_1 in wl_label_1_collection:
                edge = (wl_label_1, wl_label_2)
                upsert_edge(edge, wl_label_1_to_2_edge)
            processed_wl_label_2.add(wl_label_2)
        
        # Add edges from wl_label_2 to hGx
        edge = (wl_label_2, hGx)
        upsert_edge(edge, wl_label_2_to_hGx_edge)

    # new_g is directional graph
    # Only wl_label_0 nodes have valid h one hot encoding
    diagonal_ones = torch.eye(max_degree)
    wl_label_0_h_one_hot = diagonal_ones.index_select(0, torch.LongTensor(unique_label_list))
    
    new_g = dgl.graph(tuple(zip(*edges)))
    new_g.edata["weight"] = torch.tensor(edge_weight)
    new_g.ndata['h'] = torch.zeros(new_g.num_nodes(), max_degree)
    new_g.ndata['h'][:n_wl_label_0] = wl_label_0_h_one_hot
    return {
        "old_g": g,
        "new_g": new_g,
        "wl_label_0_nodes": sorted(list(wl_label_0_to_label.keys())),
        "wl_label_1_nodes": sorted(list(wl_label_0_to_1.dst_to_src_collection.keys())),
        "wl_label_2_nodes": sorted(list(wl_label_1_to_2.dst_to_src_collection.keys())),
        "wl_label_0_to_label": wl_label_0_to_label,
        "wl_label_1_to_0": wl_label_0_to_1.dst_to_src_collection,
        "wl_label_2_to_1": wl_label_1_to_2.dst_to_src_collection,
        "wl_label_0_to_1_edge": wl_label_0_to_1_edge,
        "wl_label_1_to_2_edge": wl_label_1_to_2_edge,
        "wl_label_2_to_hGx_edge": wl_label_2_to_hGx_edge
    }


def read_and_split_to_individual_graph(fname):
    
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
                g.add_edge(tmp[1], tmp[2], att="0")
    
        glist.append(g)
    
    return glist


def make_a_dglgraph(g: nx.classes.graph.Graph) -> DGLHeteroGraph:
    
    # max_deg = 40 # largest node degree 
    # ones = torch.eye(max_deg)
   
    edges = [[],[]]
    for edge in g.edges():    # create un-directed graph
        end1 = edge[0]
        end2 = edge[1]
        edges[0].append(int(end1))
        edges[1].append(int(end2))
        edges[0].append(int(end2))
        edges[1].append(int(end1))
    dg:DGLHeteroGraph = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])))
    
    # h0 = dg.in_degrees().view(1,-1).squeeze()    # h0.shape -> torch.Size([19])
    dg.ndata['label'] = dg.in_degrees().type(torch.float)
    # h0 = ones.index_select(0, h0).float()    # convert to one hot tensor, h0.shape -> torch.Size([19, 20])
    # dg.ndata['h'] = h0

    return dg

if __name__ == "__main__":

    entire_dataset: List[nx.classes.graph.Graph] = read_and_split_to_individual_graph("aids.txt")
    entire_dataset = [g for g in entire_dataset if g.number_of_nodes() < 10]
    entire_dataset = sorted(entire_dataset, key=lambda x: x.number_of_edges())
    print('entire_dataset len: ', len(entire_dataset))
    
    output_dir = "visualized_graphs"

    gid2dgmap = {}    # key: graph id, value: DGLHeteroGraph
    for g in entire_dataset:
        gid = g.graph.get('id')
        # if gid != "3134":
        #     continue
        dg = make_a_dglgraph(g)
        dg = dgl.add_self_loop(dg)
        gid2dgmap[gid] = dg
        cg = compress_graph(gid, dg, 20)
        visualize_wl_label_graph(gid, cg, output_dir)
    
    # remove useless *.svg files and keep *.svg.svg files
    for i in os.listdir(output_dir):
        if i.endswith(".svg") and not i.endswith(".svg.svg"):
            os.remove(os.path.join(output_dir, i))