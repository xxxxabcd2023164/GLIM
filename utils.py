import torch
import torch.nn.functional as F
import networkx as nx
import logging
import os

def create_logger(args):
    log_file = os.path.join(args.output + 'test.log')
    logger = logging.getLogger()
    log_level = logging.INFO
    logger.setLevel(level=log_level)
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info("PARAMETER" + "-" * 10)
    for attr, value in sorted(args.__dict__.items()):
        logger.info("{}={}".format(attr.upper(), value))
    logger.info("---------" + "-" * 10)
    return logger

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def add_pseudo_train(model, optimizer, pseudo_label, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.pseudo_train_mask], pseudo_label[data.pseudo_train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    pred_ori = model(data.x, data.edge_index, data.edge_weight)
    pred_class = pred_ori.argmax(dim=-1)
    pred_softmax = torch.softmax(pred_ori, dim=1)
    pred_max_class_prob = pred_softmax.max(dim=1)[0]
    pred_entropy = torch.sum(-torch.log2(pred_softmax) * pred_softmax, dim=1)
    pred_result = (pred_class == data.y).type(torch.long)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred_class[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs, pred_result, pred_entropy, pred_class, pred_max_class_prob, pred_softmax

def find123Nei(G, node):
    nei1_li = []
    nei2_li = []
    nei3_li = []
    for FNs in list(nx.neighbors(G, node)):  # find 1_th neighbors
        nei1_li.append(FNs)
    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):  # find 2_th neighbors
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)
    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)
    nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
    if node in nei3_li:
        nei3_li.remove(node)
    return nei1_li, nei2_li, nei3_li


def cal_local_clustering_coefficient(G, A, node):
    hop2_node = torch.where(A[node] > 0)[0]
    if len(hop2_node) == 1 or len(hop2_node) == 0:
        return 0
    else:
        local_graph = G[hop2_node, :][:, hop2_node]
        edge_num = len(torch.where(local_graph > 0)[0]) / 2
    return edge_num / (len(hop2_node) * (len(hop2_node) - 1) / 2)

def construct_local_graph(G, X, node, adj):
    hop1_node = torch.where(G[node] > 0)[0]
    local_graph = G[hop1_node, :][:, hop1_node]
    local_feature = X[hop1_node, :]
    return local_graph, local_feature, hop1_node
