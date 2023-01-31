# -*- coding: utf-8 -*-

import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor
import numpy as np
import networkx as nx
import time
import random
from utils import train, add_pseudo_train, test, find123Nei, cal_local_clustering_coefficient, \
    construct_local_graph, create_logger
from spectral_matching import spectral_graph_matching
from model import GCN
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='Cora, CiteSeer, PubMed, CoraFull, cs')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sparse_threshold', type=float, default=0.05)
    parser.add_argument('--dense_threshold', type=float, default=0.90)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--pseudo_rate', type=float, default=0.6)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--ass_score_threshold', type=float, default=0)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--fixed', default=True)
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument('--sparse_clustering_coefficient', type=float, default=0.1)
    parser.add_argument('--num_train_per_class', type=int, default=20)
    parser.add_argument('--split', type=str, default='public')
    t_start = time.time()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'CoraFull' or args.dataset == 'cs':
        if args.dataset == 'CoraFull':
            dataset = CoraFull(root='dataset/')
        else:
            dataset = Coauthor(root='dataset/',name=args.dataset)
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask = torch.full((1, len(data.y)), fill_value=False, dtype=bool)[0], torch.full(
            (1, len(data.y)), fill_value=False, dtype=bool)[0] \
            , torch.full((1, len(data.y)), fill_value=False, dtype=bool)[0]
        for c in range(max(data.y)+1):
            idx = (data.y == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:args.num_train_per_class]]
            data.train_mask[idx] = True

        remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]
        data.val_mask[remaining[:500]] = True

        data.test_mask[remaining[500:500 + 1000]] = True
    else:
        dataset = Planetoid(root='dataset/', name=args.dataset, num_train_per_class=args.num_train_per_class,
                            transform=T.NormalizeFeatures(),split=args.split)
        data = dataset[0]

    if args.fixed:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)
            torch.backends.cudnn.deterministic = True
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

    logger = create_logger(args)
    if args.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)

    # 》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
    # Raw GCN model
    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes,args)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    # GCN before Migration
    add_pseudo_model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes,args)
    add_pseudo_model, data = add_pseudo_model.to(device), data.to(device)
    add_pseudo_optimizer = torch.optim.Adam([
        dict(params=add_pseudo_model.conv1.parameters(), weight_decay=5e-4),
        dict(params=add_pseudo_model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    # GCN after Migration
    add_final_pseudo_model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes,args)
    add_final_pseudo_model, data = add_final_pseudo_model.to(device), data.to(device)
    add_final_pseudo_optimizer = torch.optim.Adam([
        dict(params=add_final_pseudo_model.conv1.parameters(), weight_decay=5e-4),
        dict(params=add_final_pseudo_model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    final_result = []
    match_result = []
    for cur_hop in range(args.stage):
        if cur_hop == 0:
            best_val_acc = final_test_acc = best_epoch = 0
            train_acc_all = []
            val_acc_all = []
            test_acc_all = []
            pred_class = []
            pred_entropy = []
            pred_max_class_prob = []
            pred_all_class_prob = []
            for epoch in range(1, args.epochs + 1):
                loss = train(model, optimizer, data)
                tmp_accs, tmp_pred_result, tmp_pred_entropy, tmp_pred_class, tmp_pred_max_class_prob, tmp_pred_softmax = test(
                    model, data)
                tmp_train_acc, tmp_val_acc, tmp_test_acc = tmp_accs
                pred_class.append(tmp_pred_class)
                pred_max_class_prob.append(tmp_pred_max_class_prob)
                pred_all_class_prob.append(tmp_pred_softmax)
                train_acc_all.append(tmp_train_acc)
                val_acc_all.append(tmp_val_acc)
                test_acc_all.append(tmp_test_acc)
                pred_entropy.append(tmp_pred_entropy)
                if tmp_val_acc > best_val_acc:
                    best_val_acc = tmp_val_acc
                    test_acc = tmp_test_acc
                    best_epoch = epoch
        else:
            data.pseudo_train_mask[data.test_mask] = False
            best_val_acc = final_test_acc = best_epoch = 0
            train_acc_all = []
            val_acc_all = []
            test_acc_all = []
            pred_class = []
            pred_entropy = []
            pred_max_class_prob = []
            pred_all_class_prob = []
            for epoch in range(1, args.epochs + 1):
                loss = add_pseudo_train(add_final_pseudo_model, add_final_pseudo_optimizer, pseudo_label, data)
                tmp_accs, tmp_pred_result, tmp_pred_entropy, tmp_pred_class, tmp_pred_max_class_prob, tmp_pred_softmax =  test(add_final_pseudo_model, data)
                tmp_train_acc, tmp_val_acc, tmp_test_acc = tmp_accs
                pred_class.append(tmp_pred_class)
                pred_max_class_prob.append(tmp_pred_max_class_prob)
                pred_all_class_prob.append(tmp_pred_softmax)
                train_acc_all.append(tmp_train_acc)
                val_acc_all.append(tmp_val_acc)
                test_acc_all.append(tmp_test_acc)
                pred_entropy.append(tmp_pred_entropy)
                if tmp_val_acc > best_val_acc:
                    best_val_acc = tmp_val_acc
                    test_acc = tmp_test_acc
                    best_epoch = epoch
            match_result.append(test_acc)
        final_result.append(test_acc)
        pred_class = torch.cat(pred_class).reshape(-1, pred_class[0].shape[0])
        pred_max_class_prob = torch.cat(pred_max_class_prob).reshape(-1, pred_max_class_prob[0].shape[0])
        pred_all_class_prob = torch.stack(pred_all_class_prob)
        pred_entropy = torch.cat(pred_entropy).reshape(-1, pred_entropy[0].shape[0])

        # Pseudo label generation
        select_epoch = epoch - 1   # epoch
        temp_pred_max_class_prob_all = pred_max_class_prob[select_epoch, :]
        temp_pred_class_all = pred_class[select_epoch, :]
        if cur_hop == 0:
            pseudo_label_select = temp_pred_max_class_prob_all > 2
        select_set = torch.bitwise_not(data.train_mask)
        select_index = torch.where(select_set == True)[0]
        # with training dynamic
        pred_entropy_mean = torch.mean(pred_entropy[-100:, ], dim=0)
        pred_entropy_std = torch.std(pred_entropy[-100:, ], dim=0)
        pred_entry_all = pred_entropy_mean + pred_entropy_std
        temp_select_entroy = pred_entry_all[select_index].sort(descending=False)
        pseudo_label_select_index = select_index[temp_select_entroy[1][:int(data.y.shape[0] * args.pseudo_rate)]]
        pseudo_label_select[pseudo_label_select_index] = True
        train_pseudo_mask = data.train_mask | pseudo_label_select
        data.pseudo_train_mask = train_pseudo_mask.clone()
        data.pseudo_train_mask[data.test_mask] = False
        pseudo_label = data.y.clone()
        pseudo_label[pseudo_label_select] = temp_pred_class_all[pseudo_label_select]
        logger.info(
            'Total node number: {:d}. pseudo label number: {:d}'.format(data.y.shape[0], torch.where(pseudo_label_select == True)[0].shape[0]))
        best_val_acc = 0
        add_pseudo_best_val_acc = add_pseudo_final_test_acc = best_epoch = 0
        add_pseudo_train_acc_all = []
        add_pseudo_val_acc_all = []
        add_pseudo_test_acc_all = []
        for epoch in range(1, args.epochs + 1):
            loss = add_pseudo_train(add_pseudo_model, add_pseudo_optimizer, pseudo_label, data)
            tmp_train_acc, tmp_val_acc, tmp_test_acc = test(add_pseudo_model, data)[0]
            add_pseudo_train_acc_all.append(tmp_train_acc)
            add_pseudo_val_acc_all.append(tmp_val_acc)
            add_pseudo_test_acc_all.append(tmp_test_acc)
            if tmp_val_acc > best_val_acc:
                best_val_acc = tmp_val_acc
                test_acc = tmp_test_acc
                best_epoch = epoch
        final_result.append(test_acc)
        # Label Density Detection
        data_edge_weight = torch.ones((1, data.edge_index[0].shape[0]))
        data_edge_weight = torch.squeeze(data_edge_weight, dim=0)
        adj_sparse = torch.sparse.FloatTensor(data.edge_index.cpu(), data_edge_weight,
                                              torch.Size([data.x.shape[0], data.x.shape[0]]))
        adj_dense = adj_sparse.to_dense()
        hop1_neighbors = torch.zeros([data.x.shape[0], data.x.shape[0]])
        hop2_neighbors = torch.zeros([data.x.shape[0], data.x.shape[0]])
        hop3_neighbors = torch.zeros([data.x.shape[0], data.x.shape[0]])
        G = nx.Graph()
        G.add_nodes_from(range(data.y.shape[0]))
        G.add_edges_from(data.edge_index.t().tolist())
        for i in range(data.y.shape[0]):
            neighbors = find123Nei(G, i)
            hop1_neighbors[i][neighbors[0]] = 1
            hop2_neighbors[i][neighbors[1]] = 1
            hop3_neighbors[i][neighbors[2]] = 1
        true_label_index = torch.where(data.train_mask > 0)[0]
        pseudo_label_index = torch.where(pseudo_label_select > 0)[0]
        all_label_index = list(set(true_label_index.tolist() + pseudo_label_index.tolist()))
        dict_hopk_neighbors = {1: hop1_neighbors, 2: hop2_neighbors, 3: hop3_neighbors}
        topk = 3
        label_rate = []
        predict_acc_rate = []
        for i in range(data.y.shape[0]):
            hopk_neighbors_index = [i]
            for j in range(1, topk + 1):
                hopj_neighbors = dict_hopk_neighbors[j]
                hopk_neighbors_index = hopk_neighbors_index + list(torch.where(hopj_neighbors[i] > 0)[0].numpy())
            all_label_neighbors_count = list(set(hopk_neighbors_index) & set(all_label_index))
            label_rate.append(len(all_label_neighbors_count) / len(hopk_neighbors_index))
        logger.info('Migration hop: {0}'.format(cur_hop+ 1))

        sparse_local_node = np.where(np.array(label_rate) < args.sparse_threshold)[0].tolist()
        density_local_node = np.where(np.array(label_rate) > args.dense_threshold)[0].tolist()
        adj_dense = adj_sparse.to_dense().to(device)
        adj_dense2 = torch.matmul(adj_dense, adj_dense)
        out = add_pseudo_model(data.x, data.edge_index, data.edge_weight).to(device)
        sparse_local_dict = {}
        density_local_dict = {}
        for sparse_local_node_item in sparse_local_node:
            local_graph, local_feature, sparse_local_node_neighbor = construct_local_graph(adj_dense, out,
                                                                                           sparse_local_node_item,
                                                                                           adj_dense2)
            if local_feature.shape[0] > 1:
                sparse_local_dict[sparse_local_node_item] = [local_graph, local_feature, sparse_local_node_neighbor]
            else:
                continue
        for density_local_node_item in density_local_node:
            local_graph, local_feature, density_local_node_neighbor = construct_local_graph(adj_dense, out,
                                                                                            density_local_node_item,
                                                                                            adj_dense2)
            if local_feature.shape[0] > 1:
                density_local_dict[density_local_node_item] = [local_graph, local_feature, density_local_node_neighbor]
            else:
                continue

        local_clustering_coefficient = []
        [local_clustering_coefficient.append(cal_local_clustering_coefficient(adj_dense,adj_dense + adj_dense2, i)) for i in range(data.y.shape[0])]
        local_clustering_coefficient = torch.tensor(local_clustering_coefficient)
        total_local_graph = len(sparse_local_dict.keys()) * len(density_local_dict.keys())
        total_count = 0
        logger.info(f'Number of sparse subgraph: {len(sparse_local_dict.keys())}. Number of dense subgraph: {len(density_local_dict.keys())}')
        density_neibor_node = torch.where(torch.sum(dict_hopk_neighbors[1][density_local_node, :], dim=0) > 0)[0].to(device)
        density_local_node = torch.tensor(density_local_node).to(device)
        density_node = torch.unique(torch.hstack((density_local_node, density_neibor_node)))
        node_id = density_node
        combine_density_fea_matirx = out[density_node, :]
        combine_density_graph = adj_dense[density_node, :][:, density_node].to(device)
        accurate_pseudo_num = []
        generate_pseudo_num = []
        save_acr = []
        match_num = 0
        for sparse_local_node_item in sparse_local_dict.keys():
            match_num = 0
            sparse_graph, sparse_feature, hop2_sparse_node = sparse_local_dict[
                sparse_local_node_item]
            cur_list = [len(hop2_sparse_node)]
            sparse_clustering_coefficient = local_clustering_coefficient[hop2_sparse_node].unsqueeze(0).t().to(device)
            density_clustering_coefficient = local_clustering_coefficient[node_id].unsqueeze(0).t().to(device)
            M = torch.matmul(sparse_feature,
                             combine_density_fea_matirx.T) + args.sparse_clustering_coefficient * torch.matmul(
                sparse_clustering_coefficient, density_clustering_coefficient.T)
            sparse_graph = sparse_graph.to(device)
            final_assm = spectral_graph_matching(sparse_graph, combine_density_graph, M,device)
            transform_matrix = final_assm
            hop2_density_node = node_id
            sparse_index, density_index = torch.where(transform_matrix > 0.1)
            cur_list.append(len(sparse_index))
            sparse_index = hop2_sparse_node[sparse_index]
            density_index = hop2_density_node[density_index]
            sparse_select = torch.bitwise_not(data.pseudo_train_mask[sparse_index])
            sparse_index = sparse_index[sparse_select]
            density_index = torch.tensor(density_index)
            density_index = density_index[sparse_select]
            density_select = train_pseudo_mask[density_index]
            density_index = density_index[density_select]
            sparse_index = sparse_index[density_select]
            data.pseudo_train_mask[sparse_index] = True
            pseudo_label[sparse_index] = pseudo_label[density_index]
            pseudo_label_select[sparse_index] = True

            if len(sparse_index) != 0:
                temp_accurate_pseudo_num = sum(pseudo_label[sparse_index] == data.y[sparse_index])
                temp_generate_pseudo_num = len(sparse_index)
                accurate_pseudo_num.append(temp_accurate_pseudo_num)
                generate_pseudo_num.append(temp_generate_pseudo_num)
                cur_list += [temp_generate_pseudo_num, temp_accurate_pseudo_num / temp_generate_pseudo_num]
            else:
                cur_list += [0, 0]
            save_acr.append(cur_list)

    data.pseudo_train_mask[data.test_mask] = False
    best_val_acc = 0
    add_pseudo_best_val_acc = add_pseudo_final_test_acc = best_epoch = 0
    add_pseudo_train_acc_all = []
    add_pseudo_val_acc_all = []
    add_pseudo_test_acc_all = []
    for epoch in range(1, args.epochs + 1):
        loss = add_pseudo_train(add_final_pseudo_model, add_final_pseudo_optimizer, pseudo_label, data)
        tmp_train_acc, tmp_val_acc, tmp_test_acc = test(add_final_pseudo_model, data)[0]
        add_pseudo_train_acc_all.append(tmp_train_acc)
        add_pseudo_val_acc_all.append(tmp_val_acc)
        add_pseudo_test_acc_all.append(tmp_test_acc)
        if tmp_val_acc > best_val_acc:
            best_val_acc = tmp_val_acc
            test_acc = tmp_test_acc
            best_epoch = epoch
    final_result.append(test_acc)
    match_result.append(test_acc)
    t_end = time.time()
    logger.info('GCN graph matching result:' + str(max(match_result)))


