import torch

def get_combined_matrix(local_dict):
    count = 0
    for local_node_item in local_dict.keys():
        graph, feature, hop2_sparsenode = local_dict[local_node_item]
        count += len(graph)
    cur_num = 0
    combine_matrix = torch.zeros((count,count))
    for local_node_item in local_dict.keys():
        graph, feature, hop2_sparsenode = local_dict[local_node_item]
        if cur_num == 0:
            combine_fea_matirx = feature
            node_id = hop2_sparsenode
        else:
            combine_fea_matirx  = torch.cat((combine_fea_matirx, feature), dim=0)
            node_id = torch.cat((node_id, hop2_sparsenode), dim=-1)
        combine_matrix[cur_num:cur_num+len(graph),cur_num:cur_num+len(graph)] = graph
        cur_num += len(graph)
    return combine_matrix,combine_fea_matirx,node_id


def spectral_graph_matching(adj1,adj2,M,device):
    d1 = adj1.shape[0]
    d2 = adj2.shape[0]
    feature_sim = M.T.reshape(-1)
    # edge_index
    x1 = torch.nonzero(adj1)[:,0]
    y1 = torch.nonzero(adj1)[:,1]
    x2 = torch.nonzero(adj2)[:,0]
    y2 = torch.nonzero(adj2)[:,1]

    # global_matrix index
    dic = {}
    count = 0
    for i in range(0, d2):
        for j in range(0, d1):
            dic[(j, i)] = count
            count += 1
    d = d1 * d2
    global_aff_matrix = torch.zeros((d, d)).to(device)

    # # # # off-diagonal elements (Edge similarity)
    for i in range(len(x1)):
        for j in range(len(x2)):
            global_aff_matrix[dic[(x1[i].item(), x2[j].item())], dic[(y1[i].item(), y2[j].item())]] = 1

    # diagonal elements (Node similarity and information propagation ability)
    for i in range(d):
        global_aff_matrix[i,i] = feature_sim[i]

    eig_val_cov, eig_vec_cov = torch.linalg.eigh(global_aff_matrix)
    max_eig_val = list(eig_val_cov).index(eig_val_cov.max())
    x_best = eig_vec_cov[:, max_eig_val]

    # Match from big to small
    x_best[x_best< 0] = 0
    ass_score = 0
    final_ass = torch.zeros((d1, d2))
    sum_ass = 0
    ass_matrix = x_best.reshape(d2, d1).T
    while not torch.all(ass_matrix == 0):
        x_res = torch.zeros(d1 * d2).to(device)
        best_ass_score = torch.max(ass_matrix)
        cur_ass = torch.where(ass_matrix == best_ass_score)
        x = cur_ass[0][0].item()
        y = cur_ass[1][0].item()
        ass_matrix[x, :] = 0
        ass_matrix[:, y] = 0
        x_res[dic[(x, y)]] = 1
        ass_score += feature_sim[dic[(x, y)]]
        sum_ass += 1
        if torch.matmul(torch.matmul(x_res, global_aff_matrix), x_res.T) >= 10:
            final_ass[cur_ass[0][0], cur_ass[1][0]] = 1
    return final_ass
