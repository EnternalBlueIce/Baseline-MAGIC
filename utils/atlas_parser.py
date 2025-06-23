import argparse
import os
import json
import pickle as pkl
import networkx as nx

node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0

metadata = {
    'M1': {
        'train': ['graph_M1-CVE-2015-5122_windows_h1.dot'],
        'test': ['graph_M1-CVE-2015-5122_windows_h2.dot']
    }
}


def read_single_graph(dataset, malicious, filename, test=False):
    global node_type_cnt, edge_type_cnt
    g = nx.DiGraph()
    path = f'../atlas_data/{dataset}/{filename}.txt'
    print(f'Loading graph from {path}...')

    with open(path, 'r') as f:
        lines = []
        for line in f:
            src, src_type, dst, dst_type, edge_type, ts = line.strip().split('\t')
            ts = int(ts)

            if not test:
                if src in malicious or dst in malicious:
                    if src in malicious and src_type != 'MemoryObject':
                        continue
                    if dst in malicious and dst_type != 'MemoryObject':
                        continue

            for t in [src_type, dst_type]:
                if t not in node_type_dict:
                    node_type_dict[t] = node_type_cnt
                    node_type_cnt += 1
            if edge_type not in edge_type_dict:
                edge_type_dict[edge_type] = edge_type_cnt
                edge_type_cnt += 1

            if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
                lines.append([dst, src, dst_type, src_type, edge_type, ts])
            else:
                lines.append([src, dst, src_type, dst_type, edge_type, ts])

    lines.sort(key=lambda x: x[5])
    node_map = {}
    node_cnt = 0
    for src, dst, src_type, dst_type, edge_type, _ in lines:
        src_id = node_map.setdefault(src, node_cnt)
        if src_id == node_cnt:
            g.add_node(src_id, type=node_type_dict[src_type])
            node_cnt += 1

        dst_id = node_map.setdefault(dst, node_cnt)
        if dst_id == node_cnt:
            g.add_node(dst_id, type=node_type_dict[dst_type])
            node_cnt += 1

        if not g.has_edge(src_id, dst_id):
            g.add_edge(src_id, dst_id, type=edge_type_dict[edge_type])
    return node_map, g


def read_graphs(dataset):
    # 读取标注的恶意实体
    with open(f'../atlas_data/{dataset}/label.txt', 'r') as f:
        malicious_entities = {line.strip() for line in f}

    train_gs = []
    for file in metadata[dataset]['train']:
        _, g = read_single_graph(dataset, malicious_entities, file, test=False)
        train_gs.append(g)

    test_gs = []
    test_node_map = {}
    node_offset = 0
    for file in metadata[dataset]['test']:
        node_map, g = read_single_graph(dataset, malicious_entities, file, test=True)
        test_gs.append(g)
        for k, v in node_map.items():
            if k not in test_node_map:
                test_node_map[k] = v + node_offset
        node_offset += g.number_of_nodes()

    # 处理恶意实体 -> final node ID & readable name
    final_malicious = []
    malicious_names = []
    id_nodetype_map = {}
    id_nodename_map = {}

    # 加载已有名字和类型（如果存在）
    names_path = f'../atlas_data/{dataset}/names.json'
    types_path = f'../atlas_data/{dataset}/types.json'
    if os.path.exists(names_path) and os.path.exists(types_path):
        id_nodename_map = json.load(open(names_path))
        id_nodetype_map = json.load(open(types_path))

    with open(f'../atlas_data/{dataset}/malicious_names.txt', 'w', encoding='utf-8') as f:
        for uuid in malicious_entities:
            if uuid in test_node_map:
                if uuid in id_nodetype_map and id_nodetype_map[uuid] in ['MemoryObject', 'UnnamedPipeObject']:
                    continue
                final_malicious.append(test_node_map[uuid])
                name = id_nodename_map.get(uuid, uuid)
                malicious_names.append(name)
                f.write(f'{uuid}\t{name}\n')

    # 保存所有数据为pkl
    pkl.dump((final_malicious, malicious_names), open(f'../atlas_data/{dataset}/malicious.pkl', 'wb'))
    pkl.dump([nx.node_link_data(g) for g in train_gs], open(f'../atlas_data/{dataset}/train.pkl', 'wb'))
    pkl.dump([nx.node_link_data(g) for g in test_gs], open(f'../atlas_data/{dataset}/test.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="M1")
    args = parser.parse_args()
    read_graphs(args.dataset)
