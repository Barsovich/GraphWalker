from collections import defaultdict

import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

MAX_NEIGHBOR_COUNT = 150
RELATION_EMBEDDING_LENGTH = 400


class DatasetMetaQAWalker(Dataset):
    def __init__(self, data, hops, word2ix, relations, entities, entity2idx, idx2relation, entity_paths, graph, mode):
        self.data = data
        self.hops = hops
        self.relations = relations
        self.entities = entities
        self.word_to_ix = {}
        self.entity2idx = entity2idx
        self.idx2relation = idx2relation
        self.entity_paths = entity_paths
        self.graph: nx.DiGraph = graph
        self.word_to_ix = word2ix
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def entity_ids_to_neighbor_indices(self, entities):
        result = []
        for i in range(len(entities) - 1):
            source_id = entities[i]
            target_id = entities[i + 1]
            neighbors = list(self.graph[source_id].keys())
            result.append(neighbors.index(target_id))
        return result

    def process_entity_path_into_neighbor_entity_ids(self, entities):
        # Initialize with -1
        target_vector = np.zeros((self.hops + 1, MAX_NEIGHBOR_COUNT), dtype=np.int64)
        for hop in range(len(entities)):
            source_id = entities[hop]
            neighbors = np.array(list(self.graph[source_id].keys()))
            if len(neighbors) > MAX_NEIGHBOR_COUNT:
                neighbors = neighbors[:MAX_NEIGHBOR_COUNT]
            target_vector[hop, :len(neighbors)] = neighbors
        return target_vector

    def process_entity_path_into_neighbor_relation_embeddings(self, entities):
        target_vector = np.zeros((self.hops + 1, MAX_NEIGHBOR_COUNT, RELATION_EMBEDDING_LENGTH), dtype=np.float32)
        for hop in range(len(entities)):
            source_id = entities[hop]
            for i, neighbor in enumerate(self.graph[source_id].values()):
                if i == MAX_NEIGHBOR_COUNT:
                    break
                target_vector[hop, i] = self.relations[self.idx2relation[neighbor['relation']]]
        return target_vector

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_ids = [self.word_to_ix.get(word, 0) for word in question_text.split()]
        head_id = self.entity2idx[data_point[0].strip()]
        answer_entities = list(map(lambda x: self.entity2idx[x], data_point[2]))
        entity_path = self.entity_paths[index][0]  # Use path of the first possible valid answer

        tail_path = self.entity_ids_to_neighbor_indices(entity_path)
        tail_path.append(MAX_NEIGHBOR_COUNT)  # Ad
        tail_onehot = torch.zeros(self.hops + 1, MAX_NEIGHBOR_COUNT + 1)  # d Stop signal at the end of the tail
        for i in range(len(tail_path)):
            tail_onehot[i, tail_path[i]] = 1
        path_entities = torch.from_numpy(self.process_entity_path_into_neighbor_entity_ids(entity_path))

        path_relation_embeddings = torch.from_numpy(self.process_entity_path_into_neighbor_relation_embeddings(entity_path))
        return question_ids, torch.tensor(entity_path), tail_onehot, path_entities, path_relation_embeddings, \
               torch.tensor(answer_entities), index


def _collate_fn(batch):
    sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
    longest_sample = sorted_seq_lengths[0]
    minibatch_size = len(batch)
    most_answer_alternatives = 1

    input_lengths = []
    p_heads = []
    answer_ids = []
    p_tail = []
    batch_path_entities = []
    batch_path_relation_embeddings = []
    indices = []
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)
    for x in range(minibatch_size):
        indices.append(sorted_seq[x][6])
        sample = sorted_seq[x][0]
        p_heads.append(sorted_seq[x][1])
        tail_onehot = sorted_seq[x][2]
        path_entities = sorted_seq[x][3]
        path_relation_embeddings = sorted_seq[x][4]
        answer_ids.append(sorted_seq[x][5])
        if sorted_seq[x][5].shape[0] > most_answer_alternatives:
            most_answer_alternatives = sorted_seq[x][5].shape[0]
        batch_path_entities.append(path_entities)
        batch_path_relation_embeddings.append(path_relation_embeddings)
        p_tail.append(tail_onehot)
        seq_len = len(sample)
        input_lengths.append(seq_len)
        sample = torch.tensor(sample, dtype=torch.long)
        sample = sample.view(sample.shape[0])
        inputs[x].narrow(0, 0, seq_len).copy_(sample)

    return inputs, \
           torch.tensor(input_lengths, dtype=torch.long), \
           nn.utils.rnn.pad_sequence(p_heads, batch_first=True, padding_value=0), \
           torch.stack(p_tail), \
           nn.utils.rnn.pad_sequence(batch_path_entities, batch_first=True, padding_value=0), \
           nn.utils.rnn.pad_sequence(batch_path_relation_embeddings, batch_first=True), \
           nn.utils.rnn.pad_sequence(answer_ids, batch_first=True, padding_value=-1), \
           torch.tensor(indices)


class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
