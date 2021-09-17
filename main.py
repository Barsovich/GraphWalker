import argparse
import os

import networkx as nx
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from dataloader import DatasetMetaQAWalker, DataLoaderMetaQA
from model import QuestionEncoder, WalkingDecoder
from torch.utils.tensorboard import SummaryWriter

def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key.strip()] = i
        idx2entity[i] = key.strip()
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
        sent = d[1]
        for word in sent.split():
            if word not in word_to_ix:
                idx2word[len(word_to_ix)] = word
                word_to_ix[word] = len(word_to_ix)

        length = len(sent.split())
        if length > maxLength:
            maxLength = length

    return word_to_ix, idx2word, maxLength


def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}

    f = open(entity_dict, 'r')
    for line in f:
        line = line.strip().split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()

    f = open(relation_dict, 'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e, r


def get_shortest_entity_relation_paths(filename):
    entity_path_dict = {}
    relation_path_dict = {}
    with open(filename) as f:
        for line in f:
            line_content = line.strip().split(':')
            q_idx = int(line_content[0])
            tail = line_content[1]
            alternatives = tail.split('|')
            entity_paths = []
            relation_paths = []
            for alt in alternatives:
                paths = alt.split(';')
                entity_paths_str = paths[0].split(',')
                relation_paths_str = paths[1].split(',')
                entity_paths_int = list(map(int, entity_paths_str))
                entity_paths.append(entity_paths_int)
                if len(relation_paths_str) == 1 and relation_paths_str[0] == '':
                    relation_paths.append([])
                else:
                    relation_paths.append(list(map(int, relation_paths_str)))
            entity_path_dict[q_idx] = entity_paths
            relation_path_dict[q_idx] = relation_paths
    return entity_path_dict, relation_path_dict


def make_graph(entity2id_map, relation2id_map, root):
    triples = []
    for source in ['train.txt', 'valid.txt', 'test.txt']:
        with open(root + source) as f:
            for line in f:
                line = line.strip().split('\t')
                line[0] = entity2id_map[line[0]]
                line[1] = relation2id_map[line[1]]
                line[2] = entity2id_map[line[2]]
                triples.append(line)
    G = nx.DiGraph()
    for t in triples:
        e1 = t[0]
        e2 = t[2]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2, relation=t[1])
        G.add_edge(e2, e1, relation=int(t[1]) + 1)
    return G


def find_neighbor_entity_ids(args, graph: nx.DiGraph, source_id):
    neighbors = list(graph[source_id].keys())
    if len(neighbors) > args.max_neighbor_count:
        neighbors = neighbors[:args.max_neighbor_count]
    for i in range(len(neighbors), args.max_neighbor_count):
        neighbors.append(0)  # Zero-padding to have a length of MAX_NEIGHBOR_COUNT
    return neighbors


def find_neighbor_relation_embeddings(args, graph: nx.DiGraph, source_id, idx2relation, relations):
    target_vector = np.zeros((args.max_neighbor_count, args.graph_embedding_dim))
    for i, neighbor in enumerate(graph[source_id].values()):
        if i == args.max_neighbor_count:
            break
        target_vector[i] = relations[idx2relation[neighbor['relation']]]
    return target_vector


def validate_walker(args, device, encoder, decoder, loader, graph, idx2relation, idx2entity, valid_data, relations,
                    epoch):
    encoder.train()
    decoder.train()
    total_correct = 0
    total_datasize = 0
    answer_output = open(f'answers/best_answers_{str(epoch).zfill(3)}.txt', 'w')
    output_counter = 0

    for i_batch, a in enumerate(loader):
        question_word_ids = a[0].to(device)
        question_len = a[1].to(device)
        batch_current_node_id = a[2].to(device)
        positive_tail = a[3].to(device)  # (batch_size, max_hop_count + 1, max_neighbor_count + 1)
        neighbor_entity_ids = a[4].to(device)  # (batch_size, max_hop_count, max_neighbor_count)
        neighbour_relation_embeddings = a[5].to(device)  # (batch_size, max_hop_count, max_neighbor_count, relation_embedding_len)
        batch_ground_truths = a[6].to(device)  # (batch_size, answer_alternative_count (different for each batch))
        question_ids = a[7]
        batch_size = batch_current_node_id.shape[0]
        answers = torch.ones(args.batch_size).to(device) * -1

        # Initial values for the forward function
        batch_current_node_id = batch_current_node_id[:, 0]
        neighbor_entity_ids = neighbor_entity_ids[:, 0]  # (batch_size, max_neighbor_count)
        neighbour_relation_embeddings = neighbour_relation_embeddings[:, 0]  # (batch_size, max_neighbor_count, relation_emb_len)
        hidden = torch.zeros((1, batch_size, args.hidden_dim), device=device)
        question_encoder_outputs = encoder(question_word_ids, question_len)
        for hop in range(args.hops + 1):
            prediction, hidden = decoder(batch_current_node_id, question_encoder_outputs,
                                         neighbor_entity_ids, neighbour_relation_embeddings, hidden)
            decisions = torch.argmax(prediction, dim=1)  # 50
            neighbor_entity_ids = []
            neighbour_relation_embeddings = []
            batch_current_node_id_list = []
            for index in range(batch_size):
                neighbors = list(graph[batch_current_node_id[index].item()].keys())
                # If the first option is infeasable
                if decisions[index] != args.max_neighbor_count and decisions[index].item() >= len(neighbors):
                    # if decision is to stop
                    if torch.max(prediction[index][:len(neighbors)], dim=0)[0] < prediction[index, args.max_neighbor_count]:
                        decisions[index] = args.max_neighbor_count
                    else:
                        decisions[index] = torch.argmax(prediction[index][:len(neighbors)], dim=0).item()

                # Consider STOP Signal below
                if decisions[index] == args.max_neighbor_count:
                    if answers[index] == -1:
                        answers[index] = batch_current_node_id[index]
                    sample_neighbor_entity_ids = torch.zeros(args.max_neighbor_count, device=device, dtype=torch.long)
                    sample_neighbour_relation_embeddings = torch.zeros(args.max_neighbor_count, args.graph_embedding_dim, device=device)
                    neighbor_entity_ids.append(sample_neighbor_entity_ids)
                    neighbour_relation_embeddings.append(sample_neighbour_relation_embeddings)
                    batch_current_node_id_list.append(args.max_neighbor_count)
                    continue

                decision = decisions[index]
                current_node_id = neighbors[decision]
                batch_current_node_id_list.append(current_node_id)
                sample_neighbor_entity_ids = find_neighbor_entity_ids(args=args, graph=graph,
                                                                      source_id=current_node_id)
                sample_neighbour_relation_embeddings = find_neighbor_relation_embeddings(args=args, graph=graph,
                                                                                         source_id=current_node_id,
                                                                                         idx2relation=idx2relation,
                                                                                         relations=relations)
                sample_neighbor_entity_ids = torch.tensor(sample_neighbor_entity_ids, device=device)
                sample_neighbour_relation_embeddings = torch.tensor(sample_neighbour_relation_embeddings,
                                                                    dtype=torch.float, device=device)
                neighbor_entity_ids.append(sample_neighbor_entity_ids)
                neighbour_relation_embeddings.append(sample_neighbour_relation_embeddings)
                # In last hop, add the prediction to answers
                if hop == args.hops:
                    if answers[index] == -1:
                        answers[index] = current_node_id
            neighbor_entity_ids = torch.stack(neighbor_entity_ids, dim=0)
            neighbour_relation_embeddings = torch.stack(neighbour_relation_embeddings, dim=0)
            batch_current_node_id = torch.tensor(batch_current_node_id_list, device=device)

        for i in range(batch_size):
            total_datasize += 1
            end_index = batch_ground_truths[i].shape[0]
            index_tensor = (batch_ground_truths[i] == -1).nonzero(as_tuple=True)[0]
            if index_tensor.shape[0] != 0:
                end_index = index_tensor[0].item()
            if answers[i] in batch_ground_truths[i][0:end_index]:
                total_correct += 1
            answer_output.write(f'Question {str(output_counter).zfill(3)}:\n'
                                f'\tText: {valid_data[question_ids[i].item()][1]},\n'
                                f'\tSource Entity: {valid_data[question_ids[i].item()][0]},\n'
                                f'\tPrediction: {idx2entity[answers[i].item()]},\n'
                                f'\tTruth:{list(map(lambda x: idx2entity[x], batch_ground_truths[i][0:end_index].tolist()))}\n')
            output_counter += 1
    answer_output.close()
    accuracy = total_correct / total_datasize
    return accuracy


def train(args, data_path, valid_path, entity_path, relation_path, entity_dict, relation_dict, bfs_result_path,
          root_path):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    # Tensorboard Visualization
    writer = SummaryWriter(args.tensorboard_log_folder)
    # Load Data
    entities = np.load(entity_path)
    relations = np.load(relation_path)
    e, r = preprocess_entities_relations(entity_dict, relation_dict, entities, relations)
    # e, r : Dict, entity/relation name -> embedding
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    relation2idx, idx2relation, _ = prepare_embeddings(r)
    graph = make_graph(entity2idx, relation2idx, root=root_path)

    # Training data
    bfs_result_entities, bfs_result_relations = get_shortest_entity_relation_paths(filename=bfs_result_path)
    data = process_text_file(data_path, split=False)  # [head, question, tail]
    word2ix, idx2word, max_len = get_vocab(data)
    # Training dataset and data loader
    dataset = DatasetMetaQAWalker(data=data,
                                  hops=args.hops,
                                  word2ix=word2ix,
                                  relations=r,
                                  entities=e,
                                  entity2idx=entity2idx,
                                  idx2relation=idx2relation,
                                  entity_paths=bfs_result_entities,
                                  graph=graph,
                                  mode='train')
    data_loader = DataLoaderMetaQA(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   drop_last=True)

    # Validation data
    if not args.overfit:
        bfs_dev_path = bfs_result_path.replace('train.txt', 'test.txt')
    else:
        bfs_dev_path = bfs_result_path
    bfs_dev_entities, bfs_dev_relations = get_shortest_entity_relation_paths(filename=bfs_dev_path)
    valid_data = process_text_file(valid_path, split=False)
    # Validation dataset and data loader
    dataset_dev = DatasetMetaQAWalker(data=valid_data,
                                      hops=args.hops,
                                      word2ix=word2ix,
                                      relations=r,
                                      entities=e,
                                      entity2idx=entity2idx,
                                      idx2relation=idx2relation,
                                      entity_paths=bfs_dev_entities,
                                      graph=graph,
                                      mode='dev')
    data_loader_dev = DataLoaderMetaQA(dataset_dev, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    # Models
    encoder = QuestionEncoder(vocab_size=len(word2ix),
                              hidden_dim=args.hidden_dim,
                              graph_embedding_dim=args.graph_embedding_dim,
                              word_embedding_dim=args.word_embedding_dim,
                              bidirectional=args.bidirectional)
    decoder = WalkingDecoder(pretrained_embeddings=embedding_matrix,
                             hidden_dim=args.hidden_dim,
                             graph_embedding_dim=args.graph_embedding_dim,
                             freeze_graph_embeddings=args.freeze_graph_embeddings,
                             max_neighbor_count=args.max_neighbor_count,
                             relation_embedding_len=args.relation_embedding_len,
                             question_embedding_len=args.question_embedding_len,
                             entity_embedding_len=args.entity_embedding_len)

    encoder.load_state_dict(torch.load('../../checkpoints_full-150-128-e4-linear-50-50-reduce-before-gru-10x/MetaQA/best_3_hop_encoder.pt'))
    decoder.load_state_dict(torch.load('../../checkpoints_full-150-128-e4-linear-50-50-reduce-before-gru-10x/MetaQA/best_3_hop_decoder.pt'))
    encoder.to(device)
    decoder.to(device)

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ], lr=args.lr, weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, args.decay)
    loss_criterion = torch.nn.CrossEntropyLoss()

    # Start training
    best_score = -float("inf")
    best_encoder = encoder.state_dict()
    best_decoder = decoder.state_dict()
    no_update = 0
    for epoch in range(args.nb_epochs):
        phases = []
        for i in range(args.validate_every):
            # phases.append('train')
            pass
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                encoder.train()
                decoder.train()
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                num_correct = 0
                total = 0
                batch_count = len(loader)
                for i_batch, a in enumerate(loader):
                    optimizer.zero_grad()
                    question_word_ids = a[0].to(device)
                    question_len = a[1].to(device)
                    current_node_id = a[2].to(device)
                    positive_tail = a[3].to(device)  # (batch_size, max_hop_count + 1, max_neighbor_count + 1)
                    neighbor_entity_ids = a[4].to(device)  # (batch_size, max_hop_count, max_neighbor_count + 1)
                    neighbour_relation_embeddings = a[5].to(
                        device)  # (batch_size, max_hop_count, max_neighbor_count + 1, relation_embedding_size)
                    # relation embeddings (batch_size, max_hop_count, max_neighbor_count + 1, relation_embedding_size)
                    batch_size = current_node_id.shape[0]
                    hidden = torch.zeros((1, batch_size, args.hidden_dim), device=device)
                    loss = 0
                    question_encoder_outputs = encoder(question_word_ids, question_len)
                    is_correct_decision = []
                    for i in range(args.hops + 1):
                        prediction, hidden = decoder(current_node_id[:, i], question_encoder_outputs,
                                                     neighbor_entity_ids[:, i], neighbour_relation_embeddings[:, i],
                                                     hidden)
                        loss += loss_criterion(prediction, positive_tail[:, i].argmax(dim=1))
                        with torch.no_grad():
                            is_correct_decision.append(prediction.argmax(dim=1) == positive_tail[:, i].argmax(dim=1))
                    with torch.no_grad():
                        is_matching = is_correct_decision[0].detach().cpu()
                        for i in range(args.hops):
                            is_matching *= is_correct_decision[i + 1].detach().cpu()
                        num_correct_one = is_matching.sum().item()
                        num_correct += num_correct_one
                        total += len(is_matching)
                        writer.add_scalar("Accuracy_batch/train", num_correct_one / len(is_matching),
                                          epoch * batch_count + i_batch)

                    writer.add_scalar("Loss/train", loss, epoch * batch_count + i_batch) # Tensorboard
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, args.nb_epochs))
                    loader.update()
                writer.add_scalar("Accuracy_epoch/train", num_correct / total, epoch)
                print(f'Training accuracy for this epoch is {num_correct / total}')
                scheduler.step()
            elif phase == 'valid':
                loader_dev = tqdm(data_loader_dev, total=len(data_loader_dev), unit="batches")
                eps = 0.0001
                with torch.no_grad():
                    score = validate_walker(args=args, encoder=encoder, decoder=decoder, device=device, loader=loader_dev,
                                            graph=graph, idx2relation=idx2relation, relations=r, valid_data=valid_data,
                                            idx2entity=idx2entity, epoch=epoch)
                writer.add_scalar("Accuracy", score, epoch) # Tensorboard
                checkpoint_path = '../../checkpoints_full-150-128-e4-linear-50-50-reduce-before-gru-10x/MetaQA/'
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_encoder = encoder.state_dict()
                    best_decoder = decoder.state_dict()
                    print(str(args.hops) + " hop Validation accuracy increased from previous epoch", score)
                    checkpoint_file_name = checkpoint_path + "best_" + str(args.hops) + "_hop"
                    print('Saving checkpoint to ', checkpoint_file_name)
                    torch.save(best_encoder, checkpoint_file_name + "_encoder.pt")
                    torch.save(best_decoder, checkpoint_file_name + "_decoder.pt")
                elif (score < best_score + eps) and (no_update < args.patience):
                    no_update += 1
                    print("Validation has not increased. Last score is %f, best score was %f, %d more epoch to check" % (
                        score, best_score, args.patience - no_update))
                elif no_update == args.patience:
                    print("Model has exceeded patience. Saving best model and exiting")
                    torch.save(best_encoder, checkpoint_path + "last_score_encoder_model.pt")
                    torch.save(best_decoder, checkpoint_path + "last_score_decoder_model.pt")
                    return
                if epoch == args.nb_epochs - 1:
                    print("Final Epoch has reached. Stopping and saving model.")
                    torch.save(best_encoder, checkpoint_path + "last_score_encoder_model.pt")
                    torch.save(best_decoder, checkpoint_path + "last_score_decoder_model.pt")
                    return
    writer.close()


def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1 + 'NE' + question_2
        ans = data_line[1].split('|')
        data_array.append([head, question.strip(), ans])
    if split == False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data


def data_generator(data, word2ix, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1].strip().split(' ')
        encoded_question = [word2ix[word.strip()] for word in question]
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question, dtype=torch.long), ans, torch.tensor(
            len(encoded_question), dtype=torch.long), data_sample[1]


def main(args):
    data_root = '../../data'
    hops = str(args.hops)
    if hops in ['1', '2', '3']:
        hops = hops + 'hop'

    data_path = os.path.join(data_root, 'QA_data/MetaQA/qa_train_' + hops + '.txt')
    valid_data_path = os.path.join(data_root, 'QA_data/MetaQA/qa_test_' + hops + '.txt')
    test_data_path = os.path.join(data_root, 'QA_data/MetaQA/qa_test_' + hops + '.txt')
    if args.overfit:
        data_path = os.path.join(data_root, 'QA_data/MetaQA/qa_train_' + hops + '_overfit.txt')
        valid_data_path = os.path.join(data_root, 'QA_data/MetaQA/qa_train_' + hops + '_overfit.txt')
    print(f'Train file is {data_path}, Validation file is {valid_data_path}')

    model_name = args.model
    kg_type = args.kg_type
    print('KG type is', kg_type)
    embedding_folder = '../../pretrained_models/embeddings/' + model_name + '_MetaQA_' + kg_type

    bfs_result_path = os.path.join(data_root, f'MetaQA/bfs_{hops}_train.txt')
    entity_embedding_path = embedding_folder + '/E.npy'
    relation_embedding_path = embedding_folder + '/R.npy'
    entity_dict = embedding_folder + '/entities.dict'
    relation_dict = embedding_folder + '/relations.dict'

    if args.mode == 'train':
        train(args=args,
              data_path=data_path,
              valid_path=valid_data_path,
              entity_path=entity_embedding_path,
              relation_path=relation_embedding_path,
              entity_dict=entity_dict,
              relation_dict=relation_dict,
              bfs_result_path=bfs_result_path,
              root_path=os.path.join(data_root, 'MetaQA/'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hops', type=int, default=1)
    parser.add_argument('--ls', type=float, default=0.0)
    parser.add_argument('--validate_every', type=int, default=5)
    parser.add_argument('--model', type=str, default='Rotat3')
    parser.add_argument('--kg_type', type=str, default='full')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--tensorboard_log_folder', type=str, default='runs/full-200-128-weight-decay-e3')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--entdrop', type=float, default=0.0)
    parser.add_argument('--reldrop', type=float, default=0.0)
    parser.add_argument('--scoredrop', type=float, default=0.0)
    parser.add_argument('--l3_reg', type=float, default=0.0)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--shuffle_data', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--nb_epochs', type=int, default=90)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--neg_batch_size', type=int, default=128)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--overfit', type=bool, default=False)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_neighbor_count', type=int, default=150)
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--graph_embedding_dim', type=int, default=400)
    parser.add_argument('--word_embedding_dim', type=int, default=400)
    parser.add_argument('--freeze_graph_embeddings', type=bool, default=True)
    parser.add_argument('--relation_embedding_len', type=int, default=10)
    parser.add_argument('--question_embedding_len', type=int, default=800)
    parser.add_argument('--entity_embedding_len', type=int, default=50)
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    main(args)
