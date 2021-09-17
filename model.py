import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import init


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 400, graph_embedding_dim: int = 400,
                 word_embedding_dim: int = 400, bidirectional: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.bidirectional = bidirectional

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.question_encoder = nn.GRU(input_size=graph_embedding_dim, hidden_size=self.hidden_dim, num_layers=1,
                                       bidirectional=self.bidirectional, batch_first=True)

        self.apply(weight_init)

    def forward(self, question_word_ids, question_len):

        # shape (batch_size, max_sentence_len, word_embed_len)
        embeds = self.word_embeddings(question_word_ids)

        packed_output = pack_padded_sequence(embeds, lengths=question_len.cpu(), batch_first=True)

        # shape output: (batch_size, output_size)
        # shape _: (batch_size, hidden_size)
        output, _ = self.question_encoder(packed_output)

        output = pad_packed_sequence(output)[0]
        question_encoder_outputs = output.permute(1, 0, 2)

        return question_encoder_outputs


class InputEncoder(nn.Module):
    def __init__(self, pretrained_embeddings: torch.Tensor, hidden_dim: int = 400, graph_embedding_dim: int = 400,
                 freeze_graph_embeddings: bool = True, max_neighbor_count: int = 100, relation_embedding_len: int = 40,
                 entity_embedding_len: int = 200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.relation_embedding_len = relation_embedding_len
        self.entity_embedding_len = entity_embedding_len
        self.freeze_graph_embeddings = freeze_graph_embeddings
        self.pretrained_embeddings = pretrained_embeddings
        self.max_neighbor_count = max_neighbor_count

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings),
                                                      freeze=self.freeze_graph_embeddings)

        self.current_node_encoder = nn.Sequential(
            nn.Linear(self.graph_embedding_dim, self.entity_embedding_len),
        )

        self.relation_encoder = nn.Sequential(
            nn.Linear(self.graph_embedding_dim, self.relation_embedding_len),
        )

        self.entity_encoder = nn.Sequential(
            nn.Linear(self.graph_embedding_dim, self.entity_embedding_len),
        )

        self.apply(weight_init)

    def forward(self, current_node_id, neighbor_entity_ids, neighbour_relation_embeddings):
        batch_size = current_node_id.shape[0]

        current_node_embedding = self.embedding(current_node_id).unsqueeze(1)
        current_node_embedding = self.current_node_encoder(current_node_embedding).squeeze()

        neighbour_relation_embeddings = neighbour_relation_embeddings.unsqueeze(1)
        neighbour_relation_embeddings = self.relation_encoder(neighbour_relation_embeddings)
        neighbour_relation_embeddings = neighbour_relation_embeddings.view(batch_size, -1)

        neighbor_embeddings = self.embedding(neighbor_entity_ids)
        neighbor_embeddings[neighbor_entity_ids == 0] = 0
        neighbor_embeddings.unsqueeze(1)
        neighbor_embeddings = self.entity_encoder(neighbor_embeddings)
        neighbor_embeddings = neighbor_embeddings.view(batch_size, -1)

        return current_node_embedding, neighbour_relation_embeddings, neighbor_embeddings


class Attention(nn.Module):
    """
    Based on: www.adeveloperdiary.com/data-science/deep-learning/nlp/machine-translation-using-attention-with-pytorch/
    Performs Additive Attention by Louong et. al.
    """
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_hidden_vector = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.attn_scoring_fn = nn.Linear(decoder_hidden_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(1, momentum=0.95)
        self.prelu = nn.PReLU()

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        attn_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden).squeeze(2)
        attn_scoring_vector = attn_scoring_vector.permute(1, 0)
        attn_softmax = F.softmax(attn_scoring_vector, dim=1).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        w = torch.bmm(attn_softmax, encoder_outputs)
        w = w.squeeze()
        return w


class WalkingDecoder(nn.Module):
    def __init__(self, pretrained_embeddings: [torch.Tensor], max_neighbor_count: int = 100,
                 relation_embedding_len: int = 40, question_embedding_len: int = 800, entity_embedding_len: int = 200,
                 hidden_dim: int = 400, graph_embedding_dim: int = 400, freeze_graph_embeddings: bool = True):
        super().__init__()
        self.max_neighbor_count = max_neighbor_count
        self.relation_embedding_len = relation_embedding_len
        self.question_embedding_len = question_embedding_len
        self.entity_embedding_len = entity_embedding_len
        self.graph_embedding_dim = graph_embedding_dim
        self.hidden_dim = hidden_dim

        self.input_encoder = InputEncoder(pretrained_embeddings=pretrained_embeddings,
                                          hidden_dim=hidden_dim,
                                          graph_embedding_dim=graph_embedding_dim,
                                          freeze_graph_embeddings=freeze_graph_embeddings,
                                          max_neighbor_count=max_neighbor_count,
                                          relation_embedding_len=relation_embedding_len,
                                          entity_embedding_len=entity_embedding_len)

        self.attention = Attention(self.hidden_dim * 2, self.hidden_dim)

        self.decoder_input_dim = \
            self.entity_embedding_len + \
            self.question_embedding_len + \
            self.max_neighbor_count * (self.relation_embedding_len + self.entity_embedding_len)

        self.decoder_input_normalizer = nn.Sequential(
            nn.BatchNorm1d(self.decoder_input_dim),  # MOMENTUM: HYPER-PARAMETER
            nn.PReLU(),
            nn.Linear(self.decoder_input_dim, self.decoder_input_dim // 10),
            nn.BatchNorm1d(self.decoder_input_dim // 10),  # MOMENTUM: HYPER-PARAMETER
            nn.PReLU()
        )

        self.walking_decoder = nn.GRU(input_size=self.decoder_input_dim // 10, hidden_size=self.hidden_dim, num_layers=1,
                                      bidirectional=False, batch_first=True)
        self.hidden_to_pred = nn.Linear(self.hidden_dim, self.max_neighbor_count + 1)

        self.apply(weight_init)

    def forward(self, current_node_id, question_encoder_outputs, neighbor_entity_ids, neighbour_relation_embeddings,
                hidden):
        current_node_embedding, relation_embeddings, neighbor_embeddings = \
            self.input_encoder(current_node_id, neighbor_entity_ids, neighbour_relation_embeddings)

        attended_question_embedding = self.attention(hidden, question_encoder_outputs)

        decoder_input = torch.cat(
            [current_node_embedding,  # 400
             attended_question_embedding,  # 800
             relation_embeddings,  # 100 * 40
             neighbor_embeddings  # 100 * 200
             ], dim=1)
        decoder_input = self.decoder_input_normalizer(decoder_input)
        decoder_input = decoder_input.unsqueeze(dim=1)

        output, hidden = self.walking_decoder(decoder_input, hidden)
        output = output.squeeze()
        prediction = self.hidden_to_pred(output)
        return prediction, hidden
