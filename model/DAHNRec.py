import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface

device = torch.device("cuda:0")

class DAHNRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DAHNRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DAHNRec'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.layer_cl = int(args['-l*'])
        self.alpha = float(args['-alpha'])
        self.dropout_rate = float(args['-dropout_rate'])
        self.model = DAHNRec_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl, self.dropout_rate)

    def train(self):
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                rec_loss = self.bbpr_loss(user_emb, pos_item_emb, neg_item_emb,self.alpha)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb,
                                                          cl_item_emb)
                batch_loss = rec_loss + self.l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def bbpr_loss(self, user_emb, pos_item_emb, neg_item_emb, alpha):

        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        pos_prob = torch.sigmoid(pos_score)
        neg_prob = torch.sigmoid(neg_score) 
        loss = -torch.log(alpha + pos_prob * (1 - neg_prob)) 
        return torch.mean(loss)

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(device)
        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss * reg

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class Distribution_aware_Noise(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(Distribution_aware_Noise, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return F.normalize(self.layer(x), dim=-1)


class DAHNRec_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl, dropout_rate):
        super(DAHNRec_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.dropout_rate = dropout_rate
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)
        self.noise_generator_user = Distribution_aware_Noise(self.emb_size, self.emb_size, self.dropout_rate).to(device)
        self.noise_generator_item = Distribution_aware_Noise(self.emb_size, self.emb_size, self.dropout_rate).to(device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size).to(device))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size).to(device))), })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if perturbed:
                noise_user = self.noise_generator_user(ego_embeddings[:self.data.user_num])
                noise_item = self.noise_generator_item(ego_embeddings[self.data.user_num:])
                noise = torch.cat([noise_user, noise_item], 0)
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * noise * self.eps

            all_embeddings.append(ego_embeddings)
            if k == self.layer_cl - 1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings,[self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl,[self.data.user_num, self.data.item_num])

        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings