# more efficient KAN version -e

import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from CDMs import CDM
# from ori_kan import *
from src.qkan.qkan import QKAN as KAN



def t_to_1(tensor, num_classes):
    return torch.nn.functional.one_hot(tensor, num_classes=num_classes).float()

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.stu_num = student_n
        self.know_num = knowledge_n
        self.exer_num = exer_n

        self.embed_dim = knowledge_n
        # self.prednet_input_len = self.knowledge_dim
        # self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # # prediction sub-net
        # self.student_emb = nn.Embedding(self.emb_num, self.embed_dim)
        # self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        # self.e_difficulty = nn.Embedding(self.exer_n, 1)


        e_size = [self.exer_num , self.embed_dim,1]
        self.kan_embed_e1 = KAN(e_size)
        self.kan_embed_e2 = KAN(e_size)
        self.kan_embed_e3 = KAN(e_size)
        self.kan_embed_e4 = KAN(e_size)
        self.kan_embed_e5 = KAN(e_size)

        k_size = [self.know_num , self.embed_dim,1]
        self.kan_embed_k1 = KAN(k_size)
        self.kan_embed_k2 = KAN(k_size)
        self.kan_embed_k3 = KAN(k_size)
        self.kan_embed_k4 = KAN(k_size)
        self.kan_embed_k5 = KAN(k_size)

        s_size = [self.stu_num,self.embed_dim,1]
        self.kan_embed_s1 = KAN(s_size)
        self.kan_embed_s2 = KAN(s_size)
        self.kan_embed_s3 = KAN(s_size)
        self.kan_embed_s4 = KAN(s_size)
        self.kan_embed_s5 = KAN(s_size)

        # self.my_kan = KAN(width=[self.prednet_input_len, 32, 1], grid=5, k=3, seed=0)
        self.kan_param = [15,1]
        self.fusion_kan = KAN(self.kan_param)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exe_id, know_id):
        # before prednet
        e_one_hot = t_to_1(exe_id, self.exer_num)
        e1_embed = self.kan_embed_e1(e_one_hot)
        e2_embed = self.kan_embed_e2(e_one_hot)
        e3_embed = self.kan_embed_e3(e_one_hot)
        e4_embed = self.kan_embed_e4(e_one_hot)
        e5_embed = self.kan_embed_e5(e_one_hot)

        e_all = torch.cat((e1_embed,e2_embed,e3_embed,e4_embed,e5_embed),dim=1)

        k1_embed = self.kan_embed_k1(know_id)
        k2_embed = self.kan_embed_k2(know_id)
        k3_embed = self.kan_embed_k3(know_id)
        k4_embed = self.kan_embed_k4(know_id)
        k5_embed = self.kan_embed_k5(know_id)
        k_all = torch.cat((k1_embed,k2_embed,k3_embed,k4_embed,k5_embed),dim=1)

        s_one_hot = t_to_1(stu_id, self.stu_num)
        s1_embed = self.kan_embed_s1(s_one_hot)
        s2_embed = self.kan_embed_s2(s_one_hot)
        s3_embed = self.kan_embed_s3(s_one_hot)
        s4_embed = self.kan_embed_s4(s_one_hot)
        s5_embed = self.kan_embed_s5(s_one_hot)
        s_all = torch.cat((s1_embed,s2_embed,s3_embed,s4_embed,s5_embed),dim=1)

        all = torch.cat((e_all,k_all,s_all),dim=1)
        output_1 = self.fusion_kan(all)
        output_1 = torch.sigmoid(output_1)

        return output_1.view(-1)


class KAN2CD_e(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n,dataset_name = ''):
        super(KAN2CD_e, self).__init__()
        self.dataset_name = dataset_name
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_auc = 0
        best_acc = 0
        best_epoch = -1

        out_str = f'{self.__class__.__name__}   kan_size:{str(self.ncdm_net.kan_param)}\nauc/acc from epoch 0 to epoch {epoch - 1}\n'

        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                y = y.double()
                pred = pred.double()
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            self.save(f"KAN2CD_{self.dataset_name}_{epoch_i}.snapshot")

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    best_acc = accuracy
                    best_epoch = epoch_i
                    self.save(f"best_KAN2CD_{self.dataset_name}.snapshot")
                print(f'\nbest auc{best_auc}')
                out_str = out_str + f"{auc:.6f} {accuracy:.6f}\n"
            else:
                exit(233)
        print(f"train finish best auc = {best_auc:.6f} best_acc{best_acc:.6f} at epoch {best_epoch}")
        print(out_str)

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        # self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)

