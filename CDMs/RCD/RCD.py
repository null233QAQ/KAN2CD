

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from CDMs import CDM
from ori_kan import *

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n,embed_dim = 123):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.embed_dim = self.prednet_input_len
        self.student_emb = nn.Embedding(self.emb_num, self.embed_dim )
        self.k_difficulty = nn.Embedding(self.exer_n, self.embed_dim )
        self.e_difficulty = nn.Embedding(self.exer_n, self.embed_dim )




        self.kan_param = [2*self.prednet_input_len, 1]
        self.kan = KAN(self.kan_param,grid=15)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        stu_emb = self.student_emb(stu_id)
        exe_emb = self.e_difficulty(input_exercise)
        kc_emb  = self.k_difficulty(input_exercise)

        concat_C_S = torch.sigmoid(torch.cat((kc_emb, stu_emb), dim=-1))
        concat_E_C = torch.sigmoid(torch.cat((exe_emb, kc_emb), dim=-1))

        CS_minus_EC = concat_C_S - concat_E_C

        tmp = self.kan(CS_minus_EC)
        output_1 = torch.sigmoid(tmp)
        output_1 = torch.mean(output_1, dim=1)
        return output_1.view(-1)


class RCD(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n,data_set_name = ''):
        super(RCD, self).__init__()
        self.data_set_name = data_set_name
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        # self.ncdm_net.train()
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

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    best_acc = accuracy
                    best_epoch = epoch_i
                    self.save(f"{self.data_set_name}_rcd.snapshot")
                    print(f"best auc updated!: {best_auc:.6f}")

                out_str = out_str + f"{auc:.6f} {accuracy:.6f}\n"
            self.save(f"{self.data_set_name}_rcd_{epoch_i}.snapshot")
        out_str = out_str + f"\n\n\nbest auc: {best_auc:.6f},best_acc{best_acc:.6f},best_epoch{best_epoch}\n"

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
