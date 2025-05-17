

import logging
import torch
import json
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

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)




        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point

        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        # output_1 = torch.sigmoid(self.ori_kan(input_x))

        return output_1.view(-1)


class NCDMori(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n,dataset_name=''):
        super(NCDMori, self).__init__()
        self.dataset_name = dataset_name
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        logging.info("traing... (lr={})".format(lr))
        self.ncdm_net = self.ncdm_net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)

        # 按照train函数的格式修改输出字符串
        kan_param = getattr(self.ncdm_net, 'kan_param', 'N/A')
        out_str = f'{self.__class__.__name__}   kan_size:{str(kan_param)}\nauc/acc from epoch 0 to epoch {epoch - 1}\n'

        best_auc = 0
        best_acc = 0
        best_epoch = -1

        # 初始化指标数据字典，用于保存JSON
        dataset_name = getattr(self, 'dataset_name', 'unknown_dataset')

        metrics_data = {
            "summary": out_str,
            "model_info": {
                "class_name": self.__class__.__name__,
                "kan_size": str(kan_param),
                "dataset": dataset_name
            },
            "best_model": {
                "epoch": -1,
                "auc": 0.0,
                "accuracy": 0.0
            },
            "epochs": []
        }

        for epoch_i in range(epoch):
            self.ncdm_net.train()
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

            avg_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" % (epoch_i, avg_loss))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, avg_loss))

            # 添加模型保存，与train函数保持一致的命名规则
            self.save(f"NCDMori_{dataset_name}_{epoch_i}.snapshot")

            # 初始化每个epoch的数据
            epoch_data = {
                "epoch": epoch_i,
                "training": {
                    "avg_loss": avg_loss
                },
                "validation": {}
            }

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)

                # 将验证指标添加到epoch数据中
                epoch_data["validation"]["auc"] = float(auc)
                epoch_data["validation"]["accuracy"] = float(accuracy)

                if auc > best_auc:
                    best_auc = auc
                    best_acc = accuracy
                    best_epoch = epoch_i

                    # 保存最佳模型，命名规则与train函数保持一致
                    self.save(f"best_NCDMori_{dataset_name}.snapshot")

                    # 标记为最佳epoch
                    epoch_data["is_best"] = True

                    # 更新顶层的最佳模型信息
                    metrics_data["best_model"] = {
                        "epoch": best_epoch,
                        "auc": float(best_auc),
                        "accuracy": float(best_acc)
                    }
                else:
                    epoch_data["is_best"] = False

                print("val acc auc [Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, accuracy))
                logging.info("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, accuracy))

                out_str = out_str + f"{auc:.6f} {accuracy:.6f}\n"
            else:
                print("Error: test_data is None")
                exit(233)

            # 更新顶层的摘要字符串
            metrics_data["summary"] = out_str

            # 将epoch数据添加到指标中
            metrics_data["epochs"].append(epoch_data)

            # 每个epoch后保存JSON（以防训练中断）
            with open(f"training_metrics_{dataset_name}.json", "w") as f:
                json.dump(metrics_data, f, indent=4)

        # 打印最终结果
        print(f"train finish best auc = {best_auc:.6f} best_acc{best_acc:.6f} at epoch {best_epoch}")
        print(out_str)
        print(f"Training metrics saved to training_metrics_{dataset_name}.json")

        ret = (f"\n\n{self.__class__.__name__} , "
               f"at epoch {best_epoch} best auc/acc: {best_auc:.6f}/{best_acc:.6f} \n")
        print(ret)
        return ret


    def train_ori(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_auc = 0
        best_acc = 0
        best_epoch = -1

        # out_str = f'{self.__class__.__name__}   kan_size:{str(self.ncdm_net.kan_param)}\nauc/acc from epoch 0 to epoch {epoch - 1}\n'
        out_str = ''
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
                if auc >best_auc:
                    best_auc = auc
                    best_acc = accuracy
                    best_epoch = epoch_i
                print(f'\nbest auc{best_auc}')
                out_str = out_str + f"{auc:.6f} {accuracy:.6f}\n"
            else:
                exit(233)
        print(f"train finish best auc = {best_auc:.6f} best_acc{best_acc:.6f} at epoch {best_epoch}")
        print(out_str)

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
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
