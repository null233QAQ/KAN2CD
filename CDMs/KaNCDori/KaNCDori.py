
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

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        # self.kan_param = [self.prednet_input_len, 1]
        # self.kan_grid = 15

        # self.ori_kan = qkan.KAN(self.kan_param, grid_size=self.kan_grid)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(input_exercise)
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # prednet
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        # output_1 = torch.sigmoid(self.ori_kan(input_x))
        return output_1.view(-1)


class KaNCDori(CDM):
    def __init__(self,dataset_name = '' ,**kwargs):
        super(KaNCDori, self).__init__()
        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf'
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], mf_type, kwargs['dim'])
        self.dataset_name = dataset_name # 默认值

    def train(self, train_set, valid_set, lr=0.002, device='cpu', epoch_n=15):
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # 按照第一份代码的格式修改输出字符串
        # 检查网络模型是否有kan_param属性，如果没有则使用空字符串
        kan_param = getattr(self.net, 'kan_param', 'N/A')
        out_str = f'{self.__class__.__name__}   kan_size:{str(kan_param)}\nauc/acc from epoch 0 to epoch {epoch_n - 1}\n'

        best_auc = 0
        best_acc = 0
        best_epoch = -1

        # 初始化指标数据字典，用于保存JSON
        # 确保dataset_name存在，如果不存在则使用默认值
        dataset_name = self.dataset_name

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

        for epoch_i in range(epoch_n):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_set, "Epoch %s" % epoch_i):
                batch_count += 1
                user_info, item_info, knowledge_emb, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb)
                pred = pred.double()
                y = y.double()
                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            avg_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" % (epoch_i, avg_loss))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, avg_loss))

            # 添加模型保存，命名规则与第一份代码保持一致
            # 确保dataset_name存在
            dataset_name = getattr(self, 'dataset_name', 'unknown_dataset')
            self.save(f"KaNCDori_{dataset_name}_{epoch_i}.snapshot")

            # 初始化每个epoch的数据
            epoch_data = {
                "epoch": epoch_i,
                "training": {
                    "avg_loss": avg_loss
                },
                "validation": {}
            }

            auc, acc = self.eval(valid_set, device)

            # 将验证指标添加到epoch数据中
            epoch_data["validation"]["auc"] = float(auc)
            epoch_data["validation"]["accuracy"] = float(acc)

            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_epoch = epoch_i

                # 保存最佳模型，命名规则与第一份代码保持一致
                dataset_name = getattr(self, 'dataset_name', 'unknown_dataset')
                self.save(f"best_KaNCDori_{dataset_name}.snapshot")

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

            print("val acc auc [Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            logging.info("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))

            out_str = out_str + f"{auc:.6f} {acc:.6f}\n"

            # 更新顶层的摘要字符串
            metrics_data["summary"] = out_str

            # 将epoch数据添加到指标中
            metrics_data["epochs"].append(epoch_data)

            # 每个epoch后保存JSON（以防训练中断）
            dataset_name = getattr(self, 'dataset_name', 'unknown_dataset')
            with open(f"training_metrics_{dataset_name}.json", "w") as f:
                json.dump(metrics_data, f, indent=4)

        # 打印最终结果
        print(f"train finish best auc = {best_auc:.6f} best_acc{best_acc:.6f} at epoch {best_epoch}")
        print(out_str)
        dataset_name = getattr(self, 'dataset_name', 'unknown_dataset')
        print(f"Training metrics saved to training_metrics_{dataset_name}.json")

        ret = (f"\n\n{self.__class__.__name__} , "
               f"at epoch {best_epoch} best auc/acc: {best_auc:.6f}/{best_acc:.6f} \n")
        print(ret)
        return ret

    def train_ori(self, train_set, valid_set, lr=0.002, device='cpu', epoch_n=15):
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        out_str = ""
        # out_str = f'{self.__class__.__name__}   kan_size:{str(self.net.kan_param)},\nauc/acc from epoch 0 to epoch {epoch_n - 1}\n'
        best_auc = 0
        best_acc = 0
        best_epoch = -1
        for epoch_i in range(epoch_n):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_set, "Epoch %s" % epoch_i):
                batch_count += 1
                user_info, item_info, knowledge_emb, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb)
                pred = pred.double()
                y = y.double()
                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            auc, acc = self.eval(valid_set, device)
            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_epoch = epoch_i
                # self.save(f"kancd_best_{self.net.kan_param}.snapshot")
            print("val acc auc [Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            logging.info("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))

            out_str = out_str + f" :{auc:.6f} {acc:.6f}\n"
        out_str = out_str + f"best auc: {best_auc:.6f}\n"

        ret = (f"\n\n{self.__class__.__name__} , "  # kan_size:{str(self.net.kan_param)} , kan_grid :{str(self.net.kan_grid)}\n"
               f"at epoch {best_epoch} best auc/acc: {best_auc:.6f}/{best_acc:.6f} \n")
        print(ret)
        return ret

    def eval(self, test_data, device="cpu"):
        logging.info('eval ... ')
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred = self.net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)
