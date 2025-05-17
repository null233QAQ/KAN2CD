
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
from ori_kan import KAN

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

        # self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        # self.drop_1 = nn.Dropout(p=0.5)
        # self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        # self.drop_2 = nn.Dropout(p=0.5)
        # self.prednet_full3 = PosLinear(self.prednet_len2, 1)


        # self.ori_kan = KAN(width=[self.prednet_input_len, 32, 1], grid=5, k=3, seed=0)
        self.kan_param = [self.prednet_input_len,1]
        self.kan = KAN(self.kan_param,grid=5)
        #self.ori_kan.plot()
        #self.ori_kan.cuda()

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

        # input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        # input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        # output_1 = torch.sigmoid(self.prednet_full3(input_x))

        output_1 = torch.sigmoid(self.kan(input_x))

        return output_1.view(-1)


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n,data_set_name =''):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)
        self.data_set_name = data_set_name

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        best_auc = 0
        best_acc = 0
        best_epoch = -1

        # Create model summary string
        out_str = f'{self.__class__.__name__}   kan_size:{str(self.ncdm_net.kan_param)}\nauc/acc from epoch 0 to epoch {epoch - 1}\n'

        # Initialize JSON training log using the preferred format
        metrics_data = {
            "summary": out_str,
            "model_info": {
                "class_name": self.__class__.__name__,
                "kan_size": str(self.ncdm_net.kan_param),
                "dataset": self.data_set_name
            },
            "best_model": {
                "epoch": best_epoch,
                "auc": best_auc,
                "accuracy": best_acc
            },
            "epochs": []
        }

        log_file_path = f"{self.data_set_name}_metrics.json"

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

            avg_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" % (epoch_i, avg_loss))

            # Initialize epoch data entry
            epoch_data = {
                "epoch": epoch_i,
                "avg_loss": avg_loss,
                "is_best_model": False
            }

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)

                # Update epoch data with validation metrics
                epoch_data["auc"] = float(auc)
                epoch_data["accuracy"] = float(accuracy)

                # Check if this is the best model so far
                if auc > best_auc:
                    best_auc = auc
                    best_acc = accuracy
                    best_epoch = epoch_i
                    epoch_data["is_best_model"] = True

                    # Save the best model snapshot
                    self.save(f"{self.data_set_name}_ncdm_best.snapshot")

                    # Update best model info in the JSON
                    metrics_data["best_model"] = {
                        "epoch": best_epoch,
                        "auc": float(best_auc),
                        "accuracy": float(best_acc)
                    }

                print(f'\nbest auc{best_auc}')
                out_str = out_str + f"{auc:.6f} {accuracy:.6f}\n"
            else:
                exit(233)

            # Save regular model snapshot for this epoch
            self.save(f"{self.data_set_name}_ncdm_{epoch_i}.snapshot")

            # Add epoch data to the log
            metrics_data["epochs"].append(epoch_data)

            # Update the summary output string in the log
            metrics_data["summary"] = out_str

            # Write updated log to file after each epoch to prevent data loss
            with open(log_file_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)

        # Print final training results
        print(f"train finish best auc = {best_auc:.6f} best_acc{best_acc:.6f} at epoch {best_epoch}")
        print(out_str)

        return best_auc, best_acc, best_epoch

    def train_ori(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
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
                if auc >best_auc:
                    best_auc = auc
                    best_acc = accuracy
                    best_epoch = epoch_i
                    self.save(f"{self.data_set_name}_ncdm.snapshot")
                print(f'\nbest auc{best_auc}')
                out_str = out_str + f"{auc:.6f} {accuracy:.6f}\n"
            else:
                exit(233)
            self.save(f"{self.data_set_name}_ncdm_{epoch_i}.snapshot")
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
