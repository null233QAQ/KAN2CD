
import logging
from CDMs import KAN2CD_e_pykan
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# 普通embed版本
def main(dataset_name = ''):
    data_path = '../../data/' + dataset_name + '/'
    train_data = pd.read_csv(data_path + "train.csv")
    valid_data = pd.read_csv(data_path + "valid.csv")
    test_data = pd.read_csv(data_path + "test.csv")
    df_item = pd.read_csv(data_path + "item.csv")

    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    batch_size = 128
    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
    knowledge_n = np.max(list(knowledge_set))


    def transform(user, item, item2knowledge, score, batch_size):
        knowledge_emb = torch.zeros((len(item), knowledge_n))
        for idx in range(len(item)):
            knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

        data_set = TensorDataset(
            torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
            torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
            knowledge_emb,
            torch.tensor(score, dtype=torch.float32)
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=True)


    train_set, valid_set, test_set = [
        transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
        for data in [train_data, valid_data, test_data]
    ]

    logging.getLogger().setLevel(logging.INFO)
    cdm = KAN2CD_e_pykan(knowledge_n, item_n, user_n, dataset_name=dataset_name)
    cdm.train(train_set, valid_set, epoch=18, device="cpu")
    # cdm.save(f"{dataset_name}_KAN2CD.snapshot")

    cdm.load(f"best_KAN2CD_embed_{dataset_name}.snapshot")
    auc, accuracy = cdm.eval(test_set)
    print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))


if __name__ == '__main__':
    for dataset_name in ['junyi', 'SLP', 'a0910','FrcSub']:
        print('\n\n\data_set_name:',dataset_name)
        main(dataset_name)