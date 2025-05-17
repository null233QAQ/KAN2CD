
import logging
from CDMs import KAN2CD_e,utils_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 保留原版代码主干，仅在数据加载部分做区分
def main(dataset_name='', use_subset=False, subset_ratio=0.1):
    # 判断是否为assist12或assist17数据集
    if dataset_name in ['assist12', 'assist2017']:
        # 使用utils_dataset加载数据
        print(f"Using utils_dataset method for {dataset_name}")
        train_set, valid_set, test_set, user_n, item_n, knowledge_n = utils_dataset.get_dataset(
            name=dataset_name,
            batch_size=128,
            ratio=0.8
        )

        # 如果use_subset为True，则只使用指定比例的数据
        if use_subset:
            subset_ratio_percent = subset_ratio * 100
            print(f"Using {subset_ratio_percent:.1f}% of the data for {dataset_name}")
            # 对训练集进行采样
            subset_size = int(len(train_set.dataset) * subset_ratio)
            subset_indices = torch.randperm(len(train_set.dataset))[:subset_size]
            subset_dataset = torch.utils.data.Subset(train_set.dataset, subset_indices)
            train_set = DataLoader(subset_dataset, batch_size=128, shuffle=True)
    else:
        # 使用原始方法加载数据
        print(f"Using original method for {dataset_name}")
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

    # 以下是原始代码主干，不做修改
    logging.getLogger().setLevel(logging.INFO)
    cdm = KAN2CD_e(knowledge_n, item_n, user_n, dataset_name=dataset_name)
    cdm.train(train_set, valid_set, epoch=25, device="cuda")
    cdm.save(f"{dataset_name}_KAN2CD.snapshot")

    cdm.load(f"best_KAN2CD_embed_{dataset_name}.snapshot")
    auc, accuracy = cdm.eval(test_set)
    print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))


if __name__ == '__main__':
    # 添加assist12和assist17到数据集列表
    dataset_list = ['junyi', 'SLP', 'a0910', 'FrcSub']

    # 数据子集选项
    use_subset = False  # 是否使用数据子集，默认为False
    subset_ratio = 0.1  # 使用原始数据的比例，默认为0.1（10%）

    for dataset_name in dataset_list:
        main(dataset_name)
        # print('\n\n\data_set_name:', dataset_name)
        # try:
        #     # 只对assist12和assist17应用use_subset参数
        #     if dataset_name in ['assist12', 'assist17']:
        #         main(dataset_name, use_subset, subset_ratio)
        #     else:
        #         main(dataset_name)
        # except Exception as e:
        #     print(f"Error processing {dataset_name}: {str(e)}")