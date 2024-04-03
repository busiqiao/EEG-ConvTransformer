from sklearn.model_selection import KFold
import pickle
from data_load.dataset import EEGImagesDataset

num_class = 6
dataPath = 'H:\\EEG\\EEGDATA\\img_pkl_124'

# 创建KFold对象
k = 10
k_fold = KFold(n_splits=k, shuffle=True, random_state=42)

def create_kfold_indices():
    all_indices = []
    for i in range(0, 10):
        # 数据集
        dataset = EEGImagesDataset(file_path=dataPath + '\\' + 'S' + str(i + 1) + '\\', num_class=num_class)

        indices = [(train_index, test_index) for train_index, test_index in k_fold.split(dataset)]

        all_indices.append(indices)
        print(f'Sub {i + 1} indices created.')

    # 保存所有的索引到同一个文件中
    with open('kfold_indices.pkl', 'wb') as f:
        pickle.dump(all_indices, f)
    print('All indices created and saved to kfold_indices.pkl')


if __name__ == '__main__':
    create_kfold_indices()
