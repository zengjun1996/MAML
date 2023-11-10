import numpy as np
import h5py
import os

path = str(os.path.abspath(os.path.join(os.getcwd(), "..")))


class TaskGenerator(object):
    def __init__(self):
        self.support_rate = 0.5
        self.training_rate = 0.9

    def source_data(self):
        sample_num = 20
        random_task_index = np.random.randint(0, high=5, size=1)  # 随机抽取一个任务
        if random_task_index == 0:
            filepath = path+'/transfer_learning/CDL_A/data_train.mat'
            task_class = 'A'
        elif random_task_index == 1:
            filepath = path+'/transfer_learning/CDL_B/data_train.mat'
            task_class = 'B'
        elif random_task_index == 2:
            filepath = path+'/transfer_learning/CDL_C/data_train.mat'
            task_class = 'C'
        elif random_task_index == 3:
            filepath = path+'/transfer_learning/CDL_D/data_train.mat'
            task_class = 'D'
        else:
            filepath = path+'/transfer_learning/CDL_E/data_train.mat'
            task_class = 'E'

        data = h5py.File(filepath, 'r')  # (4000, 72, 28, 32, 2)
        data_set = data['data_train']
        support_set = data_set[0:int(self.support_rate*4000), :, :, :, :]  # 支持集，查询集五五开
        query_set = data_set[int(self.support_rate*4000):, :, :, :, :]
        support_set_mini = support_set[np.random.randint(0, high=len(support_set), size=sample_num)]
        query_set_mini = query_set[np.random.randint(0, high=len(query_set), size=sample_num)]
        x_support = support_set_mini
        y_support = support_set_mini
        x_query = query_set_mini
        y_query = query_set_mini
        return x_support, y_support, x_query, y_query, task_class

    def target_data(self, task_class):
        sample_num = 50
        if task_class == 'CDL_A':
            filepath = path+'/transfer_learning/CDL_A/data_test.mat'
        elif task_class == 'CDL_B':
            filepath = path+'/transfer_learning/CDL_B/data_test.mat'
        elif task_class == 'CDL_C':
            filepath = path+'/transfer_learning/CDL_C/data_test.mat'
        elif task_class == 'CDL_D':
            filepath = path+'/transfer_learning/CDL_D/data_test.mat'
        else:
            filepath = path+'/transfer_learning/CDL_E/data_test.mat'

        data = h5py.File(filepath, 'r')  # (5000, 72, 28, 32, 2)
        data_set = data['data_test']
        train_set = data_set[0:int(self.training_rate*5000), :, :, :, :]  # 训练集，测试集五五开
        test_set = data_set[int(self.training_rate*5000):, :, :, :, :]
        train_set_mini = train_set[np.random.randint(0, high=len(train_set), size=sample_num)]
        test_set_mini = test_set[np.random.randint(0, high=len(test_set), size=sample_num)]
        x_train = train_set_mini
        y_train = train_set_mini
        x_test = test_set_mini
        y_test = test_set_mini
        return x_train, y_train, x_test, y_test
