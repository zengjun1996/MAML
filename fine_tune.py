import tensorflow as tf
from model import MAMLModel
from data_generation import TaskGenerator
import os
import pandas as pd
import time


path = str(os.path.abspath(os.path.join(os.getcwd(), "..")))+'/transfer_learning/'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)


if __name__ == "__main__":
    task_class = 'CDL_A'
    # grad_steps = 1
    inner_lr = 0.001
    meta_lr = 0.001
    train_lr = 0.0001
    epoch = 1000

    task_dist = TaskGenerator()

    with tf.Session() as sess:
        models = {"meta": MAMLModel("meta", sess, inner_lr, meta_lr, train_lr)}
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        losses_train = {}
        losses_test = {}

        saver.restore(sess, save_path='./CDL_A/1000_2w_s_0.5/model')
        print('模型恢复成功')
        print('用{}的数据进行fine-tune'.format(task_class))
        for model_name, model in models.items():
            if model_name not in losses_train:
                losses_train[model_name] = []
                losses_test[model_name] = []
            # 用训练集进行fine-tune
            time_start = time.perf_counter()
            for i in range(epoch):
                x_train, y_train, x_test, y_test = task_dist.target_data(task_class)
                train_loss, test_loss = model.test(x_train, y_train, x_test, y_test)
                print('第{}次训练的训练集损失为{}，验证集损失为{}'.format(i+1, train_loss, test_loss))
                losses_train[model_name].append(train_loss)
                losses_test[model_name].append(test_loss)
            time_end = time.perf_counter()
            print('训练集初始loss:{}-------训练集最终loss:{}'.format(losses_train['meta'][0], losses_train['meta'][-1]))
            print('验证集初始loss:{}-------验证集最终loss:{}'.format(losses_test['meta'][0], losses_test['meta'][-1]))
            print('训练{}次总耗时{:.2f}s'.format(epoch, time_end-time_start))
        saver.save(sess, save_path='./{}/model'.format(task_class))

        train_losses = losses_train['meta']
        test_losses = losses_test['meta']
        pd.DataFrame(train_losses).to_csv('./{}/train_loss.csv'.format(task_class), index=0, header=0)
        pd.DataFrame(test_losses).to_csv('./{}/test_loss.csv'.format(task_class), index=0, header=0)
        print('训练集和验证集的loss保存完成')
