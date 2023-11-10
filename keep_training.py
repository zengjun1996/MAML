import tensorflow as tf
from my_model import MAMLModel
from data_generation import TaskGenerator
import os
import pandas as pd
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

if __name__ == "__main__":
    # grad_steps = 1
    inner_lr = 0.001
    meta_lr = 0.001
    train_lr = 0.0001
    epochs = 500
    task_dist = TaskGenerator()

    with tf.Session() as sess:
        models = {"meta": MAMLModel("meta", sess, inner_lr, meta_lr, train_lr)}
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        losses_support = {}
        losses_query = {}

        saver.restore(sess, save_path='./newmodel/model')
        print('模型恢复成功')
        time_start = time.perf_counter()
        for i in range(epochs):  # 元训练一万次
            x_support, y_support, x_query, y_query, task_class = task_dist.source_data()
            for model_name, model in models.items():
                if model_name not in losses_support:  # meta
                    losses_support[model_name] = []
                    losses_query[model_name] = []
                loss_support_temp, loss_query_temp = model.train(x_support, y_support, x_query, y_query)
                print('使用CDL_{}的数据进行第{}次元训练,支持集的损失为:{},查询集的损失为:{}'.format(task_class, i+1, loss_support_temp, loss_query_temp))
                # print(norm)
                losses_support[model_name].append(loss_support_temp)
                losses_query[model_name].append(loss_query_temp)
        saver.save(sess, save_path='./model')

        time_end = time.perf_counter()
        print('训练{}次总耗时{:.2f}s'.format(epochs, time_end-time_start))

        support_loss = losses_support['meta']
        query_loss = losses_query['meta']
        pd.DataFrame(support_loss).to_csv('./support_loss.csv', index=0, header=0)
        pd.DataFrame(query_loss).to_csv('./query_loss.csv', index=0, header=0)
        print('支持集和查询集的loss保存完成')
