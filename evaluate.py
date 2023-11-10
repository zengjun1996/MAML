import tensorflow as tf
import numpy as np
import h5py
import os

path = str(os.path.abspath(os.path.join(os.getcwd(), "..")))+'/transfer_learning/'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)


# def model(data, envir):
def model(data):
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('./'+envir+'/model.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./'+envir+'/'))
        saver = tf.train.import_meta_graph('./model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        data_input = graph.get_tensor_by_name('meta/inputs1:0')
        data_output = graph.get_tensor_by_name('meta/output:0')
        data_predict = sess.run(data_output, feed_dict={data_input: data})
        # loss_testset = sess.run(tf.losses.mean_squared_error(data, data_predict))
    return data_predict

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('model.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('./'))
#     print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph())[0:5])
#     print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph())[-1])
    # for variable_name in tf.global_variables():
#     print(variable_name)


envir = 'CDL_A'
filepath = path + envir + '/data_val.mat'
compression_ratio = '1_8'
# epoch = 50

data = h5py.File(filepath, 'r')
data_test = data['data_val']
data_predict = np.empty(shape=(1000, 72, 28, 32, 2), dtype=np.float32)

for i in range(5):
    data_test_temp = data_test[i*200:(i+1)*200, :, :, :, :]
    # data_predict_temp = model(data_test_temp, envir)  # 考虑分批进行预测，再把所有的预测组合起来
    data_predict_temp = model(data_test_temp)  # 考虑分批进行预测，再把所有的预测组合起来
    print('第{}组预测完毕'.format(i+1))
    data_predict[i*200:(i+1)*200, :, :, :, :] = data_predict_temp

loss_testset = np.mean(np.square(data_test-data_predict))
print('测试集损失为: {}'.format(loss_testset))

data_test_real = np.reshape(data_test[:, :, :, :, 0], (len(data_test), -1))
data_test_imag = np.reshape(data_test[:, :, :, :, 1], (len(data_test), -1))
data_test_C = data_test_real + 1j*data_test_imag    # (5000, 72*28*32)

data_predict_real = np.reshape(data_predict[:, :, :, :, 0], (len(data_predict), -1))
data_predict_imag = np.reshape(data_predict[:, :, :, :, 1], (len(data_predict), -1))
data_predict_C = data_predict_real + 1j*data_predict_imag    # (5000, 72*28*32)

n1 = np.sqrt(np.sum(abs(np.conj(data_test_C)*data_test_C), axis=1))  # (5000, 1)
n2 = np.sqrt(np.sum(abs(np.conj(data_predict_C)*data_predict_C), axis=1))  # (5000, 1)
aa = np.sum(abs(np.conj(data_test_C)*data_predict_C), axis=1)
rho = np.mean(aa/(n1*n2))
power = np.sum(abs(data_test_C)**2, axis=1)
mse = np.sum(abs(data_test_C-data_predict_C)**2, axis=1)
print('在{}环境中'.format(envir))
print('当压缩率为:{}'.format(compression_ratio))
# print('epoch为: {}'.format(epoch))
print('NMSE为:{}dB'.format(10*np.log10(np.mean(mse/power))))
print('余弦相似度为:{}'.format(rho))
