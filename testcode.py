import numpy as np

# # 从.npy文件加载数据
# data = np.load('/home/cosi/project/Integrated_RegNet/regretNet/experiments/aaa/train_vol.npy')
#
# data2d = data.reshape(-1, data.shape[-1])
#
# np.savetxt('/home/cosi/project/Integrated_RegNet/regretNet/experiments/aaa/train_vol1.txt', data2d)


import tensorflow as tf

# CTR = np.array([0.8,0.6,0.4])
#
# result = np.zeros((2,4,3))
#
# uniform_value = 2 * np.ones((result.shape[0], result.shape[1]))
#
# print(uniform_value)
#
# x = np.tile(np.reshape(uniform_value, [result.shape[0], result.shape[1], -1]), [1, 1, result.shape[-1]])
#
# value = np.tile(CTR.reshape(1, 1, 3), [2, 4, 1])
#
# print('\n')
#
# # with tf.Session() as sess:
# #     result = sess.run(x)
# print(x * value)

CTR = np.array([[0.8,0.6,0.4],[0.4,0.6,0.8],[0.3,0.6,0.9]])

x = tf.reduce_mean(CTR,axis=0)
with tf.Session() as sess:
    result = sess.run(x)
print(result)