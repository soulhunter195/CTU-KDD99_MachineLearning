import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import random
import csv

def next_batch(feature_list,label_list,size):
    feature_batch_temp=[]
    label_batch_temp=[]
    f_list = random.sample(range(len(feature_list)), size)
    for i in f_list:
        feature_batch_temp.append(feature_list[i])
    for i in f_list:
        label_batch_temp.append(label_list[i])
    return feature_batch_temp,label_batch_temp

def weight_variable(shape):
    #定义一个shape形状的weights张量
     Weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name='W')
     return Weights

def bias_variable(shape):
    #定义一个shape形状的bias张量
    biases = tf.Variable(tf.constant(0.1, shape=shape),name='b')
    return biases

def conv2d(x, W):
    #卷积计算函数
    # stride [1, x步长, y步长, 1]
    # padding:SAME/FULL/VALID（边距处理方式）
    h_conv2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv2d

def max_pool_2x2(x):
    # max池化函数
    # ksize [1, x边长, y边长,1] 池化窗口大小
    # stride [1, x步长, y步长, 1]
    # padding:SAME/FULL/VALID（边距处理方式）
    h_pool = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return h_pool

def load_data():
    global feature
    global label
    global feature_full
    global label_full
    feature=[]
    label=[]
    feature_full=[]
    label_full=[]
    file_path ='kddcup.data_10_percent_corrected.csv'
    with (open(file_path,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        for rowi in csv_reader:
            label_list=[0]*23
            feature.append(rowi[:36])
            label_list[int(rowi[41])]=1
            label.append(label_list)
    file_path_full ='kddcup.data_10_percent_corrected.csv'
    with (open(file_path_full,'r')) as data_from_full:
        csv_reader_full=csv.reader(data_from_full)
        for rowj in csv_reader_full:
            label_list_full=[0]*23
            feature_full.append(rowj[:36])
            label_list_full[int(rowj[41])]=1
            label_full.append(label_list_full)


if __name__  == '__main__':
    global feature
    global label
    global feature_full
    global label_full
    # load数据
    load_data()
    feature_test = feature
    feature_train =feature_full
    label_test = label
    label_train= label_full
    # 定义用以输入的palceholder
    xs = tf.placeholder(tf.float32, [None, 36]) # 6x6
    ys = tf.placeholder(tf.float32, [None, 23])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 6, 6, 1])    # -1表示不约束这个位置 1表示信道1（灰度图仅有一个信道）

    ## 第一个卷积层 ##
    W_conv1 = weight_variable([3,3,1,32])    # 卷积窗 3x3, 输入厚度 1, 输出厚度 32
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.sigmoid(conv2d(x_image,W_conv1) + b_conv1)    # 输出大小： 6x6x32
    h_pool1 = max_pool_2x2(h_conv1)                             # 输出大小： 3x3x32


    ## 第一个全连接层 ##
    # 带有dropout
    W_fc1 = weight_variable([3*3*32,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool1, [-1,3*3*32])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## 第二个全连接层 ##
    W_fc2 = weight_variable([1024, 23])
    b_fc2 = bias_variable([23])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # 计算loss/cost
    cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))       # loss

    # 计算accuracy
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 使用Adam优化器来实现梯度最速下降
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.Session() as sess:
        # 初始化所有张量
        sess.run(tf.global_variables_initializer())
        for step in range(501):
            feature_train_batch, label_train_batch = next_batch(feature_train, label_train,1000)  # 随机梯度下降训练，每次选大小为1000的batch
            feature_test_batch, label_test_batch = next_batch(feature_test, label_test,1000)  # 随机梯度下降训练，每次选大小为1000的batch
            sess.run(train_step, feed_dict={xs: feature_train_batch, ys: label_train_batch, keep_prob: 0.5})
            if step % 50 == 0:

                print(step,
                      sess.run(tf.argmax(prediction, 1)[7:27], feed_dict={xs: feature_test, ys: label_test, keep_prob: 1}),
                      sess.run(tf.argmax(ys, 1)[7:27], feed_dict={xs: feature_test, ys: label_test, keep_prob: 1}),
                      sess.run(accuracy, feed_dict={xs: feature_test, ys: label_test, keep_prob: 1}),
                      sess.run(accuracy, feed_dict={xs: feature_train_batch, ys: label_train_batch, keep_prob: 1}))