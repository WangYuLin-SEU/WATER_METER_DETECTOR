#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

"""
权重初始化  weight init
初始化为一个接近0的很小的正数  init to a small number close to zero
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

"""
卷积和池化，使用卷积步长为1（stride size）,0边距（padding size）
池化用简单传统的2x2大小的模板做max pooling
conv and pool
"""
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # x(input)  : [batch, in_height, in_width, in_channels]
    # W(filter) : [filter_height, filter_width, in_channels, out_channels]
    # strides   : The stride of the sliding window for each dimension of input.
    #             For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # x(value)              : [batch, height, width, channels]
    # ksize(pool大小)        : A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides(pool滑动大小)   : A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.

start = time.clock() #计算开始时间
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #MNIST数据输入

"""
第一层 卷积层   1st layer  conv layer
x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 16)
"""
x = tf.placeholder(tf.float32,[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])  
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# x_image -> [batch, in_height, in_width, in_channels]
#            [batch, 28, 28, 1]
# W_conv1 -> [filter_height, filter_width, in_channels, out_channels]
#            [3, 3, 1, 16]
# output  -> [batch, out_height, out_width, out_channels]
#            [batch, 28, 28, 16]
h_pool1 = max_pool_2x2(h_conv1)
# h_conv1 -> [batch, in_height, in_weight, in_channels]
#            [batch, 28, 28, 16]
# output  -> [batch, out_height, out_weight, out_channels]
#            [batch, 14, 14, 16]

"""
第二层 卷积层  2nd layer  conv layer
h_pool1(batch, 14, 14, 16) -> h_pool2(batch, 7, 7, 32)
"""
W_conv2 = weight_variable([3, 3, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool1 -> [batch, 14, 14, 16]
# W_conv2 -> [5, 5, 16, 32]
# output  -> [batch, 14, 14, 32]
h_pool2 = max_pool_2x2(h_conv2)
# h_conv2 -> [batch, 14, 14, 32]
# output  -> [batch, 7, 7, 32]

"""
第三层 全连接层   3rd layer  full connect layer 
h_pool2(batch, 7, 7, 32) -> h_fc1(1, 1)
"""
W_fc1 = weight_variable([7 * 7 * 32, 32])
b_fc1 = bias_variable([32])

h_pool2_flat = tf.layers.flatten(h_pool2)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

"""
Dropout
h_fc1 -> h_fc1_drop
"""
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

"""
第四层 Softmax输出层   4th layer  Softmax output
"""
W_fc2 = weight_variable([32, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

"""
训练和评估模型		train and eval the module
ADAM优化器来做梯度最速下降,feed_dict中加入参数keep_prob控制dropout比例
"""
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv)) #计算交叉熵  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #使用adam优化器来以0.0001的学习率来进行微调
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #判断预测标签和实际标签是否匹配
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
saver = tf.train.Saver()

sess = tf.Session() #启动创建的模型
#sess.run(tf.initialize_all_variables()) #旧版本
sess.run(tf.global_variables_initializer()) #初始化变量
merged = tf.summary.merge_all() 
writer = tf.summary.FileWriter('logs',sess.graph)

for i in range(20000): #开始训练模型，循环训练5000次
    batch = mnist.train.next_batch(50) #batch大小设置为50
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session = sess,
                                       feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, train_accuracy %g" %(i, train_accuracy))
        saver.save(sess, 'model/mnist.ckpt')
    train_step.run(session = sess, feed_dict = {x:batch[0], y_:batch[1],
                   keep_prob:0.5}) #神经元输出保持不变的概率 keep_prob 为0.5

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, 'model/mnist.ckpt')
print( "test accuracy %g" % accuracy.eval(feed_dict={
    x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))


#get parms
WC1 = W_conv1.eval()
BC1 = b_conv1.eval()
WC2 = W_conv2.eval()
BC2 = b_conv2.eval()
WF1 = W_fc1.eval()
BF1 = b_fc1.eval()
WF2 = W_fc2.eval()
BF2 = b_fc2.eval()
#restruct new graph to save
g = tf.Graph()
with g.as_default():
    x_image=tf.placeholder("float", shape=[None,28,28,1], name="inputs")

    WC1 = tf.constant(WC1, name="WC1")
    BC1 = tf.constant(BC1, name="BC1")
    CONV1 = tf.nn.relu(conv2d(x_image, WC1) + BC1)
    MAXPOOL1 = max_pool_2x2(CONV1)

    WC2 = tf.constant(WC2, name="WC2")
    BC2 = tf.constant(BC2, name="BC2")
    CONV2 = tf.nn.relu(conv2d(MAXPOOL1, WC2) + BC2)
    MAXPOOL2 = max_pool_2x2(CONV2)

    WF1 = tf.constant(WF1, name="WF1")
    BF1 = tf.constant(BF1, name="BF1")
    FC1 = tf.layers.flatten(MAXPOOL2)   
    FC1 = tf.nn.relu(tf.matmul(FC1, WF1) + BF1)

    WF2 = tf.constant(WF2, name="WF2")
    BF2 = tf.constant(BF2, name="BF2")
    OUTPUT = tf.nn.softmax(tf.matmul(FC1, WF2) + BF2,name="output")
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    graph_def = g.as_graph_def()
    tf.train.write_graph(graph_def, "./", "mnist.pb", as_text=False)


