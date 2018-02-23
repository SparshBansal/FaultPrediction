import tensorflow as tf

def model(X):
    # define weights for single layer neural network
    with tf.variable_scope('weights_l1') as scope:
        weight = tf.get_variable('weight' , shape=[X.get_shape()[1],10],initializer=tf.random_normal_initializer())
        bias = tf.get_variable('bias' , shape=[10], initializer=tf.constant_initializer(0))
        
    z_1 = tf.matmul(X,weight) + bias
    a_1 = tf.nn.sigmoid(z_1)

    with tf.variable_scope('weights_l2') as scope:
        weight = tf.get_variable('weight' , shape=[10,1], initializer=tf.random_normal_initializer())
        bias = tf.get_variable('bias',shape=[1] , initializer=tf.constant_initializer(0))
    
    z_2 = tf.matmul(a_1,weight) + bias
    return z_2

def create_model(features_batch, labels_batch):
    with tf.variable_scope('nn') as scope:
        train_z = model(features_batch)
        
        test_x = tf.placeholder(tf.float32,shape=[None,features_batch.get_shape()[1]])
        test_y = tf.placeholder(tf.float32,shape=[None,labels_batch.get_shape()[1]])

        scope.reuse_variables()

        test_z = model(test_x)
        test_a = tf.nn.sigmoid(test_z)

        return train_z, test_x, test_y, test_a

