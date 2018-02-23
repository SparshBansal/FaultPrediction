import tensorflow as tf 


def model(X):
    with tf.variable_scope('reg') as scope:
        weights = tf.get_variable('weights', shape=[37,1], initializer=tf.random_normal_initializer())
        bias = tf.get_variable('bias' , shape=[1] , initializer=tf.constant_initializer(0))
    z = tf.nn.bias_add(tf.matmul(X, weights) , bias )
    return z

def create_model(features_batch, labels_batch):
    with tf.variable_scope('logistic_regression') as scope:
        train_z = model(features_batch)

        test_x = tf.placeholder(tf.float32 , shape=[None,features_batch.get_shape()[1]])
        test_y = tf.placeholder(tf.float32 , shape=[None,labels_batch.get_shape()[1]])

        scope.reuse_variables()
        
        test_z = model(test_x)
        test_output = tf.nn.sigmoid(test_z)

    return train_z, test_x, test_y, test_output

