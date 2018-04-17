import tensorflow as tf
import input
import linear
import nn

training_batch_size=659
testing_batch_size=100

learning_rate=0.01
batch_size = tf.placeholder(tf.int32,shape=[])

def initialize_lookup_table():
    mapping_strings = tf.constant([
        'Y',
        'N'
    ])

    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, default_value=-1)
    tf.tables_initializer().run()

    return table

def compute_loss(z):
    with tf.variable_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_batch,logits=z)
    return loss

def compute_accuracy(prediction , labels):
    logits = tf.floor(prediction + 0.5)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels,logits), tf.float32))
    return accuracy

with tf.Session() as sess:

    table = initialize_lookup_table()

    features_batch , labels_batch = input.get_features_and_labels('./data/train/PC1.csv',table, batch_size)

    print "getting test_features and labels"
    test_features_batch, test_labels_batch = input.get_features_and_labels('./data/test/test.csv' ,table, batch_size)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    test_features , test_labels = sess.run([test_features_batch,test_labels_batch] , feed_dict={batch_size : testing_batch_size})

    train_z , test_x, test_y, test_prediction = nn.create_model(features_batch,labels_batch)
    loss_a = tf.reduce_mean(compute_loss(train_z))
    
    train_prediction = tf.nn.sigmoid(train_z)
    train_accuracy = compute_accuracy(train_prediction, labels_batch)
    test_accuracy = compute_accuracy(test_prediction , test_y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss_a)
    
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graph')
    writer.add_graph(sess.graph)

    for _ in range(2000):
        _ , loss_val, acc_val = sess.run([train_step , loss_a , train_accuracy], feed_dict={batch_size : training_batch_size})
        print "Loss : {} / Accuracy {}".format(loss_val,acc_val)

    # lets test our accuracy 
    acc_val , pred = sess.run([test_accuracy, test_prediction], feed_dict={test_x : test_features, test_y : test_labels , batch_size : testing_batch_size})

    print "Prediction vector:-"
    print pred
    print "Testing Accuracy : {}".format(acc_val)


