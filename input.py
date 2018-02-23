import tensorflow as tf

training_batch_size=150
testing_batch_size=100

learning_rate=0.1

def initialize_lookup_table():
    mapping_strings = tf.constant([
        'N',
        'Y'
    ])

    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, default_value=-1)
    tf.tables_initializer().run()

    return table


def get_features_and_labels(data_filename, lookup_table, batch_size):
    filename_queue = tf.train.string_input_producer([data_filename] , shuffle=False)

    reader = tf.TextLineReader()
    key , value = reader.read(filename_queue)
    
    record_defaults = []
    for _ in range(37):
        record_defaults.append([0.0])

    record_defaults.append(['N'])
    cols = tf.decode_csv(value, record_defaults=record_defaults)

    features = tf.stack(cols[:-1])
    labels = cols[-1]

    features_batch , labels_batch = tf.train.batch([features,labels] , batch_size=batch_size)

    labels_batch = lookup_table.lookup(labels_batch)
    labels_batch = tf.cast(tf.reshape(labels_batch,shape=[batch_size,1]), tf.float32)

    return features_batch , labels_batch


with tf.variable_scope('reg') as scope:
    weights = tf.get_variable('weights', shape=[37,1], initializer=tf.random_normal_initializer())
    bias = tf.get_variable('bias' , shape=[1] , initializer=tf.constant_initializer(0))
    batch_size = tf.placeholder(tf.int32,shape=[])
    

def model(X):
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

    features_batch , labels_batch = get_features_and_labels('./data/train/PC1.csv',table, batch_size)

    print "getting test_features and labels"
    test_features_batch, test_labels_batch = get_features_and_labels('./data/test/test.csv' ,table, batch_size)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    test_features , test_labels = sess.run([test_features_batch,test_labels_batch] , feed_dict={batch_size : testing_batch_size})

    train_z , test_x, test_y, test_prediction = create_model(features_batch,labels_batch)
    loss_a = tf.reduce_mean(compute_loss(train_z))
    
    train_prediction = tf.nn.sigmoid(train_z)
    train_accuracy = compute_accuracy(train_prediction, labels_batch)
    test_accuracy = compute_accuracy(test_prediction , test_y)

    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss_a)
    
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graph')
    writer.add_graph(sess.graph)

    for _ in range(3000):
        _ , loss_val, acc_val = sess.run([train_step , loss_a , train_accuracy], feed_dict={batch_size : training_batch_size})
        print "Loss : {} / Accuracy {}".format(loss_val,acc_val)

    # lets test our accuracy 
    acc_val = sess.run(test_accuracy, feed_dict={test_x : test_features, test_y : test_labels , batch_size : testing_batch_size})
    print "Testing Accuracy : {}".format(acc_val)
