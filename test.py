import tensorflow as tf

def initialize_lookup_table():
    mapping_strings = tf.constant([
        'N',
        'Y'
    ])

    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, default_value=-1)
    tf.tables_initializer().run()

    return table


def get_features_and_labels(data_filename, batch_size=10):
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

    #labels_batch = lookup_table.lookup(labels_batch)
    #labels_batch = tf.cast(tf.reshape(labels_batch,shape=[batch_size,1]), tf.float32)

    return features_batch , labels_batch

with tf.Session() as sess:
    
    # table = initialize_lookup_table()


    features_batch , labels_batch = get_features_and_labels('./data/PC1.csv')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    feat , lab = sess.run([features_batch, labels_batch] )
    print feat
