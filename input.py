import tensorflow as tf

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
