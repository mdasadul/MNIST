import tensorflow as tf
import numpy as np
sess=tf.Session() 
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'x_input'
output_key = 'y_output'

export_path =  './savedmodel'
meta_graph_def = tf.saved_model.loader.load(
           sess,
          [tf.saved_model.tag_constants.SERVING],
          export_path)
signature =  meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
batch_x= mnist.test.images[:1].reshape(-1,28,28)
# Reshape data to get 28 seq of 28 elements
batch_y = mnist.test.labels[:1][0]
prediction = sess.run([y], feed_dict={x: batch_x})
print("HERE")
print(batch_y)
print(np.argmax(batch_y))
print(np.argmax(prediction))
correct_pred = tf.equal(np.argmax(prediction), np.argmax(batch_y))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print(sess.run(correct_pred))