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
signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1) # only difference


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
num_tests = 100
accuracy = 0.0
for i in range(num_tests):
    batch_x= mnist.test.images[i].reshape(-1,28,28)
    # Reshape data to get 28 seq of 28 elements
    batch_y = mnist.test.labels[i]

    logits = sess.run([y], feed_dict={x: batch_x})
    prediction  = softmax(logits)#tf.exp(logits) / tf.reduce_sum(tf.exp(logits), -1)

    # print(batch_y)
    # print(np.argmax(batch_y))
    # print(np.argmax(prediction))
    correct_pred = sess.run(tf.equal(np.argmax(prediction), np.argmax(batch_y)))
    if correct_pred:
        accuracy +=1
print("Accuracy= %f"%(accuracy*1.0/num_tests))