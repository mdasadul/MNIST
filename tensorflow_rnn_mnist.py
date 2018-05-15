from __future__ import print_function
import os
import tensorflow as tf


dir = os.path.dirname(os.path.realpath(__file__))
from tensorflow.contrib import rnn
tf.app.flags.DEFINE_integer('model_version', 3, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
# Training Parameters
learning_rate = 0.001
BATCH_SIZE = 128
EPOCHS = 5

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 256 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)
output_layer = ['output_layer/add']
# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input],name="Input_X")
Y = tf.placeholder("float", [None, num_classes])
batch_size = tf.placeholder(tf.int64)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))}

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
n_batches = len(train_images) // BATCH_SIZE

dataset = tf.data.Dataset.from_tensor_slices((X,Y)).batch(BATCH_SIZE).repeat().prefetch(10000)

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

iterator = dataset.make_initializable_iterator()
batch_x, batch_y =iterator.get_next()

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    x = tf.unstack(x, timesteps, 1)
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    with tf.name_scope('output_layer'):
        logit = tf.add(tf.matmul(outputs[-1], weights['out']) , biases['out'],name ='add')
    return logit

logits = RNN(batch_x, weights, biases)
prediction = tf.nn.softmax(logits,name='prediction')

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=batch_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(batch_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

output_tensor = tf.get_default_graph().get_tensor_by_name("output_layer/add:0")
input_tensor = tf.get_default_graph().get_tensor_by_name("Input_X:0")
saver = tf.train.Saver()

# =]tart training
with tf.Session() as sess:
    
    # Run the initializer
    sess.run(init)
    sess.run(iterator.initializer,feed_dict={X: train_images, Y: train_labels, batch_size:BATCH_SIZE})

    for i in range(EPOCHS):
        total_loss = 0.0
        for step in range(n_batches):
            
            _,loss = sess.run([train_op,loss_op])
            total_loss +=loss
        
        acc = sess.run(accuracy)
        print("Epoch " + str(i) + ", Minibatch Loss= " + \
                    "{:.4f}".format(total_loss/n_batches) + ", Training Accuracy= " + \
                    "{:.3f}".format(acc))
    print("Optimization Finished!")
    saver.save(sess,dir+'/tmp/model.ckpt')
    # Calculate accuracy for 128 mnist test images
    sess.run(iterator.initializer,feed_dict={X: test_images,Y:test_labels, batch_size: test_images.shape[0]})
    print("Testing loss:", \
        sess.run(loss_op))

    print("Testing Accuracy:", \
        sess.run(accuracy))
    graphdef = tf.get_default_graph().as_graph_def()
    # save the model
    
    export_base_path =  './savedmodel'
    export_path = os.path.join(
      tf.compat.as_bytes(export_base_path),
      tf.compat.as_bytes(str(FLAGS.model_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(input_tensor)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(output_tensor)

    prediction_signature = (
         tf.saved_model.signature_def_utils.build_signature_def(
         inputs={'x_input': tensor_info_x},
          outputs={'y_output': tensor_info_y},
         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
       sess, [tf.saved_model.tag_constants.SERVING],
       signature_def_map={
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          prediction_signature
       },
      )
    builder.save()