import tensorflow as tf
import numpy as np
from flask import Flask, request
import json

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1) # only difference


app =Flask(__name__)
@app.route('/',methods=['POST'])
def predict():
    
    from_client = request.get_json()
    inference_data = from_client['data']
    inference_data = np.array(inference_data)
    batch_x = inference_data.reshape(-1,28,28)
    
    logits = sess.run([y],feed_dict={x:batch_x})
    prediction = softmax(logits)
    json_data = json.dumps({'y':prediction.tolist()})
    return json_data

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('model_path','./savedmodel/2/',help='model Path')
    FLAGS = tf.app.flags.FLAGS
    sess=tf.Session()

    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_key = 'x_input'
    output_key = 'y_output'

    export_path =  FLAGS.model_path
    meta_graph_def = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            export_path)
    signature = meta_graph_def.signature_def

    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name = signature[signature_key].outputs[output_key].name

    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)

    app.run()