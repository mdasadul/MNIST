import numpy as np
import requests 
import json
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from grpc.beta import implementations
import tensorflow as tf
import time
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
from tensorflow.contrib.util import make_tensor_proto


def main(args):  
    
    host, port = args.host, args.port

    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    
    start_time = time.time()
    counter = 0
    num_tests = args.num_tests
    time
    for _ in range(num_tests):
        image, label = mnist.test.next_batch(1)
        
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        request.inputs['x_input'].CopyFrom(
            make_tensor_proto(image[0],shape=[1,28,28])
        )
        result_future = stub.Predict(request,10.0)
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            response = np.array(result_future.outputs['y_output'].float_val)
            if np.argmax(response) ==np.argmax(label):
                counter +=1
            else:
                pass
    print("Accuracy= %0.2f"%((counter*1.0/num_tests)*100))
    print("Time takes to run the test %0.2f"%(time.time()-start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/tmp/data')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',default=9000)
    parser.add_argument('--num_tests',default=100)
    args = parser.parse_args()

    main(args)