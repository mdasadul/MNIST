import numpy as np
import requests 
import json
from tensorflow.examples.tutorials.mnist import input_data
import argparse


def main(args):  
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    counter = 0
    num_tests = 100
    for i in range(num_tests):
        image = mnist.test.images[i]
        data = {'data':image.tolist()}
        headers = {'content-type': 'application/json'} 
        url = args.host+':'+str(args.port)

        response = requests.post(url,json.dumps(data),headers = headers)
        response.encoding = 'utf-8'
        response = json.loads(response.text)
        response = response['y']
        if np.argmax(response) ==np.argmax(mnist.test.labels[i]):
            counter +=1
        else:
            pass
    print("Accuracy= %0.2f"%((counter*1.0/num_tests)*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/tmp/data')
    parser.add_argument('--host',default='http://127.0.0.1')
    parser.add_argument('--port',default=5000)
    args = parser.parse_args()

    main(args)