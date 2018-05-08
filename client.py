import numpy as np
from flask import Flask, request
import requests 
import json
from tensorflow.examples.tutorials.mnist import input_data

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1) # only difference


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
counter = 0
num_tests = 1000
for i in range(num_tests):
    image = mnist.test.images[i]

    #print(image.shape)
    #print(image.reshape(-1,28,28).shape)
    data = {'data':image.tolist()}

    headers = {'content-type': 'application/json'} 
    response = requests.post('http://127.0.0.1:5000/',json.dumps(data),headers = headers)
    response.encoding = 'utf-8'
    
    response = json.loads(response.text)
    response = response['y']
    # print(mnist.test.labels[i])
    if np.argmax(response) ==np.argmax(mnist.test.labels[i]):
        # print(np.argmax(response))
        # print(np.argmax(mnist.test.labels[i]))
        counter +=1
    else:
        pass
print("Accuracy= %0.2f"%((counter*1.0/num_tests)*100))