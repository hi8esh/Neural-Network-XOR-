import numpy as np
import math
from random import  choice
import pickle
def sigmoid(x):
    return 1/(1+math.exp(-x))
def activation(n):
    row = n.shape[0]
    col = n.shape[1]
    res = np.zeros([row,col])
    for i in range(row):
        for j in range(col):
            res[i][j]=sigmoid(n[i][j])
    return res

class nn:
    def __init__(self,i_nodes , h_nodes , o_nodes,l_rate):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        self.l_rate = l_rate
        self.wih = np.random.rand(self.h_nodes,self.i_nodes)
        self.who = np.random.rand(self.o_nodes,self.h_nodes)

    def predict(self,inputs_lst):
        inputs = np.array(inputs_lst,ndmin=2).T
        hidden_input = np.dot(self.wih,inputs)
        hidden_output = activation(hidden_input)
        final_inputs = np.dot(self.who,hidden_output)
        final_op = activation(final_inputs)
        return final_op



    def train(self,inputs_lst,target_lst):
        # converting to numpy matrix
        inputs = np.array(inputs_lst,ndmin=2).T
        target = np.array(target_lst, ndmin=2).T
        # calculating output
        hidden_input = np.dot(self.wih, inputs)
        hidden_output = activation(hidden_input)
        final_inputs = np.dot(self.who, hidden_output)
        final_op = activation(final_inputs)
        # calculating errors
        error_output = target - final_op
        error_hidden = np.dot(self.who.T,error_output)
        # refining weigths
        self.who += self.l_rate*(np.dot((error_output*final_op*(1.0-final_op)),np.transpose(hidden_output)))
        self.wih += self.l_rate*(np.dot((error_hidden*hidden_output*(1.0-hidden_output)),np.transpose(inputs)))


training_data = [
    {"inputs":[0,0], "targets":[0]},
    {"inputs":[0,1], "targets":[1]},
    {"inputs":[1,0], "targets":[1]},
    {"inputs":[1,1], "targets":[0]}
]

N = nn(2,10,1,1)
for i in range(10000):
    current_choice = choice(training_data)
    N.train(current_choice["inputs"], current_choice["targets"])
with open("./trainned_model",'wb') as file:
    pickle.dump(N,file)
with open('trainned_model','rb') as file:
    N = pickle.load(file)
print(N.predict([0,0]))
print(N.predict([0,1]))
print(N.predict([1,0]))
print(N.predict([1,1]))