import numpy as np
import random
import signal
import sys
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import time
import os


population = 100
gene_length = 4
parent_size = 2
child_size = 2
species_threshold = 0.05
##if the remainder of rand/mutation == 0 than mutation occurs 
mutation = 2
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
class EvolvingNN(object):
    ##column names is a list of the header values at the top of each column
    ##OUTPUT IS ALWAYS THE LAST COLUMN
    def __init__(self,filename,column_names):
        #1 in "mutation" chance of mutation
        self.sorted_index = []
        self.old_creature = []
        self.creature_error = []
        self.input_size = len(column_names)
        #index, genes
        self.creature = np.empty((population,gene_length),int)
        self.parent = np.empty((parent_size,gene_length),int)

        df = pd.read_excel(filename)
        temp = (df[column_names[0]].to_numpy(np.float64))
        self.X = np.empty((len(column_names) - 1,temp.size),np.float64)
        training_outputs = np.empty((1,temp.size),float)

        for i in range(len(column_names)):
            if i == len(column_names)-1:
                training_outputs[0] = (df[column_names[i]].to_numpy(np.float64))
                self.y = np.array(training_outputs).T
            else:
                self.X[i] = (df[column_names[i]].to_numpy(np.float64))
        self.X_full = self.X
        self.y_full = self.y
        print(self.X)
        print(self.y)
        self.scaler = (np.amax(self.y_full)*2)
        self.X = self.X.T/self.scaler
        self.y = self.y/self.scaler
        
    ###################################### Genetic Algorithm ######################################
    def spawn(self):
        for i in range(population):
            if len(self.sorted_index) < population:
                self.sorted_index.append(population-1)
            for j in range(gene_length):
                self.creature[i][j] = random.randint(0, 100)
    def test(self,depth):
        self.creature_error.clear()
        ##test each creature against the problem
        for i in range(population):
            self.creature_error.append(0)
            self.sorted_index.append(population-1)

            ##test each creature and record the score
            self.createNetwork(self.creature[i][0],self.creature[i][1],self.creature[i][2],self.creature[i][3])
            for gen in range(depth):
                self.train(self.think())
            self.creature_error[i] = np.mean(np.abs(self.y*self.scaler - self.think()[4]*self.scaler))

        #arage the best to worst creature indexes into a list
        for j in range(population):
            for i in range(j,population):
                if self.creature_error[i] < self.creature_error[self.sorted_index[j]]:
                    #i is inserted at the j index
                    self.sorted_index.insert(j,i)
    def newGen(self):
        self.old_creature = self.creature.copy()
        ###create children for all except the first child_size+parent_size
        for i in range (child_size+parent_size,population):
            ##chose any random parents
            for j in range(gene_length):
                mutate = random.randint(1,100)
                if mutate%mutation == 0:
                    self.creature[i][j] = random.randint(0, 100)
                else:
                    self.creature[i][j] =  self.old_creature[self.sorted_index[random.randint(0,population)]][j]

        ##select the best parents and propagate them to the next gen
        for i in range(parent_size):
            self.parent[i] = self.old_creature[self.sorted_index[i]].copy()
            self.creature[i] = self.old_creature[self.sorted_index[i]].copy()
        ##create first 2 children from the best 2 parrents
        for i in range(parent_size,child_size+parent_size):
            for j in range(gene_length):
                parentIndex = random.randint(0,parent_size-1)
                mutate = random.randint(1,mutation)
                if mutate%mutation == 0:
                    self.creature[i][j] = random.randint(0, 100)
                else:
                    self.creature[i][j] =  self.parent[parentIndex][j]
    def output(self,loop):
        if loop == "E":
            print(self.creature)
            print("Best = " + str(self.creature_error[self.sorted_index[0]]))
        elif loop == "S":
            print ("Output:")
            print (self.think()[4]*self.scaler)
            print ("Error:" + str(np.mean(np.abs(self.y*self.scaler - self.think()[4]*self.scaler))))
    ###################################### Genetic Algorithm ######################################

    ###################################### Neural Network #########################################
    def createNetwork(self,N1,N2,N3,N4):
        try:
            self.syn0 = np.loadtxt('syn0.txt', dtype=float)
            self.syn1 = np.loadtxt('syn1.txt', dtype=float)
            self.syn2 = np.loadtxt('syn2.txt', dtype=float)
            self.syn3 = (np.loadtxt('syn3.txt', dtype=float)).reshape((N3,1))
            #print("L0: "+str(self.syn0.shape)+"\n"+"L1: "+str(self.syn1.shape)+"\n"+"L2: "+str(self.syn2.shape)+"\n"+"L3: "+str(self.syn3.shape)+"\n")
            #print(str(self.syn3))
            np.set_printoptions(threshold=np.inf)
            print(str(self.syn0) + "\n\n\n")
            print(str(self.syn1) + "\n\n\n")
            print(str(self.syn2) + "\n\n\n")
            print(str(self.syn3) + "\n\n\n")
        except:
            # initialize weights randomly with mean 0
            np.random.seed(4)
            ########## INPUT  Layer 0 ################
            self.syn0 = 2*np.random.random((self.input_size-1,N1)) - 1
            ########## Neural Layer 1 ################
            self.syn1 = 2*np.random.random((N1,N2)) - 1
            ########## Neural Layer 2 ################
            self.syn2 = 2*np.random.random((N2,N3)) - 1
            ########## Neural Layer 3 ################
            self.syn3 = 2*np.random.random((N3,1)) - 1  
            ########## Output Layer 4 ################
            #print("L0: "+str(self.syn0.shape)+"\n"+"L1: "+str(self.syn1.shape)+"\n"+"L2: "+str(self.syn2.shape)+"\n"+"L3: "+str(self.syn3.shape)+"\n")
            #print(str(self.syn3))
    def think(self):
        # forward propagation
        l0 = self.X
        l1 = nonlin(np.dot(l0,self.syn0))
        l2 = nonlin(np.dot(l1,self.syn1))
        l3 = nonlin(np.dot(l2,self.syn2))
        l4 = nonlin(np.dot(l3,self.syn3))
        layers = [l0,l1,l2,l3,l4]
        return layers
    def train(self,layers):
        l4_error = self.y - layers[4]
        l4_delta = l4_error*nonlin(layers[4],deriv=True)

        l3_error = l4_delta.dot(self.syn3.T)
        l3_delta = l3_error*nonlin(layers[3],deriv=True)

        l2_error = l3_delta.dot(self.syn2.T)
        l2_delta = l2_error*nonlin(layers[2],deriv=True)

        l1_error = l2_delta.dot(self.syn1.T)
        l1_delta = l1_error*nonlin(layers[1],deriv=True)

        self.syn3 += layers[3].T.dot(l4_delta)
        self.syn2 += layers[2].T.dot(l3_delta)
        self.syn1 += layers[1].T.dot(l2_delta)
        self.syn0 += layers[0].T.dot(l1_delta)
    @staticmethod
    def Menu():
        print("CTRL-C opens the exit menu")
        print("Train Neural Net = S")
        print("Evolve Nerual Net Structure = E")
        loop = input("Input Your Selection: ")
        return loop
    ###################################### Neural Network #########################################
    def gety(self):
        return self.y
    def exit(self,loop):
        #print("exiting")
        #print (creature)
        #print(str(creature_error[sorted_index[0]]/len(xs)))
        if loop == "S":
            print("Ask Trained Neural Net = \"A\"")
            print("Save Trained Nerual Net = \"S\"")
            choice = input("Input Your Selection: ")
            if choice == "A":
                again = "Y"
                while again == "Y":
                    userin = np.empty((1,self.input_size-1),float)
                    for i in range(self.input_size-1):
                        userin[0][i] = input("Input #"+str(i) +": ")
                    l0 = userin/self.scaler
                    l1 = nonlin(np.dot(l0,self.syn0))
                    l2 = nonlin(np.dot(l1,self.syn1))
                    l3 = nonlin(np.dot(l2,self.syn2))
                    l4 = nonlin(np.dot(l3,self.syn3))
                    print ("Calculated: " + str(l4*self.scaler))

                    again = input("press enter to continue or input \"Y\" to ask again:")
            elif choice == "S":
                try:
                    os.remove("syn0.txt")
                    os.remove("syn1.txt")
                    os.remove("syn2.txt")
                    os.remove("syn3.txt")
                except:
                    pass
                global synanpse
                #synapse = np.array([syn0, syn1, syn2, syn3],dtype=object)
                np.set_printoptions(threshold=np.inf)
                print(str(self.syn0) + "\n\n\n")
                print(str(self.syn1) + "\n\n\n")
                print(str(self.syn2) + "\n\n\n")
                print(str(self.syn3) + "\n\n\n")
                np.savetxt('syn0.txt', self.syn0, fmt='%1.8f')
                np.savetxt('syn1.txt', self.syn1, fmt='%1.8f')
                np.savetxt('syn2.txt', self.syn2, fmt='%1.8f')
                np.savetxt('syn3.txt', self.syn3, fmt='%1.8f')
        elif loop =="E":
            save = input("press enter to continue or input \"Y\" to Save Nerual Net Structure: ")
            if save == "Y":
                try:
                    os.remove("structure.txt")
                except:
                    pass
                neurons = self.creature[0]
                np.savetxt('structure.txt', neurons, fmt='%d')
        else:
            sys.exit()
