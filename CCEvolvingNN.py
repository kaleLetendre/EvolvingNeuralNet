import cupy as cp
import numpy as np
import random
import sys
import pandas as pd
import time
import os
import multiprocessing
import warnings
import scipy


warnings.filterwarnings("ignore")

population = 100
#gene_length = 2
parent_size = int(population/10)
child_size = int(population/5)
learningRate = 0.01
##if the remainder of rand/mutation == 0 than mutation occurs 
mutation = 2
def nonlin(x,deriv=False):
    if(deriv==True):
        return (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
    return 1/(1+np.exp(-x))
class CCEvolvingNN(object):
    ##column names is a list of the header values at the top of each column
    ##OUTPUT IS ALWAYS THE LAST COLUMN
    def setGene(self,gene):
        self.gene_length = gene
    def __init__(self,filename,column_names):
        self.column_names = column_names
        try:
           self.setGene(int(input("how many hidden layers 2-5?(default is 3)")))
        except:
            self.setGene(3)
        #1 in "mutation" chance of mutation
        self.sorted_index = []
        self.old_creature = []
        self.creature_error = []
        self.input_size = len(column_names)
        #index, genes
        self.creature = np.empty((population,self.gene_length),int)
        self.parent = np.empty((parent_size,self.gene_length),int)
        #######################################
        #######################################
        ##NROWS IS THE MAX ROW TO READ TOO#####
        df = pd.read_excel(filename,nrows=10000)
        #######################################
        #######################################
        temp = (df[column_names[0]].to_numpy(np.float64))
        self.X = np.empty((len(column_names) - 1,temp.size),np.float64)
        training_outputs = np.empty((1,temp.size),float)

        for i in range(len(column_names)):
            if i == len(column_names)-1:
                training_outputs[0] = df[column_names[i]].to_numpy(np.float64)
                self.y = np.array(training_outputs).T
            else:
                self.X[i] = (df[column_names[i]].to_numpy(np.float64))
        self.X_full = self.X
        self.y_full = self.y
        self.scaler = (np.amax(self.y_full)*2)
        self.scalerx = (np.amax(self.X_full)*2)
        self.X = self.X.T/self.scalerx
        self.y = self.y/self.scaler
        #print(self.X)
        # print(self.X.shape[0]+1)
        # print(self.X.shape[1]+1)
        #print(self.gene_length)
        self.maxNode = int(((self.X.shape[0]+1)/(2*(self.X.shape[1]+1)))/self.gene_length)
        if(self.maxNode < 3):
            self.maxNode = 3
        # print(self.maxNode)
    ###################################### Genetic Algorithm ######################################
    def spawn(self):
        for i in range(population):
            if len(self.sorted_index) < population:
                self.sorted_index.append(population-1)
            for j in range(self.gene_length):
                self.creature[i][j] = random.randint(2, self.maxNode)
        try:
            self.creature[1] = np.loadtxt('structure.txt', dtype=int)
        except:
            pass
    def saveSynGA(self):
        try:
            a = self.creature[i][0];
        except:
            a = -1;
        try:
            b = self.creature[i][1];
        except:
            b = -1;
        try:
            c = self.creature[i][2];
        except:
            c = -1;
        try:
            d = self.creature[i][3];
        except:
            d = -1;
        try:
            e = self.creature[i][4];
        except:
            e = -1;
        self.createNetwork(a,b,c,d,e)
        for gen in range(depth*2):
            self.train(self.think(self.X))
        self.saveSynapse()
    def multiTest(self,depth,i):
        ##test each creature and record the score
        try:
            a = self.creature[i][0];
        except:
            a = -1;
        try:
            b = self.creature[i][1];
        except:
            b = -1;
        try:
            c = self.creature[i][2];
        except:
            c = -1;
        try:
            d = self.creature[i][3];
        except:
            d = -1;
        try:
            e = self.creature[i][4];
        except:
            e = -1;
        self.createNetwork(a,b,c,d,e)
        for gen in range(depth):
            self.train(self.think(self.X))
        var = cp.asnumpy(cp.mean(cp.abs(cp.asarray(self.y)*self.scaler - self.think(self.X)[self.gene_length + 1]*self.scaler)))
        self.creature_error[i] = var
        if(i % 1 == 0):
            print(str(i) + "  :  " + str(var) +"  :  " + str(self.creature[i]), end='\r')
        return var
    def test(self,depth,func):
        start_time = time.time()
        self.creature_error.clear()
        ##initialize lists before concurrency
        for i in range(population):
            #multiTest(population)
            self.creature_error.append(0)
            self.sorted_index.clear()

        ##test each creature and record the score with concurrency
        
        print("Testing "+ str(population) +" creatures\n")
        with multiprocessing.Pool() as pool:
            #results = []
            #for result in tqdm.tqdm(pool.imap_unordered(func, range(population)), total=len(range(population))):
            #    results.append(result)
            results = pool.map(func, range(population))
            #results = func(next(iter(range(population))))
            self.creature_error = results
        #arage the best to worst creature indexes into a list
        #for j in range(population):
        #    for i in range(j,population):
        #        if self.creature_error[i] < self.creature_error[self.sorted_index[j]]:
        #            #i is inserted at the j index
        #            self.sorted_index.insert(j,i)
        #            self.creature_error[i] = 999999999
        ######################################
        #print("________________________________________________________________________________\n")
        #print(self.creature_error)
        #print("________________________________________________________________________________\n")
        self.temp = self.creature_error
        self.temp.sort()
        self.best = self.temp[0]
        for i in range(population):
            if (self.temp[i] in self.creature_error):
                self.sorted_index.append(self.creature_error.index(self.temp[i]))
                if(i > 0):
                    self.creature_error[self.creature_error.index(self.temp[i])] = 999999999
        #print("Best = " + str(temp[0]))
        ########################################
        #print("________________________________________________________________________________\n")
        #print(self.sorted_index)
        #print("________________________________________________________________________________\n")
        #self.saveSynGA()  
    def newGen(self):
        self.maxNode = int(((self.X.shape[0]+1)/(2*(self.X.shape[1]+1)))/self.gene_length)
        if(self.maxNode < 3):
            self.maxNode = 3
        print(self.maxNode)
        self.old_creature = self.creature.copy()
        ###create children for all except the first child_size+parent_size
        for i in range (child_size+parent_size,population):
            ##chose any random parents
            for j in range(self.gene_length):
                mutate = random.randint(1,mutation)
                if mutate == mutation:
                    self.creature[i][j] = random.randint(2, self.maxNode)
                else:
                    self.creature[i][j] =  self.old_creature[self.sorted_index[random.randint(2,population-1)]][j]

        ##select the best parents and propagate them to the next gen
        for i in range(parent_size):
            self.parent[i] = self.old_creature[self.sorted_index[i]].copy()
            self.creature[i] = self.old_creature[self.sorted_index[i]].copy()
        ##create first 2 children from the best 2 parrents
        for i in range(parent_size,child_size+parent_size):
            for j in range(self.gene_length):
                parentIndex = random.randint(0,parent_size-1)
                mutate = random.randint(1,mutation)
                if mutate == mutation:
                    self.creature[i][j] = random.randint(2, self.maxNode)
                else:
                    self.creature[i][j] =  self.parent[parentIndex][j]
    def output(self,loop):
        if loop == "E":
            print(self.creature)
            print(self.maxNode)
            print("Best = " + str(self.best) + " : "+ str(self.creature[self.sorted_index[0]])+"                                                ")
            #print(self.creature_error)
            #print("Best = " + str(self.creature_error[self.sorted_index[0]]))
            self.saveStructure()
            #for i in range(3):
            #    print("Press CTRL-C in the next "+ str(3-i) +" seconds too save and quit", end="\r")
            #    time.sleep(1)
            print("                                                                                 ")
        elif loop == "S":
            #print ("Output:")
            #print (self.think()[4]*self.scaler)
            # print (self.think(self.X)[self.gene_length + 1]*self.scaler)
            # print (cp.mean(self.think(self.X)[self.gene_length + 1]*self.scaler))
            # print ("stuff: "+ str(self.think(self.X)[self.gene_length + 1]*self.scaler))
            print ("Mean Error: " + str(cp.mean(cp.abs(cp.asarray(self.y)*self.scaler) - self.think(self.X)[self.gene_length + 1]*self.scaler))+"\t\t\r")
            return cp.mean(cp.abs(cp.asarray(self.y)*self.scaler) - self.think(self.X)[self.gene_length + 1]*self.scaler)
            
    ###################################### Genetic Algorithm ######################################

    ###################################### Neural Network #########################################
    def createNetwork(self,N1,N2,N3=-1,N4=-1,N5=-1):
        #print(str(N1) + ":" + str(N2) + ":" + str(N3) + ":" + str(N4) + ":" + str(N5))
        global hiddenlayers
        try:
            self.syn0 = cp.loadtxt('syn0.txt', dtype=float)
            self.syn1 = cp.loadtxt('syn1.txt', dtype=float)
            if(N3 < 0):
                hiddenlayers = 2;
                self.syn2 = (cp.loadtxt('syn2.txt', dtype=float)).reshape((N2,1))
            else:
                self.syn2 = (cp.loadtxt('syn2.txt', dtype=float))
                if(N4 < 0):
                    hiddenlayers = 3;
                    self.syn3 = (cp.loadtxt('syn3.txt', dtype=float)).reshape((N3,1))
                else:
                    self.syn3 = (cp.loadtxt('syn3.txt', dtype=float))
                    if(N5 < 0):
                        hiddenlayers = 4;
                        self.syn4 = (cp.loadtxt('syn4.txt', dtype=float)).reshape((N4,1))
                    else:
                        hiddenlayers = 5;
                        self.syn4 = (cp.loadtxt('syn4.txt', dtype=float))
                        self.syn5 = (cp.loadtxt('syn5.txt', dtype=float)).reshape((N5,1))
            #print("L0: "+str(self.syn0.shape)+"\n"+"L1: "+str(self.syn1.shape)+"\n"+"L2: "+str(self.syn2.shape)+"\n"+"L3: "+str(self.syn3.shape)+"\n")
            #print(str(self.syn3))
            cp.set_printoptions(threshold=cp.inf)
            try:
                print("\n")
                # print(str(self.syn0) + "\n\n\n")
                # print(str(self.syn1) + "\n\n\n")
                # print(str(self.syn2) + "\n\n\n")
                # print(str(self.syn3) + "\n\n\n")
                # print(str(self.syn4) + "\n\n\n")
                # print(str(self.syn5) + "\n\n\n")
            except:
                pass
        except:
            # initialize weights randomly with mean 0
            cp.random.seed(0)
            ########## INPUT  Layer 0 ################
            self.syn0 = 2*cp.random.random((self.input_size-1,N1)) - 1
            ########## H Layer 1 ################
            self.syn1 = 2*cp.random.random((N1,N2)) - 1
            ########## H Layer 2 ################
            if(N3 < 0):
                hiddenlayers = 2;
                self.syn2 = 2*cp.random.random((N2,1)) - 1
            else:
                self.syn2 = 2*cp.random.random((N2,N3)) - 1
            ########## H Layer 3 ################
                if(N4 < 0):
                    hiddenlayers = 3;
                    self.syn3 = 2*cp.random.random((N3,1)) - 1
                else:
                    self.syn3 = 2*cp.random.random((N3,N4)) - 1
                ########## H Layer 4 ################
                    if(N5 < 0):
                        hiddenlayers = 4;
                        self.syn4 = 2*cp.random.random((N4,1)) - 1
                    else:
                        hiddenlayers = 5;
                        self.syn4 = 2*cp.random.random((N4,N5)) - 1
                    ########## H Layer 5 ################
                        self.syn5 = 2*cp.random.random((N5,1)) - 1  
            ########## Output Layer 6 ################
            #print("L0: "+str(self.syn0.shape)+"\n"+"L1: "+str(self.syn1.shape)+"\n"+"L2: "+str(self.syn2.shape)+"\n"+"L3: "+str(self.syn3.shape)+"\n")
            #print(str(self.syn3))
    def think(self,input):
        
        layers = []
        # forward propagation
        l0 = cp.asarray(input)
        #print(l0)
        #cp.random.shuffle(l0)
        #print(l0)
        layers.append(l0)
        #print(str(l0.shape)+" : "+str(self.X.shape))

        l1 = nonlin(cp.dot(l0,self.syn0))
        layers.append(l1)
        #l1 = nonlin(scipy.signal.convolve2d(l0,self.syn0, mode='full', boundary='symm', fillvalue=0))
        #print(str(l1.shape)+" : "+str(l0.shape) +" : "+str(self.syn0.shape))

        l2 = nonlin(cp.dot(l1,self.syn1))
        layers.append(l2)
        #l2 = nonlin(scipy.signal.convolve2d(l1,self.syn1, mode='full', boundary='symm', fillvalue=0))
        #print(str(l2.shape)+" : "+str(l1.shape)+" : "+str(self.syn1.shape))

        l3 = nonlin(cp.dot(l2,self.syn2))
        layers.append(l3)
        #l3 = nonlin(scipy.signal.convolve2d(l2,self.syn2, mode='full', boundary='symm', fillvalue=0))
        #print(str(l3.shape)+" : "+str(l2.shape)+" : "+str(self.syn2.shape))
        if (hiddenlayers > 2):
            l4 = nonlin(cp.dot(l3,self.syn3))
            layers.append(l4)
        if (hiddenlayers > 3):
            l5 = nonlin(cp.dot(l4,self.syn4))
            layers.append(l5)
        if (hiddenlayers > 4):
            l6 = nonlin(cp.dot(l5,self.syn5))
            layers.append(l6)
        
        return layers
    def train(self,layers):
        y_gpu = cp.asarray(self.y)
        if (hiddenlayers == 5):
            l6_error = y_gpu - layers[6]
            l6_delta = l6_error*nonlin(layers[6],deriv=True)
        if (hiddenlayers == 4):
            l5_error = y_gpu - layers[5]
            l5_delta = l5_error*nonlin(layers[5],deriv=True)
        elif(hiddenlayers > 4):
            l5_error = l6_delta.dot(self.syn5.T)
            l5_delta = l5_error*nonlin(layers[5],deriv=True)
        if (hiddenlayers == 3):
            l4_error = y_gpu - layers[4]
            l4_delta = l4_error*nonlin(layers[4],deriv=True)
        elif(hiddenlayers > 3):
            l4_error = l5_delta.dot(self.syn4.T)
            l4_delta = l4_error*nonlin(layers[4],deriv=True)
        if (hiddenlayers == 2):
            l3_error = y_gpu - layers[3]
            l3_delta = l3_error*nonlin(layers[3],deriv=True)
        elif(hiddenlayers > 2):
            l3_error = l4_delta.dot(self.syn3.T)
            l3_delta = l3_error*nonlin(layers[3],deriv=True)

        l2_error = l3_delta.dot(self.syn2.T)
        l2_delta = l2_error*nonlin(layers[2],deriv=True)

        l1_error = l2_delta.dot(self.syn1.T)
        l1_delta = l1_error*nonlin(layers[1],deriv=True)

        if (hiddenlayers > 4):
            self.syn5 += layers[5].T.dot(l6_delta*learningRate)
        if (hiddenlayers > 3):
            self.syn4 += layers[4].T.dot(l5_delta*learningRate)
        if (hiddenlayers > 2):
            self.syn3 += layers[3].T.dot(l4_delta*learningRate)
        self.syn2 += layers[2].T.dot(l3_delta*learningRate)
        self.syn1 += layers[1].T.dot(l2_delta*learningRate)
        self.syn0 += layers[0].T.dot(l1_delta*learningRate)
    def Menu():
        print("CTRL-C opens the exit menu")
        print("Train Neural Net = S")
        print("Evolve Nerual Net Structure = E")
        print("Ask Network = A")
        loop = input("Input Your Selection: ")
        return loop
    ###################################### Neural Network #########################################
    def gety(self):
        return self.y
    def saveSynapse(self):
        #try:
            #print(str(self.syn0) + "\n\n\n")
            #print(str(self.syn1) + "\n\n\n")
            #print(str(self.syn2) + "\n\n\n")
            #print(str(self.syn3) + "\n\n\n")
            #print(str(self.syn4) + "\n\n\n")
            #print(str(self.syn5) + "\n\n\n")
        #except:
            #pass
        try:
            os.remove("syn0.txt")
            os.remove("syn1.txt")
            os.remove("syn2.txt")
            os.remove("syn3.txt")
            os.remove("syn4.txt")
            os.remove("syn5.txt")
        except:
            pass
        try:
            os.remove("guesses.xlsx")
        except:
            pass
        df = pd.DataFrame(cp.asnumpy(self.think(self.X)[self.gene_length + 1]*self.scaler))
        filepath = 'guesses.xlsx'
        df.to_excel(filepath, index=False)
        #global synanpse
        #synapse = np.array([syn0, syn1, syn2, syn3],dtype=object)
        cp.set_printoptions(threshold=np.inf)
        cp.savetxt('syn0.txt', self.syn0, fmt='%1.8f')
        cp.savetxt('syn1.txt', self.syn1, fmt='%1.8f')
        cp.savetxt('syn2.txt', self.syn2, fmt='%1.8f')
        if self.gene_length >2:
            cp.savetxt('syn3.txt', self.syn3, fmt='%1.8f')
            if self.gene_length >3:
                cp.savetxt('syn4.txt', self.syn4, fmt='%1.8f')
                if self.gene_length >4:
                    cp.savetxt('syn5.txt', self.syn5, fmt='%1.8f')
    def saveStructure(self):
        try:
            os.remove("structure.txt")
        except:
            pass
        neurons = self.creature[0]
        np.savetxt('structure.txt', neurons, fmt='%d')
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
                    neurons = np.loadtxt('structure.txt', dtype=int)
                    try:
                        a = neurons[0];
                    except:
                        a = -1;
                    try:
                        b = neurons[1];
                    except:
                        b = -1;
                    try:
                        c = neurons[2];
                    except:
                        c = -1;
                    try:
                        d = neurons[3];
                    except:
                        d = -1;
                    try:
                        e = neurons[4];
                    except:
                        e = -1;
                    self.createNetwork(a,b,c,d,e)
                    layers = []
                    userin = cp.empty((1,self.input_size-1),float)
                    for i in range(self.input_size-1):
                        userin[0][i] = float(input(str(self.column_names[i]) +": "))
                    l0 = userin/self.scalerx
                    layers.append(l0)
                    l1 = nonlin(cp.dot(l0,self.syn0))
                    layers.append(l1)
                    l2 = nonlin(cp.dot(l1,self.syn1))
                    layers.append(l2)
                    l3 = nonlin(cp.dot(l2,self.syn2))
                    layers.append(l3)
                    if c!=-1:
                        l4 = nonlin(cp.dot(l3,self.syn3))
                        layers.append(l4)
                        if d!=-1:
                            l5 = nonlin(cp.dot(l4,self.syn4))
                            layers.append(l5)
                            if e!=-1:
                                l6 = nonlin(cp.dot(l5,self.syn5))
                                layers.append(l6)
                    
                    print ("Calculated: " + str(layers[len(layers) -1]*self.scaler))

                    again = input("press enter to continue or input \"Y\" to ask again:")
            elif choice == "S":
                self.saveSynapse()
        elif loop =="E":
            save = input("press enter to continue or input \"Y\" to Save Nerual Net Structure: ")
            if save == "Y":
                self.saveStructure()
                running = False
        else:
            sys.exit()
    def ask(self):
        while True:
            neurons = np.loadtxt('structure.txt', dtype=int)
            try:
                a = neurons[0];
            except:
                a = -1;
            try:
                b = neurons[1];
            except:
                b = -1;
            try:
                c = neurons[2];
            except:
                c = -1;
            try:
                d = neurons[3];
            except:
                d = -1;
            try:
                e = neurons[4];
            except:
                e = -1;
            self.createNetwork(a,b,c,d,e)
            layers = []
            userin = cp.empty((1,self.input_size-1),float)
            for i in range(self.input_size-1):
                userin[0][i] = float(input(str(self.column_names[i]) +": "))
            l0 = userin/self.scalerx
            layers.append(l0)
            l1 = nonlin(cp.dot(l0,self.syn0))
            layers.append(l1)
            l2 = nonlin(cp.dot(l1,self.syn1))
            layers.append(l2)
            l3 = nonlin(cp.dot(l2,self.syn2))
            layers.append(l3)
            if c!=-1:
               l4 = nonlin(cp.dot(l3,self.syn3))
               layers.append(l4)
               if d!=-1:
                   l5 = nonlin(cp.dot(l4,self.syn4))
                   layers.append(l5)
                   if e!=-1:
                       l6 = nonlin(cp.dot(l5,self.syn5))
                       layers.append(l6)
            
            print ("Calculated: " + str(layers[len(layers) -1]*self.scaler))