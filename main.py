from asyncio.windows_events import NULL
from EvolvingNN import EvolvingNN
import signal
import numpy as np
import time
from functools import partial
import turtle
import wmi
def exit(signum, frame):
    NN.exit(loop)
def startProgram():
    list = ['Width','Offset','Box']
    NN = EvolvingNN('file.xlsx',list)
    #list = ['Experience_level','employment_type','remote_ratio','company_size','salary_in_usd']
    #NN = CCEvolvingNN('ds_salaries.xlsx',list)
    return NN
if __name__ == "__main__":
    best = 999999999999
    while True:
        loop = EvolvingNN.Menu()
        signal.signal(signal.SIGINT, exit)
        while True:
            if loop == "S":
                if not ('NN' in locals()):
                    NN = startProgram()
                try:
                    ##load the saved structure and create network under that stucture
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
                    NN.createNetwork(a,b,c,d,e)

                except:
                    print("COULD NOT LOAD, USING DEFAULT")
                    NN.createNetwork(66,24,3)
                while True:
                    for gen in range(100):
                        NN.train(NN.think(NN.X))
                    temp = NN.output(loop)
                    if (abs(temp) < abs(best)):
                        NN.saveSynapse()
                        best = temp
                        print("################################################################################################Synapses Saved")
            elif loop == "E":
                #depth is how many iterations to run each NN test when evolving population
                try:
                    depth = int(input("how long will the creature live?(default is 100)"))
                except:
                    depth = 100
                try:
                    gens = int(input("hown many generation until network is trained?(default is 100)"))
                except:
                    gens = 100
                NN = startProgram()
                NN.spawn()
                for i in range(gens):
                    #start_time = time.time()e
                    print("Generation: " + str(i))
                    # Eefunc = partial(NN.multiTest,depth)
                    NN.test(depth)
                    NN.newGen()
                    NN.output(loop)
                    #print(str(time.time() - start_time))
                    if(i%4 == 0):
                        for j in range(10):
                            w = wmi.WMI(namespace="root\wmi")
                            temperature_info = w.MSAcpi_ThermalZoneTemperature()[0]
                            print (temperature_info.CurrentTemperature)
                            print("Cooling CPU for " +str(10-j)+" seconds", end='\r')
                            time.sleep(1)
                loop = "S"
            elif loop == "A":
                NN = startProgram()
                NN.ask()
            else:
                print("\nMaybe try a valid input this time....\n")
                loop = NN.Menu()
