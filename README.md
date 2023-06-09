

# EvolvingNeuralNet

A Python based implementation of a Neural Network (NN) optimized by a Genetic Algorithm (GA).

This project involves creating a flexible, multi-layer neural network that is initialized either with saved weights or random weights. The neural network is optimized by a genetic algorithm which allows the system to learn and make predictions based on the input data. The system is built with flexibility allowing variable number of hidden layers and neurons.

## Installation and Setup

This project is written in Python, and it requires the following Python libraries:

 - Numpy 
 - Pandas 
 - Cupy (for GPU computations)

You can install these packages via pip:

    pip install numpy pandas cupy

## Usage

After cloning the repository and installing dependencies, you can run the main script by the following command:

    python main.py

## Features

 - The Neural Network can be configured with a variable number of hidden
   layers and neurons. 
 - The system can save and load the synapse weights,    allowing
   continuous learning.
 - The system provides a terminal-based    interactive menu to train the
   network, evolve its structure, or ask    the network.

  

 - The use of a genetic algorithm allows for a more evolved    and
   optimal learning process.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Disclaimer

This project is a work in progress and might not be entirely free of bugs. Use it at your own risk. Any feedback, issue reporting and contributions to fix these are highly appreciated.
