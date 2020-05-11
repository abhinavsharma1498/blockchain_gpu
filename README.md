# Blockchain Mining using GPU
The project aims to provide minign solution for CPU as well as GPU.

# Build Status
Successful build

# Hardware requirements
Any reasonable CPU: (For CPU as well as GPU mining)
(tested with Intel i5-5200U, Intel i5-6200U)

IF YOU INTEND TO MINE ON GPU, NVIDIA CARD WITH CUDA IS REQUIRED.
(Tested with Nvidia GeFroce 920M, Nvidia Tesla K80)

Drivers version compatible with CUDA version required.

# Libraries required
Python headers:
  1. flask
  2. requests
  3. uuid
  4. argparse
  5. datetime
  6. hashlib
  7. urllib
  8. subprocess
  9. time

C++ headers:
1. cuda.h
2. string.h
3. stlib.h
4. stdio.h

# Building the GPU Code
To build the GPU Code, run the following command:
  nvcc main.cu -o run
 
#  Executing the program
Place blockchain.py, crypto.py and run in the same folder.
Now, from that directory, run the following commands:

  python3 crypto.py -p PORT_NO [-d DIFFICULTY | -D DEVICE]

Desctiption of parameters:
  1. -p: Define the port number to run the server on, REQUIRED
  2. -d: Difficulty of mining, DEFAULT = 2
  3. -D: Device on which you intend to mine, DEFAULT = 'CPU', accepted options = 'CPU' | 'GPU'

# Project Contributers
Abhinav Sharma
Rohan Deoli
Amish Tandon
Ankit Jangwan

# Project Guide
Akshay Rajput sir
