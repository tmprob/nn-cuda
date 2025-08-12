# Template  CUDA Project #

Project provides an API for building, training and evaluating a Neural Network with possibility to use CUDA. The main features follow the [.

**Getting started**

This template is build with a [Makefile](Makefile). To build first create a build folder and then use the Makefile:
```bash
mkdir build && make
```

Note: the program compile showing two warnings, this is due to the currently partially missing implememntation of the CUDA gradiendescent.cu function. The code compile and can run the serial version anyway

Source code is located in the [source](source/) folder.

For help on the make targets use:
```bash
make help
```
**Examples**

The [examples](examples/) folder contains the source code for the excecutables. They give an outline on how to use the API of this project. They are build in this folder and can be excecuted, e.g.:
```bash
main
```

First example [main](examples/main.cu): Demonstrates the usage of a simple Neural Network on a moon data set with 2 inputs and 2 labels. Uses the assignment data (input.csv and output.csv). 

The code has a structure to implement a CUDA version for a GPU parallelization, anyway a serial working version is available.

**Documentation**

The code is documented with inline comments.
