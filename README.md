# BLM IMPLEMENTATION
This repo contains the implementation of the telescopic BLM model, proposed by Mauro Brunato and Roberto Battiti, in the paper found at: https://rtm.science.unitn.it/~battiti/archive/battiti-blm-ieee-tnnls.pdf. This implementaion was made by Gabriele Pernici and is based on a previous implementation by Mauro Brunato and Roberto Battiti.
For more info on the implementation consult the manual in the doc folder.

# USAGE
To use the program the folder must be dowloaded first (using the apposite gitHub button).
After extracting the folder from the compressed archive, use the make command in order to compile the program.
After compilation the "main" executable will be produced.
To correctly start the program a ".cmd" file path must be prvided as parameter when launching "main" from command line.
The precise definition and structure of the ".cmd" file is described in the manual located in the doc folder.

# EXAMPLES
Here are reported general and incomplete examples, for more detailed ones see the folders named _SomethingExample_.
At the moment this repo contains: 
* SpiralExample = the double spiral recognition problem.

Generic examples:

Example of a .cmd file to initialize a network, train it and save its structure
```
r_seed 3
n_h_l 2
atransf 0 0 0
n_neur 20 20
d_bits 4 4 4
ini_bits 2
w_ini 0.5
w_range 6.5
train
telescopic
in_example_f ../Test.exa
out_results_f ../TestResults.exa
out_struct_f ../TestStruct.json
save
```

Example of a .exa file containing labeled examples
```
n_inp 2
n_out 1
n_patt 10
pattern 3.0 0.3 1
pattern 3.1 1.3 0
pattern 3.2 2.3 1
pattern 3.3 3.3 0
pattern 3.4 4.3 1
pattern 3.5 5.3 0
pattern 3.6 6.3 1
pattern 3.7 7.3 0
pattern 3.8 8.3 1
pattern 3.9 9.3 0
```
