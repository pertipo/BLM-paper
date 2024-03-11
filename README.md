# BLM IMPLEMENTATION
This repo contains the implementation of the telescopic BLM model, proposed by Mauro Brunato and Roberto Battiti, in the paper found at: https://rtm.science.unitn.it/~battiti/archive/battiti-blm-ieee-tnnls.pdf. 
For more info on the implementation consult the manual in the doc folder.

#USAGE
To use the program the folder must be dowloaded first (using the apposite gitHub button).
After extracting the folder from the compressed archive, use the make command in order to compile the program.
The compilation process will also assert the presence of the nlohmann-json library and, if absent, install it.
After compilation the "main" executable will be produced.
To correctly start the program a .cmd file path must be prvided as parameter when launching "main" from command line.
The precise definition and structure of the .cmd file is described in the manual located in the doc folder.
