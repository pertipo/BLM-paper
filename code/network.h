#ifndef BLM_NETWORK
#define BLM_NETWORK

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include <string>
#include <sstream>
#include <random>

namespace net {
    //TODO handle public fragments of classes

    //enumeration of the transfer functions implemented above. Used as a parameter for initialization
    enum Transfer {
        Logistic,
        HyperbolicTangent,
        Arctangent,
        Same
    };

    //utility class containing all info on the network structure and command parameters
    class Init {
    public:
        //random seed used in the generation of weights in the initialization phase
        int r_seed;
        //transfer function for each layer --> applied on Output values --> one for each hidden layer and one for the putput layer (input layer has "same" function)
        std::vector<Transfer> atransf;
        //number of neurons in each hidden layer --> the number of neurons in the input/output layers depends on the structure of the samples inputs and labels
        std::vector<unsigned> n_neur;
        //maximum number of bits used in discretization of weights in each layer --> its size is equal to the number of hidden layers plus one (for the output layer)
        std::vector<unsigned> d_bits;
        //number of hidden layer in the network
        unsigned n_h_l;
        //inital number of bits in discretization of weights
        unsigned ini_bits;
        //maximum time for the training phase
        unsigned time;
        //maximum number of iterations for the training phase
        unsigned max_iter;
        //fraction of nonzero weights requested in initialization phase
        float ini_dens;
        //threshold value for initial values of weights --> the range is -w_ini < weight values < +w_ini
        float w_ini;
        //threshold value for weight values during training --> the range is -w_range < weight values < +w_range
        float w_range;
        //fraction of samples used in the training phase
        float tr_dens;
        //determines if weight values are initialized completely randomly (with the only limit given by the maximum discretization value representable) (if true)
        //or by taking into account the w_ini parameter (if false)
        bool full_rand_ini;
        //determines if the fraction of nonzero weights (specified in the ini_dens parameter) is conserved during training (if true) or not (if false)
        bool keep_dens;
        //determines if the training follows a telescopic approach (if true) 
        //or cant modify every bit of the discretized representation from the start (if false)
        bool telescopic;
        //determines if, in the training phase, the improving move at each iteration is the besst one (if true)
        //or the first improving one (if false)
        bool d_all;
        //determines if the network structure (with the best weights found) is saved (if true) or not (if false)
        bool save_structure;
        //determines if the network structure is loaded from file (if true) or initialized from scratch (if false)
        bool load_structure;
        //determines if the program will complete a training phase (if true) or just evaluate the current structure (if false)
        bool train;
        //input and output files paths
        std::string in_example_f;
        std::string in_struct_f;
        std::string in_train_test_f;
        std::string out_results_f;
        std::string out_struct_f;

        //initialization function for the INit class that creates it from a command file
        Init(std::string path);
    };

    //class that stores the output values for each neuron, both the partial values (obtained by the dot product with the weights) and the transfer function
    class Output {
    public:
        //temporary partial values obtained, during training, after a possible weight change --> used as a test value
        std::vector<float> tmp_partials;
        //best partial values obtained by the model so far
        std::vector<float> best_partials;
        //transfer function used to convert partial values into definitive outputs
        float (*transfer)(float);

        //function that returns the final outputs of the neuron
        //best parameter determines if the function will return the temporary outputs or the best ones
        std::vector<float> values(bool best);
    };

    //class used to contain the values of each weight
    class Weight {
    public:
        //discretization value of the weight used for training
        int discrete;
        //floating consant to obtain actual weight values
        float constant;

        //initialization function based on the layer position
        void init(Init* in, int layer);
        //function to return the actual value of the weight used in the calculations
        float value();
        //function to flip a specific bit of the discretization value (in gray code)
        //limit is needed to keep track of negative numbers and sign change == is the bit limit for the representation
        void bitFlip(unsigned bit, unsigned limit);
    };

    //Structure used to contain the info on a single sample
    struct Example {
        //values of each feature for the specific sample
        std::vector<float> input;
        //values of each expected outcome for the sample
        std::vector<float> label;
    };
    //class that summarizes the entire samples set
    class ExampleSet {
    public:
        //the complete set of examples coded as Example instances
        std::vector<Example> examples;

        //initialization function based on the info contained in an Init instance
        //the training_test parameter is to decide if the set will be taken from the examples or testing file
        ExampleSet(Init* in, bool training_test);
        //function that converts to a vector of Output used in the initialization of the network to insert the samples in the first neuron layer
        std::vector<Output> toOutputs();
    };

    //class that represents a neuron of the network
    class Neuron {
    public:
        //vector contaning every weight of the neuron and its associated previous neuron (in the form of a reference) --> input layer will have an empty vector
        //in the last element of the vector is the parameter --> it will have nullptr as a previous neuron
        std::vector<std::tuple<Weight, Neuron*>> weights_and_inputs;
        //outputs of the neuron
        Output outputs;

        //function to compute the partial output of the neuron 
        //if the network has been already initialized only the temporary partial values will be computed
        //otherwise the values will be copied in the best temporary as well
        void calculate();
    };

    //class that summirizes the network structure and its functionalities
    class Network {
    public:
        //two-dimensional vector that describes the neuron disposition in the network
        //it is composed of input layer (at index 0), hidden layers and output layer (at index size-1)
        //each layer, represented as a vector of neurons, has a specific size
        std::vector<std::vector<Neuron>> neurons;
        //vector containing the discretization bit limits for each layer 
        //first hidden layer at index 0, output layer at index size-1
        std::vector<unsigned> bit_limits;
        //vector containing the current number of discretization bits modifiable by the training function
        //if not telescopic mode current_bits == bit_limits
        std::vector<unsigned> current_bits;
        //range of the weight values accepted in the neurons
        float w_range;

        //initialization function of the network
        void init(Init* in, ExampleSet* ex_set);
        //compute the result of the current configuration of the network for each input sample
        //it is possible to insert a strating position (usefull to reduce number of computation in training phase when a weight is changed)
        //if nullptr is passed as position the entirety of the computation will be made
        //change in weight is the difference in value of the new weight (it's used in the partial update mode)
        std::vector<std::vector<float>> eval(int* start_position, float change_in_weight);
        //function that changes the weight at a given position using the bitflip function of the weight class
        //output is the value of the weight change
        float change(int* position, unsigned bit);
        //function that updates the best partial outputs of each neuron starting from a given position
        void updateOutput(int* position);
        //function that switches the current inputs with the one given as parameter
        void switchInputs(ExampleSet* ex_set);
        //function that saves the fundamental components of the network structure to the specified json file and the outputs of the network to the specified .exa file
        //layout of saved data is specified in the documentation
        void saveToFile(Init* in, ExampleSet* ex_set);
        //function that extracts info from a network structure json file and initializes network to match the requirements
        void loadFromFile(Init* in, ExampleSet* ex_set);
    };
}

#endif
