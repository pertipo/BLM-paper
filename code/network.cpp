#include "network.h"

using namespace net;

//global random number generator
//it will be initialized with a seed in the network initialization phase
std::mt19937 rng;

//Transfer function implemented and selectable
//same transfer is used ofr the input layer (as its data need to be unchanged) and as a default in case of errors
float same(float in) { return in; }
//logistic transfer is approximated to, 0 or 1, without the actual computations of a logistic function if the input is outside the boundaries of the defined thresholds
//this can be applied as the output of a logistic function tends to 1 (0) if the input becomes really big (small).
float logistic(float in) { return (1.0 / (1.0 + exp(-in))); }
float hyperTan(float in) { return std::tanh(in); }
float arctangent(float in) { return std::atan(in); }

//INIT CLASS IMPLEMENTATION
//initialization function for the INit class that creates it from a command file
Init::Init(std::string path) {
    //open the command file and prepare to extract the usefull parameters
    std::ifstream in(path); //TODO check if file is readable and other controls on parameters
    //map used to save all parameters present in the command file along with their value
    //a map is used as it allows to handle eventual repetitions or useless strings in the command file
    std::map<std::string, std::vector<float>> directives;
    //map similar to the previous used to save the input/output files paths
    std::map<std::string, std::string> files;
    //buffer used to exctract a line at a time from the command file and successively a word at a time
    std::string buff;
    //single parameter name used as key for the previous maps
    std::string directive;
    //values of a single parameter to be added to the directives map
    std::vector<float> param;
    //boolean flag used to check if the selected word is a directive or a value
    bool first = true;
    //boolean flag used to keep track if a file or a directive needs to be inserted in a map
    bool file = false;

    //cycle that analyzes the command file a line at a time
    while (std::getline(in, buff)) {
        //stringstream is used to separate words and remove spaces
        std::stringstream T(buff);
        //reset of boolean flags and parametric values
        first = true;
        file = false; //it is assumed that the default parameter will not be a file
        param.clear();

        //cycle that analyzes each word in a single line to determine the key-value pair to insert in the correct map
        while (std::getline(T, buff, ' ')) {
            if (first) {
                //if the considered word is the fist in the line it is the parameter's name (a.k.a. the directive name)
                //set the first flag to false adn save the current word as the directive
                first = false;
                directive = buff;
            }
            else {
                //if the considered word is not the first it is either a file path or a numerical value associated with the directive
                try {
                    //if the word is castable to a float it is obviously a value
                    param.push_back(std::stof(buff));
                }
                catch (std::invalid_argument a) {
                    //if the word is not castable (i.e. it is a string) it is a file path
                    //insert the directive-path tuple in the files map and set the file flag to true to avoid saving the tuple in the directives map (would cause an error) 
                    files.insert({ directive, buff });
                    file = true;
                }
            }
        }
        //after the entire line is analyzed, if the directive wasn't saved in the files map, the directive-param tuple must be inserted in the directives map
        if (!file) { directives.insert({ directive, param }); }
    }
    in.close();

    //All lines of the command file have been analyzed
    //Now the init object's parameters must be set in accordance with the extracted directives

    //There are two possibilities: either the network is loaded from file or it is initialized from scratch
    //Each mode needs different parameters, however some are in common

    //Each parameter's name will be searched in the appropiate map and, if pressent, its corresponding init attribute will be initialized in accordance with the value
    //Some parameters are compulsory, while the others have default values

    //load parameter is first as it's needed to determine the mode of initialization and the different required parameters
    auto current_dir = directives.find("load");
    this->load_structure = (current_dir != directives.end()) ? true : false; //default is initialization mode

    //common parameters specifiable for init (both for load and init)
    current_dir = directives.find("ini_bits");
    this->ini_bits = (current_dir != directives.end()) ? *current_dir->second.begin() : -1; //if UINT_MAX discretization starts at d_bits at each layer     

    current_dir = directives.find("time");
    this->time = (current_dir != directives.end()) ? *current_dir->second.begin() : -1; //if UINT_MAX there's no time limit

    current_dir = directives.find("max_iter");
    this->max_iter = (current_dir != directives.end()) ? *current_dir->second.begin() : -1; //if UINT_MAX there's no iteration limit

    current_dir = directives.find("tr_dens");
    this->tr_dens = (current_dir != directives.end()) ? *current_dir->second.begin() : 1; //default is all training samples used

    current_dir = directives.find("keep_dens");
    this->keep_dens = (current_dir != directives.end()) ? true : false; //default is to not keep density

    current_dir = directives.find("telescopic");
    this->telescopic = (current_dir != directives.end()) ? true : false; //default is to avoid telescopic search

    current_dir = directives.find("d_all");
    this->d_all = (current_dir != directives.end()) ? true : false; //default is to look for the first improving move

    current_dir = directives.find("save");
    this->save_structure = (current_dir != directives.end()) ? true : false; //default is not to save structure

    current_dir = directives.find("train");
    this->train = (current_dir != directives.end()) ? true : false; //default is to only evaluate

    auto current_file = files.find("in_example_f"); //compulsory parameter == if absent throw error
    if (current_file == files.end()) {
        std::cerr << "ERROR" << std::endl << "Compulsory parameter \"in_example_f\" omitted in command file: " << path << std::endl;
        exit(1);
    }
    this->in_example_f = current_file->second;

    current_file = files.find("in_train_test_f");
    this->in_train_test_f = (current_file != files.end()) ? current_file->second : ""; //if absent is left empty

    current_file = files.find("out_results_f");
    this->out_results_f = (current_file != files.end()) ? current_file->second : "./out.exa"; //default path for results

    current_file = files.find("out_struct_f");
    this->out_struct_f = (current_file != files.end()) ? current_file->second : "./net.json"; //default path for structure save

    if (!this->load_structure) { //standard initiazlization == various compulsory parameters + some optional ones
        current_dir = directives.find("r_seed");
        this->r_seed = (current_dir != directives.end()) ? *current_dir->second.begin() : 1; //default is seed==1

        current_dir = directives.find("n_neur"); //compulsory parameter == if absent throw error
        if (current_dir == directives.end()) {
            std::cerr << "ERROR" << std::endl << "Compulsory parameter \"n_neur\" omitted in command file: " << path << std::endl;
            exit(1);
        }
        for (auto i = current_dir->second.begin(); i != current_dir->second.end(); i++) {
            this->n_neur.push_back((unsigned)*i);
        }

        current_dir = directives.find("d_bits"); //compulsory parameter == if absent throw error
        if (current_dir == directives.end()) {
            std::cerr << "ERROR" << std::endl << "Compulsory parameter \"d_bits\" omitted in command file: " << path << std::endl;
            exit(1);
        }
        for (auto i = current_dir->second.begin(); i != current_dir->second.end(); i++) {
            this->d_bits.push_back((unsigned)*i);
        }

        current_dir = directives.find("n_h_l"); //compulsory parameter == if absent throw error
        if (current_dir == directives.end()) {
            std::cerr << "ERROR" << std::endl << "Compulsory parameter \"n_h_l\" omitted in command file: " << path << std::endl;
            exit(1);
        }
        this->n_h_l = *current_dir->second.begin();

        current_dir = directives.find("atransf"); //default transfer function at each layer is same
        if (current_dir == directives.end()) {
            for (unsigned i = 0; i <= this->n_h_l; i++) {
                atransf.push_back(Logistic);
            }
        }
        else {
            for (unsigned i = 0; i <= this->n_h_l; i++) {
                if (i >= current_dir->second.size()) {
                    atransf.push_back(Logistic);
                }
                else {
                    switch ((int)current_dir->second[i]) {
                    case 0:
                        atransf.push_back(Logistic);
                        break;
                    case 1:
                        atransf.push_back(HyperbolicTangent);
                        break;
                    case 2:
                        atransf.push_back(Arctangent);
                        break;
                    default:
                        atransf.push_back(Same);
                        break;
                    }
                }
            }
        }

        current_dir = directives.find("ini_dens");
        this->ini_dens = (current_dir != directives.end()) ? *current_dir->second.begin() : 1; //default is to have no zero weight

        current_dir = directives.find("w_ini"); //compulsory parameter == if absent throw error
        if (current_dir == directives.end()) {
            std::cerr << "ERROR" << std::endl << "Compulsory parameter \"w_ini\" omitted in command file: " << path << std::endl;
            exit(1);
        }
        this->w_ini = *current_dir->second.begin();

        current_dir = directives.find("w_range"); //compulsory parameter == if absent throw error
        if (current_dir == directives.end()) {
            std::cerr << "ERROR" << std::endl << "Compulsory parameter \"w_range\" omitted in command file: " << path << std::endl;
            exit(1);
        }
        this->w_range = *current_dir->second.begin();

        current_dir = directives.find("full_rand_ini"); //compulsory parameter == if absent throw error
        this->full_rand_ini = (current_dir != directives.end()) ? true : false;

    }
    else { //network initialization from file == some compulsory parameters (for the std init) are optional + one compulsory parameter needed
        current_file = files.find("in_struct_f"); //compulsory parameter == if absent throw error
        if (current_file == files.end()) {
            std::cerr << "ERROR" << std::endl << "Compulsory parameter \"in_struct_f\" omitted in command file: " << path << std::endl;
            exit(1);
        }
        this->in_struct_f = current_file->second;

        current_dir = directives.find("w_range"); //default for weight range is the one saved in the structure file
        if (current_dir != directives.end()) {
            this->w_range = *current_dir->second.begin();
        }
        else {
            this->w_range = 0;
        }

        current_dir = directives.find("d_bits"); //default for max bits are the ones saved in the structure file
        if (current_dir != directives.end()) {
            for (auto i = current_dir->second.begin(); i != current_dir->second.end(); i++) {
                this->d_bits.push_back((unsigned)*i);
            }
        }
    }
}

//OUTPUT CLASS IMPLEMENTATION
//function that returns the final outputs of the neuron
//best parameter determines if the function will return the temporary outputs or the best ones
std::vector<float> Output::values(bool best) {
    //vector to store the outputs for each sample
    std::vector<float> res;

    if (best) {
        //compute the values for the best partials
        //for each partial output (one for each sample) compute its final value using the transfer function
        for (auto p = this->best_partials.begin(); p != this->best_partials.end(); p++) {
            res.push_back(transfer(*p));
        }
    }
    else {
        //compute the values for the temporary partials
        //for each partial output (one for each sample) compute its final value using the transfer function
        for (auto p = this->tmp_partials.begin(); p != this->tmp_partials.end(); p++) {
            res.push_back(transfer(*p));
        }
    }
    return res;
}

//WEIGHT CLASS IMPLEMENTATION
//initialization function based on the layer position
void Weight::init(Init* in, int layer) {
    int max_repr_with_dbits = pow(2, (in->d_bits[layer] - 1)) - 1;
    //calculate the constant value based on weight range and maximum number of discretization bits
    float c_max = (in->w_range) / max_repr_with_dbits;

    this->constant = c_max;

    // /* NEW RNG SYSTEM */
    // //initialize the normal distribution for the ini_dens check
    // std::uniform_int_distribution<> random_percentage(1,100);
    // //generate the value for the discrete part of the weight
    // //statistically generate a "ini_dens" number of nonzero values
    // if(random_percentage(rng) <= ((in->ini_dens)*100)) {
    //     //randomly generate the initial value of the discrete value
    //     if(in->telescopic) {
    //         //if the telescopic mode is requested the discrete weights will be initialized randomly but with a logic
    //         //only the n most significant bits may be set to 1
    //         //where n is the number of starting bits

    //         //ignore the least significant bits and perform a successive l_shift (also ignore the sign bit)
    //         int max = pow(2, in->ini_bits-1) -1;
    //         std::uniform_int_distribution<> abs_significant(1, max);
    //         int abs_value = abs_significant(rng) << (in->d_bits[layer] - in->ini_bits);

    //         //consider the bit sign (as it is the most significant one it's always considered)
    //         //randomly determine the sign of the weight with equal probability
    //         this->discrete = (random_percentage(rng) > 50)? abs_value : -abs_value;
    //     } else {
    //         if(in->full_rand_ini) {
    //             //in full random mode the only constraint is the max number of bits in the representation
    //             int max = max_repr_with_dbits;
    //             std::uniform_int_distribution<> full_rand_dist(1, max);
    //             int abs_value = full_rand_dist(rng);
    //             this->discrete = (random_percentage(rng) > 50)? abs_value : -abs_value;
    //         } else { 
    //             //in non full random mode the constraint becomes the w_ini parameter 
    //             //w_ini is a limit of the actual weight, so also the constant must be taken into account
    //             //also the maximmum number of discretization bits must be taken into account
    //             int large_range = ceil((in->w_ini)/(this->constant));
    //             int max = (large_range < max_repr_with_dbits)? large_range : max_repr_with_dbits;
    //             std::uniform_int_distribution<> ini_w_dist(1, max);
    //             int abs_value = ini_w_dist(rng);
    //             this->discrete = (random_percentage(rng) > 50)? abs_value : -abs_value;
    //         }
    //     }
    // } else {
    //     //set discretization value to 0 to obtain a zero weight (not permanent as the discrete value might change in training)
    //     this->discrete = 0;
    // }

    /* OLD RNG SYSTEM */
    if (in->ini_dens >= 1.0 || rand() / (float)RAND_MAX < in->ini_dens) {
        //randomly generate the initial value of the discrete value
        if (in->full_rand_ini) {
            //in full random mode the only constraint is the max number of bits in the representation
            this->discrete = std::rand() % (2 * (int)pow(2, (in->d_bits[layer] - 1))) - pow(2, (in->d_bits[layer] - 1));
        }
        else {
            //in non full random mode the constraint becomes the w_ini parameter 
            //w_ini is a limit of the actual weight, so also the constant must be taken into account
            this->discrete = std::rand() % (2 * (int)ceil((in->w_ini) / (this->constant)) + 1) - ceil((in->w_ini) / (this->constant));
        }
    }
    else {
        //set discretization value to 0 to obtain a zero weight (not permanent as the discrete value might change in training)
        this->discrete = 0;
    }
}
//function to return the actual value of the weight used in the calculations
float Weight::value() { return discrete * constant; }
//function to flip a specific bit of the discretization value (in gray code)
//limit is needed to keep track of negative numbers and sign change == is the bit limit for the representation
void Weight::bitFlip(unsigned bit, unsigned limit) {
    //NB: you save number in normal binary, but you want to do a bit flip in gray code
    //So the idea is to get to the correct number in gray code 
    //(e.g. flipping the third bit from 2 you need to get to 5 == from 0011 to 0111 in gray, while you start with 0010 and need to get to 0101)

    //need to handle negative gray code
    //the idea is to make the negative number positive, perform the normal bitflip and change the sign once again

    //need to handle the bitflip on the last representable bit == sign bit (as it  depends on the maximum bit representation the system is using)
    //just flip the number's sign

    //the representation of negative numbers in gray code obtained by this logic is specular to the positive one, with only the sign bit differentiating a positive and negative number
    //for example 2 will be represented as 0011, while -2 will be represented as 1011 (if the maximum bit number is 4)
    //this means that 0 is represented twice, but it should be fine

    if (bit + 1 == limit) {
        this->discrete = ~this->discrete;
        return;
    }
    this->discrete = this->discrete ^ ((1 << bit + 1) - 1);
}

//EXAMPLESET CLASS IMPLEMENTATION
//initialization function based on the info contained in an Init instance
//the training_test parameter is to decide if the set will be taken from the examples or testing file
ExampleSet::ExampleSet(Init* in, bool training_test) {
    //path of the file containing the example set
    std::string ex_f = (training_test) ? in->in_train_test_f : in->in_example_f;
    //open the example file
    std::ifstream inp(ex_f);
    //temporary buffers used to store the components of each example
    std::string buff1, buff2;
    //variable containing the number of input features
    unsigned n_inp;
    //variable containing the number of samples
    unsigned n_patt;
    //variable containing the number of expected values for each sample
    unsigned n_out;
    //counter used to determine if a value is an input feature or an output value
    unsigned counter;
    //flag used to exclude the first word each sample line (a.k.a. "pattern")
    bool first;

    //get the n_inputs parameter
    //get first line and convert it into a stringstream to extract info easily
    std::getline(inp, buff1);
    std::stringstream T1(buff1);
    //first line structure is "n_inp value" --> only the value is needed --> ignore first word before ' '
    std::getline(T1, buff2, ' ');
    std::getline(T1, buff2, ' ');
    //store the value in the correct variable
    n_inp = std::stoi(buff2);

    //get the n_outputs parameter
    //the same method as n_inputs is used 
    std::getline(inp, buff1);
    std::stringstream T2(buff1);
    std::getline(T2, buff2, ' ');
    std::getline(T2, buff2, ' ');
    n_out = std::stoi(buff2);

    //get the n_patterns parameter
    //the same method as n_inputs is used
    std::getline(inp, buff1);
    std::stringstream T3(buff1);
    std::getline(T3, buff2, ' ');
    std::getline(T3, buff2, ' ');
    n_patt = std::stoi(buff2);

    //get each sample from the rest of the file
    //on each line there's a sample --> scan each line
    while (std::getline(inp, buff1)) {
        //line structure is "pattern inp1 ... inpN out1 ... outM" 
        std::stringstream T(buff1);
        //used to ignore the "pattern" word
        first = true;
        //used to count to N --> determine when the inputs are finished and the outputs start
        counter = 0;
        //dummy instance to be inserted in the example set
        Example ex;

        //scan each word of the line to extract the input and output values
        while (std::getline(T, buff2, ' ')) {
            //ignore "pattern"
            if (!first) {
                //determine based on the counter if all the inputs have been read
                if (counter < n_inp) {
                    //insert an input in the example and advance the counter
                    ex.input.push_back(stof(buff2));
                    counter++;
                }
                else {
                    //insert an output in the example
                    ex.label.push_back(stof(buff2));
                }
            }
            else {
                first = false;
            }
        } //TODO handle tr_dens
        //insert the dummy example into the example set
        this->examples.push_back(ex);
    }
    inp.close();
}
//function that converts to a vector of Output used in the initialization of the network to insert the samples in the first neuron layer
std::vector<Output> ExampleSet::toOutputs() {
    std::vector<Output> res;

    //for each input feature a node in the network is needed --> need of an Output for each feature --> scan on each feature
    for (unsigned feature = 0; feature < this->examples[0].input.size(); feature++) {
        Output o;
        //the transfer function on the input layer needs to be "same" as the input don't get touched before the first hidden layer
        o.transfer = &same;
        //collect the value of the current feature for each sample in the set
        for (auto example = this->examples.begin(); example != this->examples.end(); example++) {
            //set the same value in the tmp and best partials as none of them will change
            o.best_partials.push_back(example->input[feature]);
            o.tmp_partials.push_back(example->input[feature]);
        }
        res.push_back(o);
    }
    return res;
}

//NEURON CLASS IMPLEMENTATION
//function to compute the partial output of the neuron 
//if the network has been already initialized only the temporary partial values will be computed
//otherwise the values will be copied in the best temporary as well
void Neuron::calculate() {
    //first layer == in outputs are the examples of the system --> no need to compute anything
    if (weights_and_inputs.empty()) { return; }

    //clear the temporary partial values that may have been left behind in previous calculations
    this->outputs.tmp_partials.clear();
    //resize float vector to match the number of examples (previous layer number of outputs). Assuming previous layer already calculated
    //this should also set each tmp_partial to 0 at the start of computation
    this->outputs.tmp_partials.resize(std::get<1>(*(weights_and_inputs.begin()))->outputs.best_partials.size(), 0.0);

    //compute the dot product between weights and previous neuron's otuputs
    //to do so, for each previous neuron, add to the partial value the multiplication between the weight and the output
    for (auto w_i = weights_and_inputs.begin(); w_i != weights_and_inputs.end(); w_i++) {
        //extract from tuple the single elements, weight and previous neuron pointer
        Weight w = std::get<0>(*w_i);
        Neuron* previous = std::get<1>(*w_i);

        if (previous == nullptr) {
            //if previous equals nullptr the weight is the parameter --> just need to add it to the outputs
            //need to add the parameter to each possible output (i.e. one for each sample)
            for (auto single_o = this->outputs.tmp_partials.begin(); single_o != this->outputs.tmp_partials.end(); single_o++) {
                *single_o += w.value();
            }
        }
        else {
            //else the values of the previous layer's temporary outputs must be computed and multiplyed with the weight
            //the temporary are used instead of the best because in the training phase it will come in handy in the search for the improving move
            std::vector<float> inputs = previous->outputs.values(false);
            //need to multiply each input with the current weight
            for (auto single_i = inputs.begin(), single_o = this->outputs.tmp_partials.begin(); single_i != inputs.end(); single_i++, single_o++) {
                *single_o += (*single_i) * w.value();
            }
        }
    }

    //if it's the initialization phase and the best values haven't been computed the partial values are simply copied
    if (this->outputs.best_partials.empty()) {
        this->outputs.best_partials = this->outputs.tmp_partials;
    }
}

//NETWORK CLASS IMPLEMENTATION
//initialization function of the network
void Network::init(Init* in, ExampleSet* ex_set) {
    //need to initialize network == neuron structure + general parameters + input data  

    //initialize global random number generator with given seed 
    /* NEW RNG SYSTEM */
    rng.seed(in->r_seed);
    /* OLD RNG SYSTEM */
    std::srand(in->r_seed);

    //general parameters initialization
    //the weight range is simply present in the init instance
    this->w_range = in->w_range;
    //for each layer, the limits and initial bits for discretization must be set
    for (auto bits = in->d_bits.begin(); bits != in->d_bits.end(); bits++) {
        //set the initial bits as the bit limit for the layer or to the given value (if in telescopic mode and the value is specified)
        int ini_bits = (!in->telescopic || in->ini_bits == -1) ? (*bits) : in->ini_bits;
        //insert the bits parameters in the vectors
        this->current_bits.push_back(ini_bits);
        this->bit_limits.push_back(*bits);
    }

    //neuron structure
    //input layer creation
    //extract a suitable output vector from the example set in order to insert it in the input layer
    std::vector<Output> features = ex_set->toOutputs();
    std::vector<Neuron> input_l;
    //for each feature of the inputs one neuron must be added to the input layer
    for (auto feature_values = features.begin(); feature_values != features.end(); feature_values++) {
        //neuron initialization with the feature values
        Neuron n;
        n.outputs = *feature_values;
        input_l.push_back(n);
    }
    //insert the created input layer to the neuron structure
    this->neurons.push_back(input_l);

    //hidden layers creation
    //create one layer for each required hidden one (specified in the n_h_l parameter)
    for (unsigned layer = 0; layer < in->n_h_l; layer++) {
        std::vector<Neuron> hidden_l;
        //extract the transfer function enumerate for the current layer
        Transfer t = in->atransf[layer];
        //need to create and insert the required number of neurons for the current layer
        for (unsigned neuron = 0; neuron < in->n_neur[layer]; neuron++) {
            Neuron n;
            //need to specify the actual transfer function from the enumerate value
            switch (t) {
            case Transfer::Logistic:
                n.outputs.transfer = &logistic;
                break;
            case Transfer::HyperbolicTangent:
                n.outputs.transfer = &hyperTan;
                break;
            case Transfer::Arctangent:
                n.outputs.transfer = &arctangent;
                break;
            default:
                //as default the same function is used
                n.outputs.transfer = &same;
                break;
            }
            //initialize parameter value first (even if it will we put last) in order to comply with the original implementation
            Weight par;
            par.init(in, layer);
            //insert the connection to each previous layer's neuron and generate the associated weight 
            for (auto previous_n = this->neurons[layer].begin(); previous_n != this->neurons[layer].end(); previous_n++) {
                Weight w;
                w.init(in, layer);
                n.weights_and_inputs.push_back({ w, &(*previous_n) });
            }
            //insert the weight not associated with any neuron == parameter
            n.weights_and_inputs.push_back({ par, nullptr });
            hidden_l.push_back(n);
        }
        this->neurons.push_back(hidden_l);
    }

    //output layer creation works exactly as hidden ones
    int n_outputs = ex_set->examples[0].label.size();
    std::vector<Neuron> output_l;
    Transfer t = in->atransf[(in->n_h_l)];
    for (unsigned neuron = 0; neuron < n_outputs; neuron++) {
        Neuron n;
        switch (t) {
        case Transfer::Logistic:
            n.outputs.transfer = &logistic;
            break;
        case Transfer::HyperbolicTangent:
            n.outputs.transfer = &hyperTan;
            break;
        case Transfer::Arctangent:
            n.outputs.transfer = &arctangent;
            break;
        default:
            n.outputs.transfer = &same;
        }
        Weight par;
        par.init(in, in->n_h_l);
        for (auto previous_n = this->neurons[(in->n_h_l)].begin(); previous_n != this->neurons[(in->n_h_l)].end(); previous_n++) {
            Weight w;
            w.init(in, (in->n_h_l));
            n.weights_and_inputs.push_back({ w, &(*previous_n) });
        }
        //insert the weight not associated with any neuron == parameter
        n.weights_and_inputs.push_back({ par, nullptr });
        output_l.push_back(n);
    }
    this->neurons.push_back(output_l);

    /* TEST INITIALIZATION */
    std::ofstream t_out("../TestResults/W_init.txt");
    if(!t_out) {return;}
    t_out << "Initial discrete weights" << std::endl;
    for (int l = 1; l < this->neurons.size(); l++) {
        for (int n = 0; n < this->neurons[l].size(); n++) {
            t_out << std::get<0>(this->neurons[l][n].weights_and_inputs[this->neurons[l][n].weights_and_inputs.size() - 1]).discrete << std::endl;
            for (int w = 0; w < this->neurons[l][n].weights_and_inputs.size() - 1; w++) {
                t_out << std::get<0>(this->neurons[l][n].weights_and_inputs[w]).discrete << std::endl;
            }
        }
    }
    t_out.close();
}
//compute the result of the current configuration of the network for each input sample
//it is possible to insert a strating position (usefull to reduce number of computation in training phase when a weight is changed)
//if nullptr is passed as position the entirety of the computation will be made
//change in weight is the difference in value of the new weight (it's used in the partial update mode)
std::vector<std::vector<float>> Network::eval(int* start_position, float change_in_weight) {
    //position is formatted as: [layer, neuron, weight], or empty
    //The function should: verify if a starting position exists (otherwise compute all); compute result; return the output layer's Output vector
    if (start_position == nullptr) {
        //compute all neurons' outputs in order from first hidden layer to output (using the apposite function of the neuron class)
        for (auto layer = this->neurons.begin(); layer != this->neurons.end(); layer++) {
            for (auto neur = (*layer).begin(); neur != (*layer).end(); neur++) {
                (*neur).calculate();
            }
        }
    }
    else {
        //compute starting from position and using partial updates

        //update the output of the neuron with a changed weight
        int start_l = start_position[0], start_n = start_position[1], start_w = start_position[2];
        //verify that the changed weight isn't the parameter
        Neuron* input_n = std::get<1>(this->neurons[start_l][start_n].weights_and_inputs[start_w]);
        if (input_n == nullptr) {
            //difference in parameter must be add to each output
            for (int i = 0; i < this->neurons[start_l][start_n].outputs.tmp_partials.size(); i++) {
                this->neurons[start_l][start_n].outputs.tmp_partials[i] = this->neurons[start_l][start_n].outputs.best_partials[i] + (change_in_weight);
            }
        }
        else {
            //extract the inputs of the neuron in the starting position associated with the changed weight
            //those are the best outputs (ignore the partials in order to avoid residual computations from other attempts)
            std::vector<float> inputs = input_n->outputs.values(true);
            //compute the tmp_partial outputs of the starting neuron starting from the best ones and adding the difference in weight multiplied by the inputs obtained before
            //intuitively only the change in weight is different from the best partials (which were calculated before), so using the inputs the change in the outputs is easily calculated
            for (int i = 0; i < this->neurons[start_l][start_n].outputs.tmp_partials.size(); i++) {
                this->neurons[start_l][start_n].outputs.tmp_partials[i] = this->neurons[start_l][start_n].outputs.best_partials[i] + (change_in_weight * inputs[i]);
            }
        }

        //update the layer successive to the one with the changed weight 
        //each neuron of the successive layer has only one input changed 
        //need to add the difference (multiplied by the unchanged weights) as before for each neuron in the layer
        //obviously this is needed only if there is a successive layer
        if (start_l != this->neurons.size() - 1) {
            //extract the original values (saved as best) and the new ones (saved as partial) to compute the difference later 
            std::vector<float> previous_outputs = this->neurons[start_l][start_n].outputs.values(true);
            std::vector<float> new_outputs = this->neurons[start_l][start_n].outputs.values(false);
            //keep track of the layer reached
            int successive_l = start_l + 1;
            //for each neuron in the successive layer compute the chagne in the outputs
            for (auto neur = this->neurons[successive_l].begin(); neur != this->neurons[successive_l].end(); neur++) {
                //extract the correct weight value == the one corresponding to the starting neuron
                float w = std::get<0>((*neur).weights_and_inputs[start_n]).value();
                //compute the tmp_partial outputs with the same logic as the starting neuron's
                for (int i = 0; i < new_outputs.size(); i++) {
                    (*neur).outputs.tmp_partials[i] = (*neur).outputs.best_partials[i] + (w * (new_outputs[i] - previous_outputs[i]));
                }
            }

            //update all the other following layers (if present)
            //they have all the inputs changed
            //using the previous method would be inefficient --> recompute form scratch the output of each neuron
            while (++successive_l < this->neurons.size()) {
                for (auto neur = this->neurons[successive_l].begin(); neur != neurons[successive_l].end(); neur++) {
                    (*neur).calculate();
                }
            }
        }
        else {
            //if a weight in the last layer has been modified I need to make sure that the correct outputs are extracted in the next layer
            for (int out_n = 0; out_n < this->neurons.back().size(); out_n++) {
                if (out_n == start_n) { continue; }
                this->neurons.back()[out_n].outputs.tmp_partials = this->neurons.back()[out_n].outputs.best_partials;
            }
        }
    }

    //extract outputs of the model and return them
    //save the last layer of neurons == the output layer
    std::vector<Neuron> out_l = this->neurons.back();
    std::vector<std::vector<float>> res;
    //for each output neuron calculate the complete output values and insert them in the returning vector
    for (auto out_n = out_l.begin(); out_n != out_l.end(); out_n++) {
        res.push_back((*out_n).outputs.values(false));
    }
    //res is organized as a vector where each entry represents one dimension of the output
    //each entry is a vector containing the value for each sample
    return res;
}
//function that changes the weight at a given position using the bitflip function of the weight class
//output is the value of the weight change
float Network::change(int* position, unsigned bit) {
    //position is formatted as: [layer, neuron, weight]
    //save the unchanged value of the required weight
    Weight previous = std::get<0>(this->neurons[position[0]][position[1]].weights_and_inputs[position[2]]);
    //change the required weight
    std::get<0>(this->neurons[position[0]][position[1]].weights_and_inputs[position[2]]).bitFlip(bit, this->bit_limits[position[0] - 1]);
    //extract the changed value of the weight
    Weight current = std::get<0>(this->neurons[position[0]][position[1]].weights_and_inputs[position[2]]);
    //return the floating difference in between the original value and the modified one
    return (current.value() - previous.value());
}
//function that updates the best partial outputs of each neuron starting from a given position
void Network::updateOutput(int* position) {
    //position just needs [layer,neuron]
    //set the new output as permanent for the modified neruon 
    this->neurons[position[0]][position[1]].outputs.best_partials = this->neurons[position[0]][position[1]].outputs.tmp_partials;

    int layer = position[0];
    //set the new outputs as permanent for each neuron in each successive layer
    while (++layer < this->neurons.size()) {
        for (auto n = this->neurons[layer].begin(); n != this->neurons[layer].end(); n++) {
            (*n).outputs.best_partials = (*n).outputs.tmp_partials;
        }
    }
}
//function that switches the current inputs with the one given as parameter
void Network::switchInputs(ExampleSet* ex_set) {
    std::vector<Output> new_inputs = ex_set->toOutputs();
    for (unsigned n = 0; n < this->neurons[0].size(); n++) {
        this->neurons[0][n].outputs = new_inputs[n];
    }
}
//function that saves the fundamental components of the network structure to the specified json file and the outputs of the network to the specified .exa file
//layout of saved data is specified in the documentation
void Network::saveToFile(Init* in, ExampleSet* ex_set) {
    int curr_max = *std::max_element(this->current_bits.begin(), this->current_bits.end());
    //we are saving partials results and structure if and only if the max(current bit) != max(bit limit)
    //need to chose were to save
    std::string s_path;
    std::string r_path;
    if (*std::max_element(this->bit_limits.begin(), this->bit_limits.end()) != curr_max) {
        //find the folder where the struct file should be saved and create the partial path depending on the current bit
        std::stringstream s_folder(in->out_struct_f);
        //control is used to ignore the last component of the path (a.k.a the file name)
        //therefore it is kept one component ahead
        std::stringstream s_control(in->out_struct_f);
        std::string s_buff;
        std::getline(s_control, s_buff, '/');
        while (std::getline(s_control, s_buff, '/')) {
            std::getline(s_folder, s_buff, '/');
            //insert the component extracted from the original path in the desired one
            s_path += s_buff;
            s_path += "/";
        }
        //insert the partial output filename
        s_path += "partial_structure_";
        s_path += std::to_string(curr_max);
        s_path += ".json";

        //find the folder where the results file should be saved and create the partial path depending on the current bit
        std::stringstream r_folder(in->out_results_f);
        //control is used to ignore the last component of the path (a.k.a the file name)
        //therefore it is kept one component ahead
        std::stringstream r_control(in->out_results_f);
        std::string r_buff;
        std::getline(r_control, r_buff, '/');
        while (std::getline(r_control, r_buff, '/')) {
            std::getline(r_folder, r_buff, '/');
            //insert the component extracted from the original path in the desired one
            r_path += r_buff;
            r_path += "/";
        }
        //insert the partial output filename
        r_path += "partial_outputs_";
        r_path += std::to_string(curr_max);
        r_path += ".exa";
    }
    else {
        s_path = in->out_struct_f;
        r_path = in->out_results_f;
    }

    if (in->save_structure) {
        //save json structure of the network both final and partial
        std::ofstream out(s_path);
        out << "{" << std::endl;
        out << "\t" << "\"neurons\": [" << std::endl;
        for (int l = 1; l < this->neurons.size(); l++) {
            for (int n = 0; n < this->neurons[l].size(); n++) {
                out << "\t\t" << "{" << std::endl;
                out << "\t\t\t" << "\"pos\":[" << l << "," << n << "]," << std::endl;
                out << "\t\t\t" << "\"weights\": [" << std::endl;
                for (int w = 0; w < this->neurons[l][n].weights_and_inputs.size(); w++) {
                    Weight weight = std::get<0>(this->neurons[l][n].weights_and_inputs[w]);
                    out << "\t\t\t\t" << "{" << std::endl;
                    out << "\t\t\t\t\t" << "\"index\":" << w << "," << std::endl;
                    out << "\t\t\t\t\t" << "\"discrete\":" << weight.discrete << "," << std::endl;
                    out << "\t\t\t\t\t" << "\"constant\":" << weight.constant << std::endl;
                    if (w + 1 < this->neurons[l][n].weights_and_inputs.size()) { out << "\t\t\t\t" << "}," << std::endl; }
                    else { out << "\t\t\t\t" << "}" << std::endl; }
                }
                out << "\t\t\t" << "]" << std::endl;
                if (l + 1 == this->neurons.size() && n + 1 == this->neurons[l].size()) { out << "\t\t" << "}" << std::endl; }
                else { out << "\t\t" << "}," << std::endl; }
            }
        }
        out << "\t" << "]," << std::endl;
        out << "\t" << "\"bit_limits\": [";
        for (int i = 0; i < this->bit_limits.size(); i++) {
            if (i != 0) { out << ", "; }
            out << this->bit_limits[i];
        }
        out << "]," << std::endl;
        out << "\t" << "\"current_bits\": [";
        for (int i = 0; i < this->current_bits.size(); i++) {
            if (i != 0) { out << ", "; }
            out << this->current_bits[i];
        }
        out << "]," << std::endl;
        out << "\t" << "\"w_range\": " << this->w_range << "," << std::endl;
        out << "\t" << "\"atransf\": [";
        for (int i = 0; i < in->atransf.size(); i++) {
            if (i != 0) { out << ", "; }
            out << in->atransf[i];
        }
        out << "]" << std::endl;
        out << "}" << std::endl;
        out.close();
    }

    //save results of the network, both partial and complete
    std::ofstream r_out(r_path);
    r_out << "n_inp " << ex_set->examples[0].input.size() << std::endl;
    r_out << "n_out " << ex_set->examples[0].label.size() << std::endl;
    r_out << "n_patt " << ex_set->examples.size() << std::endl;
    r_out.precision(5);
    for (int ex = 0; ex < ex_set->examples.size(); ex++) {
        r_out << "pattern ";
        for (auto inp = ex_set->examples[ex].input.begin(); inp != ex_set->examples[ex].input.end(); inp++) {
            r_out << std::fixed << (*inp) << " ";
        }


        for (auto o_channel = this->neurons.back().begin(); o_channel != this->neurons.back().end(); o_channel++) {
            r_out << std::fixed << (*o_channel).outputs.values(true)[ex] << " ";
        }
        r_out << std::endl;
    }
    r_out.close();
}
//function that extracts info from a network structure json file and initializes network to match the requirements
void Network::loadFromFile(Init* in, ExampleSet* ex_set) {
    //initialize global random number generator with given seed 
    rng.seed(in->r_seed);

    using json = nlohmann::json;

    //need to load json file structure and extract data
    std::ifstream f(in->in_struct_f);
    json data = json::parse(f);

    //general parameters extraction and setting

    //if the init structure already contains specified values for the parameter those have the priority
    //otherwise use the one specified in the json file
    this->w_range = (in->w_range == 0) ? (float)data["w_range"] : in->w_range;
    if (!in->d_bits.empty()) {
        for (auto bit = in->d_bits.begin(); bit != in->d_bits.end(); bit++) {
            this->bit_limits.push_back(*bit);
        }
    }
    else {
        for (auto bit = data["bit_limits"].begin(); bit != data["bit_limits"].end(); bit++) {
            this->bit_limits.push_back(*bit);
        }
    }
    if (in->ini_bits != -1) {
        for (int i = 0; i < in->d_bits.size(); i++) {
            this->current_bits.push_back(in->ini_bits);
        }
    }
    else {
        for (auto bit = data["current_bits"].begin(); bit != data["current_bits"].end(); bit++) {
            this->current_bits.push_back(*bit);
        }
    }

    //neuron structure initialization

    //input layer creation with the same techinque as in the init function
    std::vector<Output> channel_in = ex_set->toOutputs();
    std::vector<Neuron> input_l;
    for (auto channel = channel_in.begin(); channel != channel_in.end(); channel++) {
        Neuron n;
        n.outputs = *channel;
        input_l.push_back(n);
    }
    this->neurons.push_back(input_l);

    //extracting info from the the json structure for the hidden layers
    //variable to keep track of the current layer while scrolling through every neuron position
    int l = 1;
    //data["neurons"] is a vector containing every neuron's info need to scan all of it
    //only the neurons from the first hidden layer onwards are saved
    std::vector<Neuron> layer;
    for (auto neuron = data["neurons"].begin(); neuron != data["neurons"].end(); neuron++) {
        //the neuron's position is saved as [layer, neuron] --> layer starts from 1 as the input layer is not saved
        //need to check if the layer has changed 
        if (l != (*neuron)["pos"][0]) {
            //if the layer has changed insert the layer constructed so far in the neuron structure
            this->neurons.push_back(layer);
            //clear the temporary layer to restart for the next layer
            layer.clear();
            //set the new layer number in l
            l = (*neuron)["pos"][0];
        }

        //create a neuron and insert it in the temporary layer

        //extract the transfer function numerical code for the current layer
        int t = data["atransf"][l - 1];
        Neuron n;
        //set the correct transfer function of the neuron baased on the numerical code
        switch (t) {
        case 0:
            n.outputs.transfer = &logistic;
            break;
        case 1:
            n.outputs.transfer = &hyperTan;
            break;
        case 2:
            n.outputs.transfer = &arctangent;
            break;
        default:
            n.outputs.transfer = &same;
            break;
        }

        //insert the previous neurons connections and extract the weight values from the json structure
        int i = 0;
        for (auto previous_n = this->neurons[l - 1].begin(); previous_n != this->neurons[l - 1].end(); previous_n++) {
            Weight w;
            w.constant = (*neuron)["weights"][i]["constant"];
            w.discrete = (*neuron)["weights"][i]["discrete"];
            n.weights_and_inputs.push_back({ w, &(*previous_n) });
            i++;
        }
        //insert the last weight == the parameter == unrelated to any previous neuron
        Weight par;
        par.constant = (*neuron)["weights"][i]["constant"];
        par.discrete = (*neuron)["weights"][i]["discrete"];
        n.weights_and_inputs.push_back({ par, nullptr });
        layer.push_back(n);
    }
    this->neurons.push_back(layer);
    f.close();
}


