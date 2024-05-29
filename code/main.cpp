#include "network.h"
#include <iomanip>

//constant used to decide how often the reports of the training are printed == it's a percentage
#define REPORT_EACH_UPDATE_PERCENTAGE 0.1

//constants used in the telescopic step calculation
//decay factor in the average update
#define DECAY_FACTOR 0.3
//fraction of possible moves that conceptually are considered containing an improving move
#define MOVES_FRACTION 1
//number of improvements before evaluation in training
#define EVAL_AFTER_STEPS 1000

using namespace std;
using namespace net;

//TODO fix stepper
//class used to calculate when the bit limit must be updated (for the first improving telescopic mode)
class TelescopicStep {
    //iterations before the first improving move
    int iteration;
    //number of possible moves with current bit number
    int moves;
    //average of previous step
    float avg;
    //decay factor for the average calculation
    float decay_fact;
    //threshold for the bit augmentation calculated in the apposite function
    float threshold;
    //fraction of possible moves that conceptually are considered containing an improving move
    float moves_fraction;

    //function that computes the threshold for the iterations analyzed
    //it is based on the possible moves and the apposite parameters
    void computeThreshold() {
        float k = floor(this->moves * this->moves_fraction);
        //TESTING FACTOR
        // this->threshold=0;
        // float a=0;
        // for(int N=k+1; N<this->moves; N++) {
        //     a = ((float)(N-k)/N) * (k/(N-1) + a);
        //     this->threshold = a + ((float)(N-k)/N)*this->threshold;
        // }
        this->threshold = k;
    }

public:
    //constructor for the class
    //needs the parameters for the stepper to work
    TelescopicStep(float decay_fact, float moves_fraction, Network* net) {
        this->decay_fact = decay_fact;
        this->moves_fraction = moves_fraction;
        this->reset(net);
    }
    //reset function to prepare the stepper for a new bit limit
    //used after the bits limit has been increased
    void reset(Network* net) {
        this->iteration = 0;
        this->avg = 0;
        if (net != NULL) {
            int moves = 0;
            for (int l = 1; l < net->neurons.size(); l++) {
                moves += net->neurons[l].size() * net->neurons[l][0].weights_and_inputs.size() * net->current_bits[l - 1];
            }
            this->moves = moves;
            this->computeThreshold();
        }
    }
    //main function of the stepper
    //tells if the network needs to update the bit limit
    //the update in bits is needed if too many moves have been tried before finding an improving one
    //the number of moves tested is authomatically saved in the stepper --> increased every time this function is called
    bool needBitIncrease() {
        this->iteration++;
        // TESTING FACTOR
        // this->avg = (this->avg)*(this->decay_fact) + (1-this->decay_fact)*(this->iteration);
        this->avg = this->iteration;
        return (this->avg >= this->threshold);
    }
};

/* OLD RNG SYSTEM */
int zero_positions = 0;
//function to reset a suitable positions vector for training
vector<vector<int>> trainingPositions(Network* net, Init* in) {
    /* NEW RNG SYSTEM */
    // vector<vector<int>> positions;
    // vector<vector<int>> zeros;
    // //an available position is a possible bitflip over a weight's discrete value in a neuron
    // //number of possible positions == SUM over each layer (number of neuorns in the layer * number of weights per neuron in the layer * number of usable bits in the layer)
    // for(int l=0; l<net->neurons.size()-1; l++) {
    //     for(int n=0; n<net->neurons[l+1].size(); n++) {
    //         for(int w=0; w<net->neurons[l+1][n].weights_and_inputs.size(); w++) {
    //             for(int b=(net->bit_limits[l] - net->current_bits[l]); b<net->bit_limits[l]; b++) {
    //                 if(in->keep_dens && get<0>(net->neurons[l+1][n].weights_and_inputs[w]).discrete == 0) {
    //                     zeros.push_back({l+1,n,w,b});
    //                 } else {
    //                     positions.push_back({l+1,n,w,b});
    //                 }
    //             }
    //         }
    //     }
    // }
    // shuffle(positions.begin(), positions.end(), rng);
    // shuffle(zeros.begin(), zeros.end(), rng);
    // if(!zeros.empty()) {
    //     positions.insert(positions.begin(), zeros.begin(), zeros.end());
    // }
    // return positions;

    /* OLD RNG SYSTEM */
    vector<vector<int>> positions;
    for (int l = 0; l < net->neurons.size() - 1; l++) {
        for (int n = 0; n < net->neurons[l + 1].size(); n++) {
            for (int w = 0; w < net->neurons[l + 1][n].weights_and_inputs.size(); w++) {
                for (int b = (net->bit_limits[l] - net->current_bits[l]); b < net->bit_limits[l]; b++) {
                    positions.push_back({ l + 1,n,w,b });
                }
            }
        }
    }

    if (in->keep_dens) {
        int i = 0, j = positions.size() - 1;
        while (i < j) {
            if (!get<0>(net->neurons[positions[i][0]][positions[i][1]].weights_and_inputs[positions[i][2]]).discrete)
                i++;
            else if (get<0>(net->neurons[positions[j][0]][positions[j][1]].weights_and_inputs[positions[j][2]]).discrete)
                j--;
            else {
                iter_swap(positions.begin() + i, positions.begin() + j);
                i++;
                j--;
            }
        }
        zero_positions = j + 1;
    }
    return positions;
}
//function to print updates of the training
//two kinds of updates are possible --> decided by the tele_update parameter
//1. if an improving move has been found
//2. if the bit limit has been updated
void printUpdate(Network* net, int iteration, int improving_iteration, int time, float error, bool tele_update) {
    if (tele_update) {
        //inform of the updated bit limit
        cout << endl << "Increased number of bits considered." << endl;
    }
    //inform of the current situation of the training phase
    //iteration reached + time passed + best error reached + bit analyzed at each layer
    cout << "Iterations: " << iteration << "\t(Improving: " << improving_iteration << ")\ttime: " << time / 1000 << "s" << endl;
    cout << "\tError: " << fixed << setprecision(6) << setfill('0') << sqrt((error) / net->neurons[0][0].outputs.best_partials.size()) 
    << "\t\tCurrent bits: ";
    for (auto bit = net->current_bits.begin(); bit != net->current_bits.end(); bit++) {
        cout << (*bit) << "  ";
    }
    cout << endl;
}
//error function for the network's output evaluation
//res is the output of the eval function of a network
float max_error = 1;
float error(vector<vector<float>>* res, ExampleSet* ex_set) {
    /* RMS AS DEFINED */
    // //RMS error 
    // float err=0;
    // for(int j=0; j<ex_set->examples.size(); j++) { //example j
    //     float distance=0;
    //     for(int i=0; i<res->size(); i++) { //channel i
    //         distance+=pow((ex_set->examples[j].label[i] - (*res)[i][j]),2);
    //     }
    //     err+=distance;
    // }
    // err/=ex_set->examples.size();
    // // err/=max_error;
    // // err=sqrt(err);
    // return err;

    /* RMS AS IMPLEMENTED */
    float err = 0;
    unsigned long long nt = 0;
    float errs[32];
    for (int j = 0; j < ex_set->examples.size(); j++) { //example j
        float distance = 0;
        for (int i = 0; i < res->size(); i++) { //channel i
            distance += pow((ex_set->examples[j].label[i] - (*res)[i][j]), 2);
        }
        int pi = 0;
        for (unsigned long long pp = 1; nt & pp; pp <<= 1, pi++)
            distance += errs[pi];
        errs[pi] = distance;
        nt++;
    }
    for (unsigned long long nnt = nt, pi = 0; nnt; nnt >>= 1, pi++) {
        if (nnt & 1) {
            err += errs[pi];
        }
    }
    return err;
}
//training funciton of the model
//it is based on the network structure in the apposite library
float train(Network* net, Init* in, ExampleSet* ex_set, ExampleSet* test_set) {
    //need a vector of not-used positions in order to avoid repeating the same study multiple times
    //it will be reset each time an improving position is found
    vector<vector<int>> positions = trainingPositions(net, in);
    //position of the best weight change formatted as [layer,neuron,weight,bit]
    //used only in the d_all mode
    vector<int> best_position;
    //USed to keep trac of the last position checked in the d_all mode so as to follow a coherent order in the testing of positions
    vector<int> current_position;
    //iteration counter
    int iterations = 0;
    //counter for iterations which provide an improvement
    int improving_iterations = 0;
    //initial best error found evaluating the network after an initialization
    vector<vector<float>> out = net->eval(NULL, 0);
    // max_error = error(&(out), ex_set);
    float best_err = error(&(out), ex_set);
    //initiate the stepper using the constants declared
    //used only in the not-d_all mode
    TelescopicStep stepper = TelescopicStep(DECAY_FACTOR, MOVES_FRACTION, net);
    //counter used to determine how often to print updates
    int update_counter = 0;
    //counter used to determine how often to evaluate
    int eval_counter = 0;
    //log file to save each error evolution
    ofstream log;
    //log file to save training evaluation results
    ofstream test_log;
    //log file to save training positions
    ofstream moves_log;

    //prepare log files to put each iteration and testing result 
    string log_f;
    string test_log_f;
    //prepare log file to put every chosen move
    string moves_log_f;
    //find the folder where the results file should be saved and create the partial path depending on the current bit
    stringstream log_folder(in->out_results_f);
    //control is used to ignore the last component of the path (a.k.a the file name)
    //therefore it is kept one component ahead
    stringstream log_control(in->out_results_f);
    string tmp_buff;
    std::getline(log_control, tmp_buff, '/');
    while (std::getline(log_control, tmp_buff, '/')) {
        std::getline(log_folder, tmp_buff, '/');
        //insert the component extracted from the original path in the desired one
        log_f += tmp_buff;
        test_log_f += tmp_buff;
        moves_log_f += tmp_buff;
        log_f += "/";
        test_log_f += "/";
        moves_log_f += "/";
    }
    //insert the log filename
    log_f += "log.txt";
    test_log_f += "test_log.txt";
    moves_log_f += "moves_log.txt";
    log = ofstream(log_f);
    log << "Training sequence" << endl << "Iteration\tError" << endl;
    moves_log = ofstream(moves_log_f);
    moves_log << "Training sequence" << endl << "Move[l,n,w,b]" << endl;
    if (test_set != nullptr) {
        test_log = ofstream(test_log_f);
        test_log << "Generalization sequence" << endl << "Evaluation completed every: " << EVAL_AFTER_STEPS << " improvements" << endl << "Iteration\tError" << endl;
    }

    printUpdate(net, iterations, improving_iterations, clock(), best_err, false);

    //training phase that has 3 possible limits (the most restrictive is chosen)
    //1. max number of iteration
    //2. max time has passed
    //3. no improving move found with maximum number of bits
    while (((in->max_iter == -1) || (iterations < in->max_iter)) && ((in->time == -1) || (clock() < in->time * 1000))) {
        iterations++;
        
        //two possible modes
        //1. look for the best improving move --> need to check all possible moves and find the best one
        //2. look for the first improving move --> test random the positions untill the stepper tells it's enough or an improving move is found
        //TODO update d_all case
        if (in->d_all) {
            //look for best improving move

            //check last tested position and find the next one to be tested
            if (current_position.empty()) {
                //first time generating a position
                //start from the first acceptable bit of the first weight of the first neuron of the first non-input layer
                //the first acceptable bit depends on the vurrent bit limit --> the limit indicates the first bits (starting from the left) modifiable
                current_position = { 1,0,0,(int)(net->bit_limits[0] - net->current_bits[0]) };
            }
            else {
                //set current position so as to go in order
                //try next bit in the same weight 
                if (current_position[3] < (net->bit_limits[current_position[0] - 1] - 1)) {
                    //there is at least one bit still unchecked 
                    current_position[3]++;
                }
                else {
                    //no other bits to check in this weight
                    current_position[3] = 0;
                    //try the next weight in the same neuron if present
                    if (current_position[2] < (net->neurons[current_position[0]][current_position[1]].weights_and_inputs.size() - 1)) {
                        //there is at least one weight left unchecked in the specified neuron
                        current_position[2]++;
                    }
                    else {
                        //all the weights in the current neuron have been checked
                        //need to go to the next available neuron's first weight
                        current_position[2] = 0;
                        //the next neuron might not be in the same layer
                        if (current_position[1] < (net->neurons[current_position[0]].size() - 1)) {
                            //there is at least one neuron left in the current layer
                            current_position[1]++;
                        }
                        else {
                            //the next neuron is the first of the next layer
                            current_position[1] = 0;
                            //there might not be a next layer
                            if (current_position[0] < (net->neurons.size() - 1)) {
                                //there is a next layer
                                current_position[0]++;
                            }
                            else {
                                //there isn't a next layer --> all possible positions have been checked
                                //must set the best_postion's weight change as permanent
                                //if the best position is still empty there is no improving move
                                if (best_position.empty()) {
                                    //if the bit limit has been reached everywhere the training has been completed, otherwise the limit must be increased
                                    if (*max_element(net->bit_limits.begin(), net->bit_limits.end()) != *max_element(net->current_bits.begin(), net->current_bits.end())) {
                                        //need to increase the bit limit of at least one layer
                                        for (int bit = 0; bit < net->current_bits.size(); bit++) {
                                            //check if limit is already reached otherwise increase it
                                            if (net->current_bits[bit] != net->bit_limits[bit]) {
                                                net->current_bits[bit]++;
                                            }
                                        }
                                    }
                                    else {
                                        //all bit limits reached --> training completed
                                        return best_err;
                                    }
                                }
                                else {
                                    //there is an improving move --> set it as permanent and reset the position for the next move search
                                    int l, n, w, b;
                                    l = best_position[0];
                                    n = best_position[1];
                                    w = best_position[2];
                                    b = best_position[3];
                                    int pos[3] = { l,n,w };
                                    //the evaluation is performed to set the tmp_partials of the best_position
                                    net->eval(pos, net->change(pos, b));
                                    //make partial_tmps permanent
                                    net->updateOutput(pos);
                                    //must reset the current and best position to the first one to restart the d_all process for next improving step
                                    current_position.clear();
                                    best_position.clear();
                                    //print the update info (not allways, only a certain percantage)
                                    if (++update_counter >= (100 / REPORT_EACH_UPDATE_PERCENTAGE)) {
                                        update_counter = 0;
                                        printUpdate(net, iterations, iterations, clock(), best_err, false);
                                    }
                                    //go to next iteration (without counting this one)
                                    iterations--;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            //current_position is now set to a suitable position to be checked
            int l, n, w, bit;
            l = best_position[0];
            n = best_position[1];
            w = best_position[2];
            bit = best_position[3];
            int pos[3] = { l,n,w };
            //change the specified bit and evaluate the error
            out = net->eval(pos, net->change(pos, bit));
            float err = error(&out, ex_set);

            //confront and update best error
            if (err < best_err) {
                //new best move found --> save position and error
                best_err = err;
                best_position = current_position;
            }
            //if the error is worse nothing must be done

            //either way the network must be reset as it was before to allow the evaluation of the next move
            net->change(pos, bit);
        }
        else {
            //look for first improving move

            /* NEW RNG SYSTEM */
            // //generate a random position
            // //the vector of all positions is shuffled in generation == just extract the last position
            // int r_layer = positions.back()[0];
            // int r_neuron = positions.back()[1];
            // int r_weight = positions.back()[2];
            // int r_bit = positions.back()[3];
            // //remove position as it was tested
            // positions.erase(positions.end()-1);

            /* OLD RNG SYSTEM */
            int pos_i = (zero_positions < positions.size()) ? zero_positions + rand() % (positions.size() - zero_positions) : rand() % positions.size();
            int r_layer = positions[pos_i][0];
            int r_neuron = positions[pos_i][1];
            int r_weight = positions[pos_i][2];
            int r_bit = positions[pos_i][3];
            iter_swap(positions.begin() + pos_i, positions.end() - 1);
            positions.erase(positions.end() - 1);

            //change that bit + evaluate and see the new error
            int pos[3] = { r_layer,r_neuron,r_weight };
            out = net->eval(pos, net->change(pos, r_bit));
            float err = error(&out, ex_set);


            //confront & update error
            if ((err) < best_err) { //improving step found
                improving_iterations++;
                //register new finding on logs
                log << improving_iterations << "\t" << err << endl;
                //log the chosen move
                int r_w_number = 0;
                for (int l = 1; l < r_layer; l++) {
                    r_w_number += net->neurons[l].size() * net->neurons[l][0].weights_and_inputs.size();
                }
                r_w_number += r_neuron * net->neurons[r_layer][0].weights_and_inputs.size();
                r_w_number += r_weight;
                moves_log << "[" << r_w_number << "," << r_bit << "]" << endl;
                //update best_error
                best_err = err;
                //recompute every output to erase any possible residues from previous moves
                //net->eval(NULL, 0);
                //make network change permanent
                net->updateOutput(pos);
                //print the update info (not allways, only a certain percantage)
                if (++update_counter >= (100 / REPORT_EACH_UPDATE_PERCENTAGE)) {
                    update_counter = 0;
                    printUpdate(net, iterations, improving_iterations, clock(), best_err, false);
                }
                //reset positions
                positions = trainingPositions(net, in);
                //reset stepper without giving the network == don't need to update the threshold
                stepper.reset(NULL);

                //if needed evaluate the training so far
                if (test_set != nullptr && ++eval_counter >= EVAL_AFTER_STEPS) {
                    eval_counter = 0;
                    //substitute inputs with the testing set
                    net->switchInputs(test_set);
                    //calculate error
                    vector<vector<float>> out = net->eval(nullptr, 0);
                    float test_err = error(&out, test_set);
                    //log error
                    test_log << improving_iterations << "\t" << test_err << endl;
                    //restore inputs to the originals
                    net->switchInputs(ex_set);
                    net->eval(nullptr, 0);
                }
            }
            else { //modification was a failure
                //reset change in weight
                net->change(pos, r_bit);
                //make a step and check result
                if (positions.empty()/*stepper.needBitIncrease()*/) { //telescopic threshold reached
                    //save results obtained so far
                    cout << "Saving partial results..." << endl;
                    net->saveToFile(in, ex_set);
                    cout << "Save completed" << endl << endl;

                    //increase bits or end training
                    if (*max_element(net->bit_limits.begin(), net->bit_limits.end()) == *max_element(net->current_bits.begin(), net->current_bits.end())) { //bit limit already reached everywhere
                        return best_err;
                    }
                    else { //increase current_bits
                        for (int bit = 0; bit < net->current_bits.size(); bit++) {
                            //check if limit is already reached
                            if (net->current_bits[bit] != net->bit_limits[bit]) {
                                net->current_bits[bit]++;
                            }
                        }
                    }
                    //reset positions
                    positions = trainingPositions(net, in);
                    //reset stepper completely == give network
                    stepper.reset(net);
                    //print update
                    printUpdate(net, iterations, improving_iterations, clock(), best_err, true);
                    update_counter = 0;
                }
            }
        }

    }

    return best_err;
}

int main(int argc, char* argv[]) {
    //the program needs a command file path in the argv parameters
    if(argc < 2) {
        cerr << "ERROR" << endl << "Missing command file"<< endl << "See usage in doc folder" << endl;
        exit(1);
    }
    //it is used to extract all the parammeters and functionalities
    //use the command file to initialize the Init instance
    string path = argv[1];
    Init in = Init(path);
    //from the Init instance initialize the ExampleSet instance 
    ExampleSet ex_set = ExampleSet(&in, false);
    Network net;

    //decide if the network structure must be initialized from scratch or loaded from a file
    if (!in.load_structure) {
        net.init(&in, &ex_set);
    }
    else {
        net.loadFromFile(&in, &ex_set);
    }

    float err;
    //decide if the program should train or evaluate the network and perform the action required
    if (in.train) {
        cout << "Starting training..." << endl;
        if (!in.in_train_test_f.empty()) {
            ExampleSet test_set = ExampleSet(&in, true);
            err = train(&net, &in, &ex_set, &test_set);
        }
        else {
            err = train(&net, &in, &ex_set, nullptr);
        }
        cout << "Training completed.";
    }
    else {
        cout << "Starting evaluation..." << endl;
        vector<vector<float>> out = net.eval(NULL, 0);
        err = error(&out, &ex_set);
        cout << "Evaluation completed.";
    }
    cout << "\tAverage error: " << sqrt(err/ex_set.examples.size()) << endl;

    //save results and eventually the structure of the network
    cout << "Saving results in: " << in.out_results_f << endl;
    if (in.save_structure) { cout << "Saving structure in: " << in.out_struct_f << endl; }
    net.saveToFile(&in, &ex_set);
    cout << "Saving completed." << endl << endl;
    cout << "Process completed." << endl;

    return 0;
}


