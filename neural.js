//
// A simple feed forward XOR neural network in Javascript
//
//      Author: Gustavo Selbach Teixeira
//


// The Layer  objects, holds the neurons and their data 
class Layer {
    values;
    bias;
    deltas;
    weights;
    n_nodes;
    n_synapses;

    constructor(n_nodes, n_connections) {
        this.n_nodes = n_nodes;
        this.n_synapses = n_connections;
        this.values = Array();
        this.bias = Array();
        this.deltas = Array();
        this.weights = Array();
        for (let i=0; i<n_nodes; i++) {
            this.values.push(Math.random());
            this.bias.push(Math.random());
            this.deltas.push(Math.random());
        }
        
        for (let i=0; i<n_connections; i++) {
            this.weights.push(Array());
            for (let k=0; k<n_nodes; k++) {
                this.weights[i].push(Math.random());
            }
        }
    }
}

// The NeuralNetwork object definition
class NeuralNetwork {
    input_layer;
    hidden_layer;
    output_layer;
    
    learning_rate;
    input_size;
    hidden_size;
    output_size;
    
    constructor(input_size, output_size, hidden_size) {
        this.input_layer = new Layer(input_size, 0);
        this.hidden_layer = new Layer(hidden_size, input_size);
        this.output_layer = new Layer(output_size, hidden_size);
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.output_size = output_size;
        this.learning_rate = 0.1
    }
    
    // Feed data to the network
    set_inputs(inputs) {
        for (var i=0; i<this.input_size; i++) {
            this.input_layer.values[i] = inputs[i];
        }
    }

    // The activation function
    activation_function (source, target) {
        var activation;
        for (let j=0; j<target.n_nodes; j++) {
            activation = target.bias[j];
            for (let k=0; k<source.n_nodes; k++) {
                activation += (source.values[k] * target.weights[k][j]);
            }
            target.values[j] = sigmoid(activation);
        }
    }
    
    // compute deltas for the output layer
    calc_delta_output(expected) {
        var errors;
        for (let i=0; i<this.output_layer.n_nodes; i++) {
            errors = (expected[i] - this.output_layer.values[i]);
            this.output_layer.deltas[i] = (errors * d_sigmoid(this.output_layer.values[i]));
        }
    }
    
    // Compute the deltas between layers
    calc_deltas(source, target) {
        for (let j=0; j<target.n_nodes; j++) {
            let errors = 0.0;
            for (let k=0; k<source.n_nodes; k++) {
                errors += (source.deltas[k] * source.weights[j][k]);
            }
            target.deltas[j] = (errors * d_sigmoid(target.values[j]));
        }
    }
    
    // Update the weights and bias
    update_weights(source, target) {
        for (let j=0; j<source.n_nodes; j++) {
            source.bias[j] += (source.deltas[j] * this.learning_rate);
            for (let k=0; k<target.n_nodes; k++) {
                source.weights[k][j] += (target.values[k] * source.deltas[j] * this.learning_rate);
            }
        }
    }

    // NeuralNetwork Activation step 
    forward_pass() {
        this.activation_function(this.input_layer, this.hidden_layer);
        this.activation_function(this.hidden_layer, this.output_layer);
    }
    
    // The back propagation process
    back_propagation(outputs) {
        this.calc_delta_output(outputs);
        this.calc_deltas(this.output_layer, this.hidden_layer);
        // update weights and bias
        this.update_weights(this.output_layer, this.hidden_layer);
        this.update_weights(this.hidden_layer, this.input_layer);
    }
    
    // Training main loop
    train (inputs, outputs, n_epochs) {
        var num_training_sets = inputs.length;
        
        for (let e=0; e<n_epochs; e++) {
            for (let i=0; i<num_training_sets; i++) {
                this.set_inputs(inputs[i]);
                this.forward_pass();
                // Show results
                console.log("Input: ", inputs[i],
                            "Expected: ", outputs[i],
                            "Output: ", this.output_layer.values);
                // Learning
                this.back_propagation(outputs[i]);
            }
        }
    }
}

// The logistical sigmoid function
function sigmoid(x) {
    return 1 / (1 + (Math.exp(-x)))
}

// The derivative of sigmoid function
function d_sigmoid(x) {
    return x * (1 - x)
}

// main function
function main() {
    var  inputs = [[0.0, 0.0],
                   [1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]]
    var outputs = [[0.0], [1.0], [1.0], [0.0]]

    nn = new NeuralNetwork(2, 1, 4);
    nn.train(inputs, outputs, 10000);
}

main()
