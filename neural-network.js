const getRandomDouble = digits => Number((Math.random()*2-1).toFixed(digits)); // random int between -1.00 and 1.00

// activation functions (https://en.wikipedia.org/wiki/Activation_function) =>

// Sigmoid
// const activation = x => 1 / (1 + Math.exp(-x));
// const derivative = x => x * (1 - x); // derivative of activation function

// ReLu
// const activation = x => Math.max(0, x);
// const derivative = x => x > 0 ? 1 : 0; // derivative of activation function

// Modified sigmoid
const activation = x => 2 / (1 + Math.exp(-x)) - 1;
const derivative = x => 0.5 * (1 + x) * (1 - x); // derivative of activation function

// Hyperbolic tangens
// const activation = x => Math.tanh(x);
// const derivative = x => 1 - Math.tanh(x)**2; // derivative of activation function

class NeuralNetwork {
    // creating NN architecture
    constructor(inputNeurons, hiddenNeurons, outputNeurons) {
        this.layers = [];
        this.lambda = 0.1; // learn rate
        // {neurons[], weights[][], biases[]}

        // number of neurons in each layer
        this.sizes = [inputNeurons, ...hiddenNeurons, outputNeurons]; 

        this.sizes.forEach((currLSize, idx) => {
            if (!this.sizes[idx+1]) { // if output layer
                let neurons = [];
                let biases = [];
                for (let n = 0; n < currLSize; n++) {
                    neurons.push(0);
                    biases.push(getRandomDouble(2));
                }

                this.layers.push({ neurons, weights: null, biases });
            } else {
                let nextLSize = this.sizes[idx+1];

                let neurons = [];
                let biases = [];
                for (let n = 0; n < currLSize; n++) {
                    neurons.push(0);
                    biases.push(getRandomDouble(2));
                }

                let weights = [];
                for (let cni = 0; cni < currLSize; cni++) {
                    let y = [];

                    for (let nni = 0; nni < nextLSize; nni++) {
                        y.push(getRandomDouble(2));
                    }

                    weights.push(y);
                }

                this.layers.push({ neurons, weights, biases });
            }
        });
    }

    // analyze algrorythm
    feedForward = input => { // https://en.wikipedia.org/wiki/Feedforward_neural_network
        if (input.length !== this.sizes[0]) {
            return console.error("wrong input array");
        }

        let { layers } = this;
        layers[0].neurons = input; 

        for (let li = 0; li < layers.length-1; li++) {
            let currLayer = layers[li];
            let nextLayer = layers[li+1];

            for (let nli = 0; nli < nextLayer.neurons.length; nli++) {
                nextLayer.neurons[nli] = 0;

                for (let cli = 0; cli < currLayer.neurons.length; cli++) {
                    let bias = 0;
                    // let bias = nextLayer.biases[nli];
                    let weight = currLayer.weights[cli][nli];
                    let neuron = currLayer.neurons[cli];

                    nextLayer.neurons[nli] += (neuron * weight) + bias;
                }

                nextLayer.neurons[nli] = activation(nextLayer.neurons[nli])
            }
        }

        let output = layers[layers.length - 1].neurons;

        this.layers = layers;

        return output;
    }

    // learn algorythm
    backPropagation = errors => { // https://en.wikipedia.org/wiki/Backpropagation
        let { layers, lambda } = this;

        let gradients = [];

        for (let li = layers.length - 1; li >= 1; li--) {
            let currLayer = layers[li];
            let nextLayer = layers[li-1];

            if (li === layers.length - 1) { // if first iteration
                for (let ni = 0; ni < currLayer.neurons.length; ni++) {
                    gradients[ni] = errors[ni] * derivative(currLayer.neurons[ni]);
                }
            }

            for (let cli = 0; cli < currLayer.neurons.length; cli++) {
                for (let nli = 0; nli < nextLayer.neurons.length; nli++) {
                    let gradient = gradients[cli];
                    let neuron = nextLayer.neurons[nli];
                    
                    // currLayer.biases[cli] += gradients[cli];
                    nextLayer.weights[nli][cli] = nextLayer.weights[nli][cli] - (lambda * gradient * neuron);
                }
            }

            let newGradients = [];
            for (let nli = 0; nli < nextLayer.neurons.length; nli++) {
                newGradients[nli] = 0;

                for (let cli = 0; cli < currLayer.neurons.length; cli++) {
                    let weight = nextLayer.weights[nli][cli];
                    let gradient = gradients[cli];
                    let neuron = nextLayer.neurons[nli];

                    newGradients[nli] += weight * gradient * derivative(neuron);
                }
            }

            gradients = newGradients;
        }
    }
}

module.exports = NeuralNetwork;