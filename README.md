# A Neural Network in Javascript

Simple feed forward neural network in Javascript


## usage:

To run on console (need to have nodejs):

```shell
    make
    # or
    nodejs neural.js
```

You can also import to html, then see results at console.

## Create a neural network:

Create a network telling the size (nodes) of earch layer.
```javascript
    var  inputs = [[0.0, 0.0],
                   [1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]]
    var outputs = [[0.0], [1.0], [1.0], [0.0]]

    nn = new NeuralNetwork(2, 1, 4);
    nn.train(inputs, outputs, 10000);
```

## To be done
    - Allow to use different logistical functions, like ReLU.
    - Make it a reusable javascript tool that can be used by webpages.
