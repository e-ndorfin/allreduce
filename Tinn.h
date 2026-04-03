/*
 * Tinn — Tiny Neural Network library
 * Original source: https://github.com/glouw/tinn
 * Author: Gustav Louw
 * License: MIT
 *
 * This file was copied from the Tinn library and modified for distributed
 * training. The Tinn struct and original API (xtpredict, xttrain, xtbuild,
 * xtsave, xtload, xtfree, xtprint) are from the original library.
 * The "Distributed training API" section below was added by us.
 */

#pragma once

typedef struct
{
    // All the weights.
    float *w;
    // Hidden to output layer weights.
    float *x;
    // Biases.
    float *b;
    // Hidden layer.
    float *h;
    // Output layer.
    float *o;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    int nb;
    // Number of weights.
    int nw;
    // Number of inputs.
    int nips;
    // Number of hidden neurons.
    int nhid;
    // Number of outputs.
    int nops;
} Tinn;

float *xtpredict(Tinn, const float *in);

float xttrain(Tinn, const float *in, const float *tg, float rate);

Tinn xtbuild(int nips, int nhid, int nops);

void xtsave(Tinn, const char *path);

Tinn xtload(const char *path);

void xtfree(Tinn);

void xtprint(const float *arr, const int size);

// Distributed training API
// grad and buf are flat arrays of (nw + nb) floats: [weights...][biases...]

// Performs forward pass given Tinn and input
void xtforward(Tinn t, const float *in);

// Performs backward pass given Tinn, input, target values, LR, output
// into buffer grad. Returns error.
float xtbackward(Tinn t, const float *in, const float *tg, float rate, float *grad);

// Apply delta changes to gradient vector
void xtapply(Tinn t, const float *grad);

// Turn weights and biases into buffer
void xtgetweights(Tinn t, float *buf);

// Load weights and biases from buffer into Tinn
void xtsetweights(Tinn t, const float *buf);
