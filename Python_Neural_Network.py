import numpy as np

L = 3
n = [2, 3, 3, 1]

W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

def prepare_data():
    X = np.array([[150, 70],[254, 73],[312, 68],[120, 60],[154, 61],[212, 65],
              [216, 67],[145, 67],[184, 64],[130, 69]])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    m = 10
    A0 = X.T
    Y = y.reshape(n[L], m)
    return A0, Y, m

def sigmoid(arr):
  return 1 / (1 + np.exp(-1 * arr))

def cost(Y_hat, Y):
    m = Y.shape[1]
    losses = -((Y*np.log(Y_hat)) + (1-Y)*np.log(1-Y_hat))
    return (1/m) * np.sum(losses)

def feed_forward(A0):
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)

    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    cache = {"A0": A0, "A1": A1, "A2": A2}
    return A3, cache

A0, Y, m = prepare_data()

def back_propagation_layer_3(Y_hat, Y, m, A2, W3):
    A3 = Y_hat
    dC_dZ3 = (1/m)*(A3-Y)
    assert dC_dZ3.shape == (n[3],m)

    dZ3_dW3 = A2
    assert dZ3_dW3.shape == (n[2],m)
    dC_dW3 = dC_dZ3 @ dZ3_dW3.T
    assert dC_dW3.shape == (n[3],n[2])

    dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)
    assert dC_db3.shape == (n[3], 1)

    dZ3_dA2 = W3
    dC_dA2 = W3.T @ dC_dZ3
    assert dC_dA2.shape == (n[2], m)

    return dC_dW3, dC_db3, dC_dA2

def back_propagation_layer_2(propagator_dC_dA2, A1, A2, W2):
    dA2_dZ2 = A2 * (1-A2)
    dC_dZ2 = propagator_dC_dA2 * dA2_dZ2
    assert dC_dZ2.shape == (n[2],m)

    dZ2_dW2 = A1
    assert dZ2_dW2.shape == (n[1],m)

    dC_dW2 = dC_dZ2 @ dZ2_dW2.T
    assert dC_dW2.shape == (n[2],n[1])

    dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)
    assert dC_db2.shape == (n[2],1)

    dZ2_dA1 = W2
    dC_dA1 = dZ2_dA1.T @ dC_dZ2
    assert dC_dA1.shape == (n[1],m)

    return dC_dW2, dC_db2, dC_dA1

def back_propagation_layer_1(propagator_dC_dA1, A1, A0, W1):
    dA1_dZ1 = A1*(1 - A1)
    dC_dZ1 = propagator_dC_dA1*dA1_dZ1
    assert dC_dZ1.shape == (n[1],m)

    dZ1_dW1 = A0
    assert dZ1_dW1.shape == (n[0],m)

    dC_dW1 = dC_dZ1 @ dZ1_dW1.T
    assert dC_dW1.shape == (n[1],n[0])

    dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)
    assert dC_db1.shape == (n[1], 1)

    return dC_dW1, dC_db1

def train():
    global W1, W2, W3, b1, b2, b3

    epochs = 500
    alpha = 0.01
    costs = []

    for epoch in range(epochs):
        Y_hat, cache = feed_forward(A0)
        err = cost(Y_hat, Y)
        costs.append(err)

        dc_dW3, dC_dB3, dC_dA2 = back_propagation_layer_3(Y_hat, Y, m, A2 = cache["A2"], W3 = W3)
        dc_dW2, dC_dB2, dC_dA1 = back_propagation_layer_2(propagator_dC_dA2=dC_dA2, A1 = cache["A1"],
                                                          A2 = cache["A2"], W2 = W2)
        dC_dW1, dC_db1 = back_propagation_layer_1(propagator_dC_dA1=dC_dA1,A1=cache["A1"],
                                                  A0=cache["A0"],W1=W1)
        W3 = W3-(alpha*dc_dW3)
        W2 = W2-(alpha*dc_dW2)
        W1 = W1-(alpha*dC_dW1)

        b3 = b3-(alpha*dC_dB3)
        b2 = b2-(alpha*dC_dB2)
        b1 = b1-(alpha*dC_db1)

        if epoch % 20 == 0:
            print(f"epoch {epoch}: cost = {err:.4f}")
    return costs

costs = train()
print(costs)