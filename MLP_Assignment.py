import math
XOR_data = [(0,0,0), (0,1,1), (1,0,1),(1,1,0)]

class MLP:
    def __init__ (self):
        # initialize weighs and biases
        # 2 inputs, 2 hidden layers, 1 output
        # weight from Input to Hidden Layer
        self.w13 = 0.2
        self.w14 = 0.5
        self.w23 = 0.7
        self.w24 = 0.1
        # weight from Hidden Layer to Output
        self.w35 = 0.3
        self.w45 = 0.4
        # bias
        self.b3 = 0.8
        self.b4 = -0.3
        self.b5 = 0.5
        self.a = 0.6 # learning rate

    def sigmoid(self,x):
        return 1/(1 + math.exp(-x))
    
    # Input to hidden layer
    def forward(self,x1,x2):
        h3 = (x1 * self.w13) + (x2 * self.w23) + self.b3
        h4 = (x1 * self.w14) + (x2 * self.w24) + self.b4
        s3 = self.sigmoid(h3)
        s4 = self.sigmoid(h4)
        # all neuron to output
        h5 = (s3 * self.w35) + (s4*self.w45) + self.b5
        o5 = self.sigmoid(h5)
        return s3,s4,o5

    def backward(self,target,x1, x2, o5, s4,s3):
        # calculate output error
        errO = o5 * (1-o5)*(target-o5)
        # update w hidden - output
        self.w35 = self.w35 +(self.a * s3 * errO)
        self.w45 = self.w45 +(self.a * s4 * errO)
        # update bias
        self.b5 = self.b5 + (self.a * errO)
        # calculate hidden error
        errH1 = s3 * (1 - s3) * (errO * self.w35)
        errH2 = s4 * (1 - s4) * (errO * self.w45)
        # update w input - hidden
        self.w13 = self.w13 + (self.a * x1 * errH1)
        self.w23 = self.w23 + (self.a * x2 * errH1)
        self.w14 = self.w14 + (self.a * x1 * errH2)
        self.w24 = self.w24 + (self.a * x2 * errH2)
        # update biases
        self.b3 = self.b3 + (self.a * errH1)
        self.b4 = self.b4 + (self.a * errH2)

mlp = MLP()
#loop for XOR_data
for epoch in range (2000):
    for x1, x2, target in XOR_data:
        s3, s4, o5 = mlp.forward(x1,x2)
        mlp.backward(target,x1,x2,o5,s4,s3)
        if epoch % 500 == 0:
            error = (target - o5)**2
            print(f"Epoch: {epoch}, Input: {x1} {x2}, Target: {target}, Predicted: {o5:.4f} Squared Error: {error:.4f}")

# print final result
print("\nFinal Result:")
total_error = 0
for x1, x2, target in XOR_data:
    s3, s4, o5 = mlp.forward(x1,x2)
    error = (target-o5)**2
    total_error = total_error + error
    print(f"Input: {x1} {x2}, Target: {target}, Predicted: {o5:.4f} Squared Error: {error:.4f}")
print(f"Sum of Squared Errors: {total_error:.4f}")


