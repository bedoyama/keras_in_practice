from keras.models import Sequential
from keras.layers import Activation, Dense

# Sequential model
model = Sequential()

# First layer: 12 neurons, each neuron accepts 8 inputs, 
# weights initialized with random uniform distribution in range (-0.05, 0.05)
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))