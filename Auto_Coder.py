import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Activation, MaxPool2D, Flatten, Reshape
import matplotlib.pyplot as plt

np.random.seed(1337)  # for reproducibility

(x_train, _), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5     # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5        # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
print(x_train.shape)
print(x_test.shape)

# in order to plot in a 2D figure
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(28, 28, 1))

# encoder layers
# encoded = Dense(128, activation='relu')(input_img)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(10, activation='relu')(encoded)
# encoder_output = Dense(encoding_dim)(encoded)

encoded = Conv2D(kernel_size=3, strides=2, padding="same", filters=128)(input_img)
encoded = Activation("relu")(encoded)
encoded = MaxPool2D(pool_size=2, strides=2, padding="same")(encoded)

encoded = Conv2D(kernel_size=3, strides=2, padding="same", filters=64)(encoded)
encoded = Activation("relu")(encoded)
encoded = MaxPool2D(pool_size=2, strides=2, padding="same")(encoded)

encoded = Conv2D(kernel_size=3, strides=1, padding="same", filters=10)(encoded)
encoded = Activation("tanh")(encoded)
encoded = MaxPool2D(pool_size=2, strides=2, padding="same")(encoded)

encoded = Flatten()(encoded)

encoder_output = Dense(encoding_dim)(encoded)



# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)
decoded = Reshape((28, 28, 1))(decoded)

# construct the autoencoder model
autoencoder = Model(inputs=input_img, outputs=decoded)

# construct the encoder model for plotting
encoder = Model(inputs=input_img, outputs=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True)

# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.savefig(filename="./output_result/output_AutoCoder.jpg")
plt.show()
