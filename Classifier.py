from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential # 一层一层去建立神经层
from keras.layers import Dense, Activation # Dende是全连接神经网络, Activation代表激活函数
from keras.optimizers import RMSprop, SGD, Adam # RMSprop是一种优化器


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# 在这里进行了正则化,但是不是很清楚正则化的具体的目的,一会进行一下实验.
X_train = X_train.reshape(X_train.shape[0], -1)/255
X_test = X_test.reshape(X_test.shape[0], -1)/255
# 下面这两句话,将编码变成one_hot编码.
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

model = Sequential([Dense(32, input_dim=784),
                    Activation("relu"),
                    Dense(10),
                    Activation("softmax")])
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001)
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=2, batch_size=32)

loss, accuracy = model.evaluate(X_test, Y_test)
print("test loss: ", loss)
print("test accuracy: ", accuracy)

