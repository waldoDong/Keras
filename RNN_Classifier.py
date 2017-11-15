from keras.layers import SimpleRNN, Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

# 这两行代码是进行批量数据的操作.
BATCH_INDEX = 0
BATCH_SIZE = 32

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape([-1, 28, 28])/255
X_test = X_test.reshape([-1, 28, 28])/255
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = Sequential()

# 在这里,首先添加了一个SimpleRNN的模型,然后指定这个模型将会输出50个数据,然后输出,每次将读入28个,同时进行28步之后,进行输出.
# 这里的batch_input_shape:每一个cell将会输出50维度,同时第一个28意思是,28个RNN循环,后面的28是每次读入28个数据.
model.add(SimpleRNN(units= 50, batch_input_shape=(None, 28, 28), unroll=True))
model.add(Dense(10))
model.add(Activation("softmax"))

adma = Adam(lr=0.001)
model.compile(optimizer=adma, loss="categorical_crossentropy", metrics=['accuracy'])

for step in range(4001):
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = Y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX = BATCH_INDEX + BATCH_SIZE
    if BATCH_INDEX > X_train.shape[0]:
        BATCH_INDEX = 0

    if step %500 ==0:
        cost, accuracy = model.evaluate(X_test, Y_test, verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

array = np.array(X_test[0, :, :])
plt.imshow(array.reshape(28, 28))
plt.show()
print(model.predict(X_test[0, :, :].reshape([1, 28, 28])))
