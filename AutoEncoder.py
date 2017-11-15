from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as kd
import matplotlib.pyplot as plt

# 这里是一个固定的写法,主要的目的是将load的mnist分别加载到X_train, 和Y_train当中.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
Y_test_tem = Y_test
# 这里是几个操作方法,将数据的X转化为[多少条,大小,大小,深度]的格式,将Y转化为one_hot的形式.
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
X_train = X_train.reshape([X_train.shape[0], 28, 28, 1])/255
X_test = X_test.reshape([X_test.shape[0], 28, 28, 1])/255

# 首先声明一个Sequential类型的模型.
model = Sequential()

# 指定卷积层的具体的参数,输入的图片是(28,28,1),一共有32个滤波器,卷积核的大小是5,步长为1,使用padding模式,同时指定通道数在最后.
model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, strides=1, padding="same", data_format="channels_last"))
model.add(MaxPool2D(pool_size=2, strides=2, padding="same", data_format="channels_last"))
model.add(Activation("relu"))

model.add(Conv2D(batch_size=64, filters=10, kernel_size=5, strides=1, padding="same", data_format="channels_last"))
model.add(MaxPool2D(pool_size=2, strides=2, padding="same", data_format="channels_last"))
model.add(Activation("relu"))

# 注意这里的Flatten(),这个函数的意思将不同维度的数据强行拉平.
model.add(Flatten())
# Dense是全连接层.
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Dense(10))
# 最后的sotfmax属于激活函数的一种.
model.add(Activation("softmax"))

# 定义一个优化器,使用Adam, 同时指定学习速率是0.0001.
adma = Adam(lr=0.0001)
# 编译整个模型,同时定义loss函数式"categorical_crossentropy", metrics:指标:这里指定使用什么指标来衡量,一般来说accuracy,
# 注意在这里不指定accuracy的话,那么后面将不能输出具体的accuracy.
model.compile(optimizer=adma, loss="categorical_crossentropy", metrics=["accuracy"])

# 这里的batch_size代表的是多少个图片执行一次梯度下降.
model.fit(X_train, Y_train, epochs=15, batch_size=32)
loss, accuracy = model.evaluate(X_test, Y_test)
print("lose: %s, accuracy: %f" %(loss, accuracy))

model.save(filepath="./model_save/model_AutoEncoder.h5")

output_2dim = kd.function([model.layers[0].input], [model.layers[9].output])
x_2dim = output_2dim([X_test])[0]

plt.scatter(x_2dim[:, 0], x_2dim[:, 1], c=Y_test_tem)
plt.colorbar()
plt.savefig(filename="./output_result/output_AutoEncoder.jpg")
plt.show()



