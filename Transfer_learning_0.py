from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as kd
import matplotlib.pyplot as plt

# 这里是一个固定的写法,主要的目的是将load的mnist分别加载到X_train, 和Y_train当中.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 这里是几个操作方法,将数据的X转化为[多少条,大小,大小,深度]的格式,将Y转化为one_hot的形式.
Y_test_tem = Y_test
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
X_train = X_train.reshape([X_train.shape[0], 28, 28, 1])/255
X_test = X_test.reshape([X_test.shape[0], 28, 28, 1])/255

# 加载我的CNN模型.
model_original = load_model(filepath="./model_save/model_CNN.h5")

# 拿到的数据是一个10维度的向量.
get_10dim_output = kd.function([model_original.layers[0].input], [model_original.layers[9].output])

# x_2dim = get_10dim_output([X_train[:10000]])[0]
# print("+++++++++++++++++++++++++\n", x_2dim.shape)

# 分批进行训练和测试


output_x_train = get_10dim_output([X_train[:10000]])[0]
output_x_test = get_10dim_output([X_test])[0]
print(output_x_train.shape)
print(output_x_test.shape)
print("---------------", "代码处理完成!!")

model = Sequential()
model.add(Dense(128, input_shape=(10, ), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Activation("softmax"))



adma = Adam(lr=0.0001)
#output_x_train = get_10dim_output([X_train[10000:20000]])[0]
output_x_train = get_10dim_output([X_train[10000:20000]])[0]

model.compile(optimizer=adma, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(output_x_train, Y_train[10000:20000], epochs=15, batch_size=32)

# for j in range(15):
#     for i in range(1, 600):
#         tem_num_1 = 100*(i-1)
#         tem_num_2 = 100*i
#         output_x_train = get_10dim_output([X_train[tem_num_1:tem_num_2]])[0]
#         output_y_train = Y_train[tem_num_1:tem_num_2]
#         loss = model.train_on_batch(output_x_train, output_y_train)
#         print("---------------------", i, loss)


loss, accuracy = model.evaluate(output_x_test, Y_test)
print("lose: %s, accuracy: %f" % (loss, accuracy))

