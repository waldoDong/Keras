import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# linspace的含义是从-1，到1生成200个数据。
X = np.linspace(-1, 1, 200)
# shuffle 的含义是将整个数据从第0个维度进行随机“拉扯”
np.random.shuffle(X)
Y = 0.5*X + 2+ np.random.normal(0, 0.05, (200,))
plt.scatter(X, Y)
plt.xlabel("initialize picture")
plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# 使用keras建立我的模型
# Sequential的含义是,我建立的这个模型是一个顺序的模型,也就是,我的层是一层一层自己添加上去的.
model = Sequential()
# Dense就是全连接层.
model.add(Dense(output_dim=1, input_dim=1))
# 使用keras必须要进行编译:
model.compile(loss="mse", optimizer="sgd")

# 训练我的神经网络
print("Training ----")
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step%100 == 0:
        print("train cost:", cost)

W, b = model.layers[0].get_weights()
print("W: ", W, "b: ", b)

Y_prediction = model.predict(X)
plt.scatter(X, Y_prediction)
plt.scatter(X, Y)
plt.xlabel("After Training")
plt.show()

