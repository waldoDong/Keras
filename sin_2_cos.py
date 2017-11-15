import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
# 这里的TimeDistributed的含义是将一个多维度的序列分开,针对其中的每一个维度进行操作.
from keras.layers import Dense, Activation
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE) / (10*np.pi)
    seq = 4 * xs + 2
    BATCH_START += TIME_STEPS
    return [xs, seq]

model = Sequential()

model.add(Dense(input_shape=[1], units=100),)
model.add(Activation("relu"))

model.add(Dense(input_shape=[100], units=1))



adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)

print('Training ------------')
for step in range(101):

    xs, ys = get_batch()
    cost = model.train_on_batch(xs, ys)
    pred = model.predict(xs)
    # 指定两种不同的颜色,并指定其中一个使用连续的线,另外一个使用间隔线.
    plt.plot(xs[:], ys, 'r', xs[:], pred[:], 'b--')
    plt.draw()
    plt.pause(0.01)
    if step % 10 == 0:
        print('train cost: ', cost)
    if step == 100:
        plt.savefig(filename="./output_result/output_sin2cos.jpg")

model.save(filepath="./model_save/model_sin2cos.h5")
del (model)

model = load_model(filepath="./model_save/model_sin2cos.h5")
