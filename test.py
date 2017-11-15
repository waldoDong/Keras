
from keras.models import Sequential, load_model


model = load_model(filepath="./model_save/model_sin2cos.h5")
weight, bias = model.layers[0].get_weights()
print(type(weight))
print(weight.shape)
print(bias)