import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Activation
import cv2


def mobilnet_block (x, filters, strides):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

input = Input(shape = (256,256,3))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same', use_bias=False)(input)
x = BatchNormalization()(x)
x = ReLU()(x)

x = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters = 128, strides = 2)
x = mobilnet_block(x, filters = 128, strides = 1)
x = mobilnet_block(x, filters = 256, strides = 2)
x = mobilnet_block(x, filters = 256, strides = 1)
x = mobilnet_block(x, filters = 512, strides = 2)

for _ in range (5):
     x = mobilnet_block(x, filters = 512, strides = 1)
x = mobilnet_block(x, filters = 1024, strides = 2)
x = mobilnet_block(x, filters = 1024, strides = 1)

x = GlobalAveragePooling2D()(x)
x = Reshape((1,1,1024))(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=2, kernel_size=1)(x)
x = Reshape((2,))(x)

# output = Activation('linear')(x)

model = Model(inputs=input, outputs=x)

input = cv2.imread('imgs/4.png')
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)[None, ...]/255

classifier = tf.keras.models.load_model('./mobilenet.savedmodel')

#load model pretrained model weights
weight_count = len(model.get_weights())
model.set_weights(classifier.get_weights()[:weight_count])

model.summary()

# print('Weights of first layer:')
# we transpose to compare if they are the same
# print(model.get_weights()[0].transpose(3,2,0,1))

# print('theirs', classifier.predict(input))
output = model.predict(input)
print(output.shape)
print('ours', output)

