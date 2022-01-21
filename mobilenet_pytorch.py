import torch.nn as nn
import torch
from torchsummary import summary
import numpy as np

import tensorflow as tf
import cv2
from matplotlib import pyplot as plt


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, bias=False),
                nn.BatchNorm2d(oup, eps=.001, momentum=.01),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride, padding=1):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp, eps=.001, momentum=.01),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, eps=.001, momentum=.01),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            nn.ConstantPad2d((0,1,0,1), 0),
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            nn.ConstantPad2d((0,1,0,1), 0),
            conv_dw(64, 128, 2, 0),
            conv_dw(128, 128, 1),
            nn.ConstantPad2d((0,1,0,1), 0),
            conv_dw(128, 256, 2, 0),
            conv_dw(256, 256, 1),
            nn.ConstantPad2d((0,1,0,1), 0),
            conv_dw(256, 512, 2, 0),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            nn.ConstantPad2d((0,1,0,1), 0),
            conv_dw(512, 1024, 2, 0),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(),
            nn.Conv2d(1024, 2, 1, 1, 0)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 2)
        return x


def convert_tf_classifier(classifier):
    model = MobileNetV1()

    pretrained_params = vars(classifier)["_serialized_attributes"]["trainable_variables"]
    total_params = vars(classifier)["_serialized_attributes"]["variables"]

    moving_means = [i.numpy() for i in total_params if "moving_mean" in i.name]
    moving_vars = [i.numpy() for i in total_params if "moving_variance" in i.name]
    
    for i, (name, param) in enumerate(model.named_parameters()):
        # print(name, i)
        block_idx = name.split(".")[1]
        inter_block_idx = name.split(".")[2]

        pre_param = pretrained_params[i].numpy()
        
        if len(pre_param.shape) == 4: # Conv layer
            if block_idx in ["1", "21"] or inter_block_idx == "3":
                pre_param = pre_param.transpose(3,2,0,1)
            else: # inter_block_idx == "0"
                pre_param = pre_param.transpose(2,3,0,1)

        # print(pretrained_params[i].name)

        param.data = torch.from_numpy(pre_param)

        if inter_block_idx in ["1", "4"] and "bias" in name:
            bn_module = model.model[int(block_idx)][int(inter_block_idx)]
            bn_module.running_mean = torch.from_numpy(moving_means.pop(0))
            bn_module.running_var = torch.from_numpy(moving_vars.pop(0))

        # print(pre_param.shape)
        # print("Torch: ", param.data.dtype)
        # print(pretrained_params[i].numpy().shape)
        # print("TF: ", pretrained_params[i].numpy())

    return model

if __name__ == '__main__':
    input = cv2.imread('imgs/5.png')
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)[None, ...]/255

    classifier = tf.keras.models.load_model('./mobilenet.savedmodel')

    print('theirs', classifier.predict(input))

    #load model pretrained model weights
    model = convert_tf_classifier(classifier)

    # save pytorch model
    torch.save(model.state_dict(), './classifier.pth')

    # torch needs input in form [B, C, H, W] but input is [B, H, W, C]
    input = torch.tensor(input.transpose(0,3,1,2), dtype=torch.float32)

    model.eval()
    output = model.forward(input)
    print('ours', output)
    summary(model, (3, 256, 256))
