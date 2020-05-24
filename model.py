import torch as pt
import torchvision as ptv
from utils import load_model_withoutstructure,gpu_to_numpy
import numpy as np

class the_model(pt.nn.Module):

    def __init__(self):
        super(the_model,self).__init__()
        # self.fc1 = pt.nn.Linear(28*28,512)
        self.fc1 = pt.nn.Conv2d(1,128,3,1,1)
        self.fc2 = pt.nn.Conv2d(128,64,3,1,1)
        self.fc3 = pt.nn.Linear(7 * 7 * 64,10)
        self.pool = pt.nn.MaxPool2d(2)

    def forward(self,x):
        x = pt.nn.functional.relu(self.fc1(x))
        x = self.pool(x)

        din = gpu_to_numpy(x)
        np.save("./input512.npy",din)

        x = self.fc2(x)

        dout = gpu_to_numpy(x)
        np.save("./output256.npy",dout)
        weight = gpu_to_numpy(self.fc2.weight.data)
        np.save("./weight.npy",weight)
        bias = gpu_to_numpy(self.fc2.bias.data)
        np.save("./bias.npy",bias)

        x = pt.nn.functional.relu(x)
        x = self.pool(x)
        # print(x.size())
        x = x.view((-1,7 * 7 * 64))
        x = pt.nn.functional.relu(self.fc3(x))
        return x

def get_model(train=True,weight=None,cuda=True):
    model = the_model()
    if train is False:
        model = load_model_withoutstructure(model,weight)
    if cuda is True:
        model = model.cuda()
    return model