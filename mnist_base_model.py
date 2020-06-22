import torch as pt
from .utils import load_model_withoutstructure

class mnist_base_model(pt.nn.Module):
    def __init__(self):
        super(mnist_base_model, self).__init__()
        self.conv1 = pt.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = pt.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = pt.nn.Linear(4*4*50, 500)
        self.fc2 = pt.nn.Linear(500, 10)

    def forward(self, x):
        x = pt.nn.functional.relu(self.conv1(x))
        x = pt.nn.functional.max_pool2d(x, 2, 2)
        x = pt.nn.functional.relu(self.conv2(x))
        x = pt.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)

        x = pt.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # return pt.nn.functional.softmax(x, dim=1)
        return x

def get_mnist_base_model(train=True,weight=None,cuda=True):
    model = mnist_base_model()
    if train is False:
        model = load_model_withoutstructure(model,weight)
    if cuda is True:
        model = model.cuda()
    return model