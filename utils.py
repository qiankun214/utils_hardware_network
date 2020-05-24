import torch as pt
import torchvision as ptv
import copy
import numpy as np

# top1 accuracy compute
def classification_accfunc_top1(dout,label):
    hi = pt.argmax(dout, dim=1)
    result = (hi == label).float().mean()
    return result.item()

# return mnist dataloader (train,test)
def return_mnist_dataloader(root,batch):
    train_set = ptv.datasets.MNIST(root,train=True,transform=ptv.transforms.ToTensor(),download=True)
    test_set = ptv.datasets.MNIST(root,train=False,transform=ptv.transforms.ToTensor(),download=True)
    train_dataset = pt.utils.data.DataLoader(train_set,batch_size=batch)
    test_dataset = pt.utils.data.DataLoader(test_set,batch_size=batch)
    return train_dataset,test_dataset

# return sgd's optimizer,loss function and accuracy computer
def return_sgd_classification(model,lr,cuda=True):
    optimizer = pt.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    if cuda:
        lossfunc = pt.nn.CrossEntropyLoss().cuda()
    else:
        lossfunc = pt.nn.CrossEntropyLoss()
    return optimizer,lossfunc,classification_accfunc_top1

def model_run_one(model,data):
    return model(data)

# just train in naive way
def naive_train_frame(model,dataloader,tools,epoch=4,printcycyle=100,cuda=True,train=True):
    if train is False:
        for i,data in enumerate(dataloader):
            if cuda:
                inputs,labels = [x.cuda() for x in data]
            else:
                inputs,labels = data
            model_run_one(model,inputs)
            break
        return model
    if len(tools) != 3:
        raise ValueError("FATAL tools length not 3")
    optimizer,lossfunc,accfunc = tools
    for x in range(epoch):
        for i,data in enumerate(dataloader):
            optimizer.zero_grad()
            if cuda:
                inputs,labels = [x.cuda() for x in data]
            else:
                inputs,labels = data
            outputs = model(inputs)
            loss = lossfunc(outputs,labels)
            loss.backward()
            optimizer.step()
            if i % printcycyle == 0:
                print(x,"-",i,":",accfunc(outputs,labels))
    return model

# save model
def save_model_withoutstructure(model,path):
    pt.save(model.state_dict(), path)

# load model
def load_model_withoutstructure(model,path):
    model.load_state_dict(pt.load(path))
    return model

# print submodule name in model
def print_model_module(model):
    for n,_ in model.named_module():
        print(n)

# move tensor from gpu to numpy
def gpu_to_numpy(data):
    return data.cpu().detach().clone().numpy()

# change data from float to int,not type
def numpy_to_int(data,max_data,factor):
    data = np.floor(copy.deepcopy(data) * factor)
    data[data > abs(max_data)] = abs(max_data)
    data[data < -abs(max_data)] = -abs(max_data)
    return data

def conv_place_compute(weight,data,wplace=(0,0),dplace=(0,0)):
    din = data[:,dplace[0],dplace[1]]
    wei = weight[:,:,wplace[0],wplace[1]]
    return np.dot(din,wei.T)

def return_randomarray_withmax(shape,max_data=1):
    return (np.random.rand(*shape) - 0.5) * max_data * 2

def error_analysis_ndarray(data,ref):
    if data.shape != ref.shape:
        raise ValueError("ERROR:shape not match")
    error = np.abs(data - ref)
    re_error = error / np.abs(ref)
    print("Info:max error = {},mean error = {},median error = {}".format(
        np.max(error),np.mean(error),np.median(error)
    ))
    print("Info:max relative error = {},mean relative error = {},median relative error = {}".format(
        np.max(re_error),np.mean(re_error),np.median(re_error)
    ))
    return error,re_error