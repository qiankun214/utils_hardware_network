该项目用于保存一些在使用神经网络（pytorch框架）时的一些utils代码，尤其是与硬件相关的
这些代码经常被不同的项目使用，因此每次使用都需要重复编写，会浪费精力与时间
因此将这些utils代码在这里进行统一维护，以降低日后的工作量，目前这些utils包括：

- classification_accfunc_top1：计算top1准确率
- return_mnist_dataloader：创建并返回mnist数据集的dataloader
- return_sgd_classification：返回简单分类任务需要的SGD优化器、代价函数和准确率函数
- model_run_one：运行一次前向传播
- naive_train_frame：输入模型，dataloader和工具集（优化器，代价函数和准确率函数），根据指定的参数进行网络训练
- save_model_withoutstructure：保存网络参数（不包括结构）
- load_model_withoutstructure：载入网络参数
- print_model_module：打印所有module的名称
- gpu_to_numpy：将GPU上的Tensor转为CPU上的numpy
- numpy_to_int：将float数据类型的array量化为整数
- conv_place_compute：截取数据和权值的一个xy位置的所有channel数据进行对应位置相乘相加
- return_randomarray_withmax：产生一个指定尺寸，在-max_data ～ max_data随机分布的ndarray
- error_analysis_ndarray：分析两个矩阵的误差和相对误差，给出最大值、平均值和中位值

**submodule的使用**

本项目一般用于作为项目的submodule，这里提供submodule的使用方法，添加为submodule的方式如下：
```shell
git submodule add <仓库地址>
```

在submodule中可以进行本module的各种修改和提交工作。如果需要克隆整个项目（主项目包括submodule），添加递归参数如下所示：
```shell
git clone <主仓库地址> --recursive
```