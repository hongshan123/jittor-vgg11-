import jittor as jt  # 将 jittor 引入
from jittor import nn, Module  # 引入相关的模块

from jittor.dataset import Dataset # jittor 数据集类
from jittor.dataset.cifar import CIFAR10
from jittor import dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl # 用于绘制 Loss 曲线 和 CIFAR10 数据

from PIL import Image
jt.flags.use_cuda = 1
class VGG(nn.Module):

    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def execute(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = jt.reshape(x, [x.shape[0],-1])
        x = self.classifier(x)
        return x
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.Pool(kernel_size=2, stride=2, op="maximum")]
        else:
            conv2d = nn.Conv(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)
def _vgg(arch, cfg, batch_norm, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model
def vgg11(pretrained=False, **kwargs):
    model = _vgg('vgg11', 'A', False, **kwargs)
    if pretrained: model.load("jittorhub://vgg11.pkl")
    return model
from jittor import transform

# 定义数据处理操作
transform = transform.Compose([
    transform.Resize(224),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
    transform.ToTensor()
])
from PIL import Image
class_to_idx = {'dog': 0, 'cat': 1}
class MyDataset(dataset.Dataset):
    def __init__(self, data_path, transform):
        super().__init__()
        self.data_path = data_path  # 数据集根路径
        self.img_names_labels = []  # 图片文件名及其对应标签的列表

        # 遍历数据集根路径下所有的文件名，并将其与对应的标签加入img_names_labels列表
        for img_name in os.listdir(self.data_path):
            label = img_name.split('.')[0]  # 文件名第一部分为标签信息
            label = class_to_idx[label]
            img_path = os.path.join(self.data_path, img_name)
            self.img_names_labels.append((img_path, label))

        self.transform = transform  # 数据变换处理操作

    def __getitem__(self, index):
        img_path, label = self.img_names_labels[index]
        img = Image.open(img_path)  # 读取图片
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_names_labels)  # 返回数据集长度
dataset = MyDataset(data_path='data/train', transform=transform)
#dataset = MyDataset(data_path = 'data/train', transform = transform) # 实例化数据集加载类 MyDataset 对象
# 创建数据加载器对象
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = dataset.set_attrs(batch_size=1,shuffle=True)
def val(model, val_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        inputs = inputs
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy() == pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
    print('Test Acc =', total_acc / total_num)
model = vgg11() # 初始化模型
model.load_parameters(jt.load("model.pkl"))
data_iter = iter(val_loader) # 读取测试数据
val_data, val_label = next(data_iter) # 取出测试数据
# outputs = model(val_data)
# prediction = np.argmax(outputs.numpy(), axis=1)
classes = ('dog','cat')
# print(classes[int(prediction)])# 输出预测
# print(classes[int(val_label)])# 输出标签
def test():
    i=0
    sum=0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        sum+=1
        outputs = model(inputs)
        prediction = np.argmax(outputs.numpy(), axis=1)
        
        if(prediction==targets):
              i+=1
        if(sum==1000):
            break

    print('acc',i/sum)
test()
              

