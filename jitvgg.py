import jittor as jt
from jittor import nn, Module
from jittor import dataset
import os
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
dataloader = dataset.set_attrs(batch_size=4,shuffle=True)
batch_size = 1  # batchsize 大小
learning_rate = 0.01  # 学习率
momentum = 0.9  # 优化器动量
weight_decay = 1e-4  # 正则化因子
epochs = 10  # 训练epoch数
losses = []  # 损失列表
losses_idx = []  # 存储损失为第几个epoch第几个batch损失下标列表
model = vgg11() # 初始化模型
# model.load_parameters(jt.load("model.pkl"))
#如果已有训练的模型将上面的代码的注释取消，加载已有模型
optimizer = nn.SGD(model.parameters(), learning_rate)  # 定义优化器
# def train(model, train_loader, optimizer, epoch, losses, losses_idx):
#     model.train()
#     lens = len(train_loader)
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         outputs = model(inputs)
#         loss = nn.cross_entropy_loss(outputs, targets)
#         optimizer.step(loss)
#         losses.append(loss.numpy()[0])
#         losses_idx.append(epoch * lens + batch_idx)

#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx, len(train_loader),
#                 100. * batch_idx / len(train_loader), loss.numpy()[0]))
def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        

        outputs = model(inputs)  # 前向计算
        loss = nn.cross_entropy_loss(outputs, targets)  # 计算loss
        optimizer.step(loss)  # 自动进行梯度清零、反向传播和梯度更新

        losses.append(loss.item())
        losses_idx.append(epoch * lens + batch_idx)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))               
for epoch in range(epochs):
   train(model, dataloader, optimizer, epoch, losses, losses_idx)
model_path = './model.pkl'
model.save(model_path)
