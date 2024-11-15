import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import functional as F
import torch
import numpy as np
import os
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPrune
from torchvision import datasets, transforms
import copy
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
folder_path = '/data1/Segcam_code/segcam/tutorials/test_car/'
folder_contents = os.listdir(folder_path)

# ts = 5
# for t in range(ts):  # 权重
names = [] # 剪枝卷积层名字
channel_prunes = [] # 通道剪枝列表
channel_retains = [] # 通道保存列表
t=290
weights_path = "/data1/Deeplabv3/save_weights_car/model_" + str(t) + ".pth"

for item in folder_contents:
    img_path = folder_path + item
    image = np.array(Image.open(img_path))
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = deeplabv3_resnet50(pretrained=False, progress=False)
    
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    model.load_state_dict(weights_dict)
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    output = model(input_tensor)

    class SegmentationModelOutputWrapper(torch.nn.Module):
        def __init__(self, model):
            super(SegmentationModelOutputWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)["out"]

    model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor)

    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    car_category = sem_class_to_idx["car"]
    car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
    car_mask_float = np.float32(car_mask == car_category)

    both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
    Image.fromarray(both_images)

    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()

        def __call__(self, model_output):
            return (model_output[self.category, :, :] * self.mask).sum()

    def select_top_channels(gray_list, T):
   
        # 选择前 T 比例的通道，基于每个通道的总和值。前 T 比例的通道标号，从大到小排序。

        # 将 gray_list 转换为 numpy 数组
        if isinstance(gray_list, list) and isinstance(gray_list[0], Image.Image):
            gray_list = np.array([np.array(img) for img in gray_list])

        # 检查转换结果是否为三维数组
        if gray_list.ndim != 3:
            raise ValueError("gray_list 应该是一个形状为 (C, W, H) 的三维数组")

        # 求所有值的总和
        total_sum = np.sum(gray_list)
        
        # 求每个通道的总和
        channel_sums = np.sum(gray_list, axis=(1, 2))
        
        # 对每个通道总和值从大到小排序，并得到排序后的索引
        sorted_indices = np.argsort(channel_sums)[::-1]
        sorted_sums = channel_sums[sorted_indices]
        
        # 计算前 T 比例的总和值
        target_sum = T * total_sum
        
        # 找到满足前 T 比例的通道
        cumulative_sum = 0
        selected_channels = []
        
        for i, sum_val in enumerate(sorted_sums):
            cumulative_sum += sum_val
            selected_channels.append(sorted_indices[i])
            if cumulative_sum >= target_sum:
                break
        
        return selected_channels

    # 自定义转换函数，用于同时处理图像和标签
    class SegmentationTransform:
        def __call__(self, image, target):
            image = F.resize(image, (256, 256))  # 调整图像大小
            image = F.to_tensor(image)  # 将图像转换为张量
            target = F.resize(target, (256, 256), interpolation=F.InterpolationMode.NEAREST)  # 调整标签大小
            target = torch.as_tensor(np.array(target), dtype=torch.long)  # 将标签转换为张量并确保为整型
            return image, target
    
    def get_all_channel_counts(model):
        channel_counts = []  # 存储所有通道数量

        def get_channels(layer):
            for layer in layer.children():
                if isinstance(layer, nn.Conv2d):  # 如果是卷积层
                    channel_counts.append(layer.out_channels)  # 记录输出通道数
                get_channels(layer)  # 递归遍历子层

        get_channels(model)
        return channel_counts

    # 加载预训练的DeepLabV3模型并将其放到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取所有卷积层及其通道的热图
    T = 1-0.2
    model_channels = get_all_channel_counts(model)

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            # 使用 GradCAM 生成热力图
            target_layers = [layer]
            targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
            with GradCAMPrune(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
                # 生成热力图，基于当前卷积层的当前通道
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                # grayscale_cam_squeezed = grayscale_cam
                grayscale_cam_squeezed = np.squeeze(grayscale_cam)
                
                gray_list = []
                # 将热力图叠加到原始图像上
                for cam_channel in range(grayscale_cam_squeezed.shape[0]):
                    # 激活图
                    mapped_matrix = (grayscale_cam_squeezed[cam_channel] * 255).astype(np.uint8)
                    gray_image = Image.fromarray(mapped_matrix, mode='L')
                    gray_list.append(gray_image)
                    full_save = True
                    if full_save:
                        # 保存热力图和叠加图像
                        cam_image = show_cam_on_image(rgb_img, grayscale_cam_squeezed[cam_channel], use_rgb=True)
                        hot = Image.fromarray(cam_image)
                        out = Image.fromarray(car_mask_uint8)
                        ppath = '/data1/Segcam_code/segcam/tutorials/test_out/weight'+ str(t) + '-' + name
                        if not os.path.exists(ppath):
                            os.makedirs(ppath)
                            gray_image.save(ppath + "/Channel" + str(cam_channel) + '-' + item  )
                            print('当前'+ name + '第' + str(cam_channel) + '通道')
                    
                    # ALL SAVE
                    if full_save:
                        ppath = '/data1/Segcam_code/segcam/tutorials'
                        path1 = ppath + '/test_out/layer_' + str(ii) + '/hot_weight_' + str(t) + '/'
                        path2 = ppath + '/test_out/layer_' + str(ii) + '/out_hot_weight_' + str(t) + '/'
                        path3 = ppath + '/test_out/layer_' + str(ii) + '/gray_hot_weight_' + str(t) + '/'
                        ii=ii+1
                        if not os.path.exists(path1):
                            os.makedirs(path1)
                        hot.save(path1 + "/" + item)
                        if not os.path.exists(path2):
                            os.makedirs(path2)
                        out.save(path2 + "/" + item)
                        if not os.path.exists(path3):
                            os.makedirs(path3)
                        gray_image.save(path3 + "/" + item)
                        print(ii)
                
                # 保留下来的通道标号
                Channel_retain = select_top_channels(gray_list,T)
                Channel_retain.sort() 
                # 获取所有通道的标号集合
                all_channels = set(range(len(gray_list)))
                # 计算被丢弃的通道标号
                Channel_prune = list(all_channels - set(Channel_retain))
                names.append(name)
                channel_prunes.append(Channel_prune)
                channel_retains.append(Channel_retain)
                print(f"Length of channel_prunes: {len(channel_prunes)}")

# 存为本地变量
with open('/data1/Segcam_code/segcam/tutorials/data.pkl', 'wb') as f:
    pickle.dump((names, channel_prunes, channel_retains), f)



# 读取数据
with open('/data1/Segcam_code/segcam/tutorials/data.pkl', 'rb') as f:
    layer_names, channels_to_prune_list, channel_retains = pickle.load(f)
# layer_names = [n.replace("model.", "") for n in layer_names]




def prune_layers(model, layer_names, channels_to_prune_list):

    # 遍历模型中的每一层
    for name, module in model.named_modules():
        # print(name)
        if name in layer_names:
            
            # 获取该层的索引位置
            layer_index = layer_names.index(name)
            channels_to_prune = channels_to_prune_list[layer_index]
            
            # 检查该层并将所有相关参数置为0
            if isinstance(module, torch.nn.Conv2d):
                # 对于Conv2d层，将权重和偏置置为0
                module.weight.data.fill_(0)
                if module.bias is not None:
                    module.bias.data.fill_(0)
                # print(f"Pruned Conv2d layer {name} - set weight and bias to 0")
            
            elif isinstance(module, torch.nn.BatchNorm2d):
                # 对于BatchNorm2d层，将所有参数置为0
                module.weight.data.fill_(0)
                module.bias.data.fill_(0)
                module.running_mean.fill_(0)
                module.running_var.fill_(0)
                # print(f"Pruned BatchNorm2d layer {name} - set weight, bias, running_mean, running_var to 0")
            

    torch.save(model.state_dict(), '/data1/Segcam_code/segcam/tutorials/pruned_model_weights_saved.pth')  
    return model

# 指定将权重加载到设备0（或者使用CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载权重
model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

# 调用函数进行剪枝
pruned_model = prune_layers(model, layer_names, channels_to_prune_list)

# 剪枝后权重
prune_weight_path = '/data1/Segcam_code/segcam/tutorials/pruned_model_weights_saved.pth'




# 创建剪枝后的空Resnet结构
# --------------------------------------------------------------------------

for layer in pruned_model.modules():
    if isinstance(layer, nn.Conv2d):
        print(f"Layer: {layer}, Out Channels: {layer.out_channels}")

state_dict = pruned_model.state_dict()
for name, param in state_dict.items():
    print(f"{name}: {param.size()}")
print(model)
print('1')
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def create_pruned_conv_layer(original_layer, in_channels, out_channels):
    """创建剪枝后的卷积层"""
    new_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=original_layer.kernel_size,
        stride=original_layer.stride,
        padding=original_layer.padding,
        dilation=original_layer.dilation,
        groups=original_layer.groups,
        bias=original_layer.bias is not None
    )
    # 强制替换卷积层的权重和偏置，确保数据尺寸匹配
    new_layer.weight.data.copy_(original_layer.weight.data[:out_channels, :in_channels, :, :])
    if original_layer.bias is not None:
        new_layer.bias.data.copy_(original_layer.bias.data[:out_channels])
    return new_layer

def create_pruned_bn_layer(original_layer, channels_to_prune):
    """创建剪枝后的批归一化层，并裁剪参数"""
    # 剪枝后的通道索引
    remaining_channels = [i for i in range(original_layer.num_features) if i not in channels_to_prune]
    num_features = len(remaining_channels)
    
    new_bn = nn.BatchNorm2d(num_features=num_features)
    
    # 强制替换BN层的参数
    new_bn.weight.data.copy_(original_layer.weight.data[remaining_channels])
    new_bn.bias.data.copy_(original_layer.bias.data[remaining_channels])
    new_bn.running_mean.data.copy_(original_layer.running_mean[remaining_channels])
    new_bn.running_var.data.copy_(original_layer.running_var[remaining_channels])
    
    return new_bn

def create_pruned_model(layer_names, channels_to_prune_list):
    # 加载原始模型并初始化新模型
    original_model = deeplabv3_resnet50(pretrained=True)
    new_model = deeplabv3_resnet50(pretrained=False)

    # 记录每一层剪枝后的输出通道数
    previous_out_channels = {}

    # 遍历每个原模型层，并根据剪枝配置创建新的层
    for name, layer in original_model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            # 检查是否需要剪枝
            if name in layer_names:
                channels_to_prune = channels_to_prune_list[layer_names.index(name)]
                out_channels = layer.out_channels - len(channels_to_prune) if hasattr(layer, 'out_channels') else layer.num_features - len(channels_to_prune)
            else:
                out_channels = layer.out_channels if hasattr(layer, 'out_channels') else layer.num_features

            # 获取当前层的输入通道数量
            in_channels = previous_out_channels.get(name.split('.')[0], layer.in_channels if hasattr(layer, 'in_channels') else out_channels)

            # 根据层类型创建新的层
            if isinstance(layer, nn.Conv2d):
                pruned_layer = create_pruned_conv_layer(layer, in_channels, out_channels)
            elif isinstance(layer, nn.BatchNorm2d):
                pruned_layer = create_pruned_bn_layer(layer, channels_to_prune)
            elif isinstance(layer, nn.Linear):
                pruned_layer = nn.Linear(in_features=in_channels, out_features=out_channels)
                pruned_layer.weight.data.copy_(layer.weight.data[:out_channels, :in_channels])
                if layer.bias is not None:
                    pruned_layer.bias.data.copy_(layer.bias.data[:out_channels])
            else:
                pruned_layer = layer

            # 更新新模型中的对应层
            parent_module, attr = new_model, name.split('.')
            for a in attr[:-1]:
                parent_module = getattr(parent_module, a)
            setattr(parent_module, attr[-1], pruned_layer)

            # 更新当前层的输出通道数
            previous_out_channels[name] = out_channels
        else:
            # 如果当前层不是卷积、BN或全连接层，保持原有结构
            parent_module, attr = new_model, name.split('.')
            for a in attr[:-1]:
                parent_module = getattr(parent_module, a)
            setattr(parent_module, attr[-1], layer)

    return new_model







# 使用 layer_names 和 channels_to_prune_list 创建全新的剪枝模型
pruned_model = create_pruned_model(layer_names, channels_to_prune_list)



pruned_model_channels = get_all_channel_counts(pruned_model) # 计算通道数量
state_dict = pruned_model.state_dict()
for name, param in state_dict.items():
    print(f"{name}: {param.size()}")
print('1')
rate = sum(pruned_model_channels)/sum(model_channels)
print('剪枝前通道数:' + str(sum(model_channels)))
print('剪枝后通道数:' + str(sum(pruned_model_channels)))
print('通道剪枝率:' + str(1-rate))
# print(model_channels)
# print(pruned_model_channels)

# for name, module in pruned_model.named_modules():
#     if isinstance(module, torch.nn.Conv2d):
#         print(f"Layer name: {name}, Layer type: {module}")
print('1')

# def update_bn_layers_with_prev_conv_out_channels(model):
#     """
#     遍历模型中的所有卷积层和批归一化层，更新批归一化层的 num_features 参数
#     为上一层卷积层的输出通道数。
#     """
#     conv_out_channels = None  # 初始化一个变量，用于保存上一层卷积层的输出通道数

#     # 遍历模型中的所有模块
#     for name, module in model.named_modules():
#         # 找到卷积层并保存其输出通道数
#         if isinstance(module, nn.Conv2d):
#             conv_out_channels = module.out_channels
#         # 找到 BN 层并更新其 num_features
#         elif isinstance(module, nn.BatchNorm2d) and conv_out_channels is not None:
#             module.num_features = conv_out_channels
#             # print(name)

#     return model

# 更新模型中的 BN 层
# pruned_model = update_bn_layers_with_prev_conv_out_channels(pruned_model)
# state_dict = pruned_model.state_dict()
# for name, param in state_dict.items():
#     print(f"{name}: {param.size()}")
# print(pruned_model)
print('1')

# 创建剪枝后的空Resnet结构
# --------------------------------------------------------------------------
import torch
import torch.nn as nn

import torch
import torch.nn as nn

def load_pruned_weights_from_original(model, prune_model, weight_file, layer_names, channels_to_prune_list):
    # lqs
    # 还要再
    # 加载原始模型的权重
    checkpoint = torch.load(weight_file, map_location='cuda')  # 加载到 GPU
    
    # 获取模型的所有层
    model_layers = dict(model.named_modules())
    prune_layers = dict(prune_model.named_modules())
    
    previous_channels_to_prune = None  # 用于存储上一层的通道索引
    
    with torch.no_grad():  # 关闭梯度计算
        for name, module in model.named_modules():
            # 判断当前层是否需要剪枝
            if name in layer_names:
                # 获取当前层在 layer_names 中的索引位置
                idx = layer_names.index(name)
                channels_to_prune = channels_to_prune_list[idx]
                previous_channels_to_prune = channels_to_prune  # 更新上一层的通道索引
            else:
                channels_to_prune = previous_channels_to_prune  # 不在 layer_names 中则复用上一层的通道索引

            # 获取原始模型和剪枝模型的对应层
            layer = model_layers.get(name, None)
            prune_layer = prune_layers.get(name, None)

            if layer is None or prune_layer is None:
                continue  # 如果找不到对应的层，跳过

            # 根据层的类型处理权重参数
            if isinstance(layer, nn.Conv2d):
                if channels_to_prune is not None:
                    pruned_weight = layer.weight[channels_to_prune, :, :, :].detach().clone()
                    pruned_bias = layer.bias[channels_to_prune].detach().clone() if layer.bias is not None else None

                    # 强制覆盖，忽略尺寸不匹配的错误
                    try:
                        prune_layer.weight.copy_(pruned_weight)
                    except RuntimeError:
                        prune_layer.weight.data = pruned_weight  # 强制覆盖
                    if pruned_bias is not None:
                        try:
                            prune_layer.bias.copy_(pruned_bias)
                        except RuntimeError:
                            prune_layer.bias.data = pruned_bias  # 强制覆盖

            elif isinstance(layer, nn.BatchNorm2d):
                if channels_to_prune is not None:
                    pruned_bn_weight = layer.weight[channels_to_prune].detach().clone()
                    pruned_bn_bias = layer.bias[channels_to_prune].detach().clone() if layer.bias is not None else None
                    pruned_bn_running_mean = layer.running_mean[channels_to_prune].detach().clone()
                    pruned_bn_running_var = layer.running_var[channels_to_prune].detach().clone()

                    # 强制覆盖，忽略尺寸不匹配的错误
                    try:
                        prune_layer.weight.copy_(pruned_bn_weight)
                    except RuntimeError:
                        prune_layer.weight.data = pruned_bn_weight  # 强制覆盖
                    if pruned_bn_bias is not None:
                        try:
                            prune_layer.bias.copy_(pruned_bn_bias)
                        except RuntimeError:
                            prune_layer.bias.data = pruned_bn_bias  # 强制覆盖
                    try:
                        prune_layer.running_mean.copy_(pruned_bn_running_mean)
                    except RuntimeError:
                        prune_layer.running_mean.data = pruned_bn_running_mean  # 强制覆盖
                    try:
                        prune_layer.running_var.copy_(pruned_bn_running_var)
                    except RuntimeError:
                        prune_layer.running_var.data = pruned_bn_running_var  # 强制覆盖

            elif isinstance(layer, nn.Linear):
                if channels_to_prune is not None:
                    pruned_weight = layer.weight[:, channels_to_prune].detach().clone()
                    pruned_bias = layer.bias.detach().clone() if layer.bias is not None else None

                    # 强制覆盖，忽略尺寸不匹配的错误
                    try:
                        prune_layer.weight.copy_(pruned_weight)
                    except RuntimeError:
                        prune_layer.weight.data = pruned_weight  # 强制覆盖
                    if pruned_bias is not None:
                        try:
                            prune_layer.bias.copy_(pruned_bias)
                        except RuntimeError:
                            prune_layer.bias.data = pruned_bias  # 强制覆盖

            else:
                continue  # 对于其他类型层，不进行操作

    return prune_model






# 加载原始模型和剪枝后的模型
original_model = deeplabv3_resnet50(pretrained=True)

# 加载剪枝后的模型权重
pruned_model = load_pruned_weights_from_original(original_model, pruned_model, weights_path, layer_names, channel_retains)
state_dict = pruned_model.state_dict()
遍历并打印每一层的权重尺寸
for name, param in state_dict.items():
    print(f"{name}: {param.size()}")
# for name, layer in pruned_model.named_modules():
#     if isinstance(layer, torch.nn.Conv2d):
#         print(f"Layer {name}: in_channels={layer.in_channels}, out_channels={layer.out_channels}")

# 查看更新后的剪枝模型结构，确保权重已更新
# print(pruned_model)
print('1')
# 保存剪枝后权重
# ----------------------------------------------------------------------
def save_pruned_weights(pruned_model, weight_file):
    """
    保存剪枝后的模型权重到指定文件。
    
    参数:
    - pruned_model: 剪枝后的模型
    - weight_file: 权重保存的文件路径
    """
    # 保存剪枝后的模型权重
    torch.save(pruned_model.state_dict(), weight_file)
    print(f"剪枝后权重已储存 {weight_file}")

save_pruned_weights(pruned_model, 'pruned_model_weights_saved.pth')

# 剪枝后微调
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.nn import functional as F
from PIL import Image
import numpy as np

# 自定义转换函数，用于同时处理图像和标签
class SegmentationTransforms:
    def __init__(self, size=(256, 256)):
        # 使用torchvision的Resize进行图像和标签的调整大小
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, image, target):
        # 调整图像和标签的大小
        image = self.resize(image)
        image = self.to_tensor(image)  # 将图像转换为张量
        
        target = self.resize(target)  # 调整标签的大小
        target = torch.as_tensor(np.array(target), dtype=torch.long)  # 转换为张量并确保为整型
        return image, target

# 微调函数
def train_finetune_pruned_model(
    model,
    weight_path,
    data_path,
    num_epochs=100,
    batch_size=16,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    model.train()

    transform = SegmentationTransforms()
    train_dataset = VOCSegmentation(
        root=data_path,
        year='2012',
        image_set='train',
        download=False,
        transforms=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(images)
            # 如果输出是直接的张量
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    torch.save(model.state_dict(), '/data1/Segcam_code/segcam/tutorials/finetuned_pruned_model.pth')
    print("微调完成，模型已保存为 'finetuned_pruned_model.pth'。")



train_finetune_pruned_model(
    model=pruned_model,  # 剪枝后的模型
    weight_path='/data1/Segcam_code/segcam/tutorials/pruned_model_weights_saved.pth',  # 权重路径
    data_path='/data1/Deeplabv3/data'  # 数据集路径
)




# # print('1')


# 剪枝后推理
# ------------------------------------------------------------------
import os
import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torchvision import models

def time_synchronized():
    """同步时间函数，确保多GPU使用时能同步"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def deeplabv3_resnet50(aux=False, num_classes=21):
    """创建DeepLabV3 ResNet50模型"""
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    if aux:
        model.aux_classifier = models.segmentation.deeplabv3.ResNet50_Secondary(num_classes)
    return model

def process_image(img_path, model, device, palette, output_dir):
    """处理单张图像并保存车类预测结果"""
    # 加载图像
    original_img = Image.open(img_path)

    # 从PIL图像转换为Tensor并进行归一化
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    
    # 扩展batch维度
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # 初始化图像尺寸
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)  # 初始化模型
        
        t_start = time_synchronized()
        output = model(img.to(device))  # 执行推理
        t_end = time_synchronized()
        print(f"推理时间: {t_end - t_start}")

        # 确保'out'是一个有效的tensor
        if 'out' in output:
            # 提取预测结果，并仅保留车类（类别7）
            prediction = output['out'].argmax(1).squeeze(0)  # 获取预测结果
            prediction = prediction.cpu().numpy().astype(np.uint8)

            # 创建一个新的掩码图像，仅保留车类（类别7），其他类别设置为0（背景）
            # car_mask = np.where(prediction == 7, 255, 0).astype(np.uint8)  # 类别7为车，其他为背景
            car_mask_img = Image.fromarray(prediction, mode='P')  # 指定模式为 'P'
            car_mask_img.putpalette(palette)  # 设置调色板

            # 保存结果图像，文件名基于原始图像文件名并转换为 .png 格式
            base_name = os.path.splitext(os.path.basename(img_path))[0]  # 去掉扩展名
            result_path = os.path.join(output_dir, f"result_{base_name}.png")  # 强制使用 .png
            car_mask_img.save(result_path)  # 保存车的预测结果
            # print(f"结果已保存到 {result_path}")
        else:
            print(f"输出不包含'out'键: {img_path}")

def main():
    aux = False  # 推理时不需要辅助分类器
    classes = 20  # 类别数量（不包括背景）
    
    weights_path = "/data1/Segcam_code/segcam/tutorials/finetuned_pruned_model.pth"
    img_dir = "/data1/Segcam_code/segcam/tutorials/test/"  # 输入图像的文件夹
    output_dir = "/data1/Segcam_code/segcam/tutorials/results/"  # 结果图像保存的文件夹
    palette_path = "/data1/Segcam_code/segcam/tutorials/palette.json"
    
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_dir), f"image directory {img_dir} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."

    # 加载调色板
    with open(palette_path, "rb") as f:
        palette_dict = json.load(f)
        palette = []
        for v in palette_dict.values():
            palette += v

    # 创建并初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet50(aux=aux, num_classes=classes + 1)
    
    # 加载模型权重
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)

    # 创建结果保存目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取图像文件列表
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    # 遍历每张图像，进行处理
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        process_image(img_path, model, device, palette, output_dir)

if __name__ == '__main__':
    main()
