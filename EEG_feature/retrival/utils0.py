from sre_constants import CATEGORY
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
import json
import torch
import re
import open_clip
from torch.utils.data import DataLoader
from natsort import natsorted
import random

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # 加载模型、训练时的预处理和特征提取器
# vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
#     model_name = 'ViT-H-14', pretrained = None, precision='fp32', device=device
# )

# # 加载你已经下载的权重文件
# model_weights_path = "/mnt/repo0/kyw/open_clip_pytorch_model.bin"
# model_state_dict = torch.load(model_weights_path, map_location=device)
# vlmodel.load_state_dict(model_state_dict)

# # 将模型设置为评估模式
# vlmodel.eval()

#生成脑电信号模块
# 定义您的模型结构
def create_model(device):
    model = models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 17*250)  # 17通道 × 100时间点 = 1700 输出
    model = model.to(device)
    return model

# 加载模型权重
def load_model_endocer(model_path, device):
    model = create_model(device)  # 首先创建模型
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['best_model'])  # 加载模型参数
    model.eval()
    return model

# 图像预处理函数
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 模型接受224x224的图像输入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # 加入 batch 维度
    return image_tensor

# 生成 EEG 信号函数
def generate_eeg(model, image_tensor, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        # 输入图像到模型，生成EEG信号
        eeg_output = model(image_tensor).detach().cpu().numpy()
        
        # 假设模型输出的是 (1, 1700) 的向量，将其重塑为 (17, 100)
        eeg_output = np.reshape(eeg_output, (17, 250))
    
    return eeg_output

# 保存 EEG 信号函数
def save_eeg_signal(eeg_signal, save_dir, idx, category):
    # 确保保存路径存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_name = f"{category}_{idx + 1}.npy"  # 文件名带有排序特征
    file_path = os.path.join(save_dir, file_name)
        
    # 保存 EEG 信号到 .npy 文件
    np.save(file_path, eeg_signal)
    # print(f"EEG 信号已保存至: {file_path}")
    
    
def extract_number(filename):
    """
    从文件名中提取数字。如果没有数字，则返回0。
    
    """
    
    numbers = re.findall(r'(\d+)', filename)
    if numbers:
        return tuple(map(int, numbers))  # 返回多个数字的元组
    return (float('inf'),)  # 如果没有数字，返回一个包含非常大值的元组

    # match = re.search(r'(\d+)', filename)
    # return int(match.group(1)) if match else 0

# 遍历图像文件夹，处理每个图像
def generate_and_save_eeg_for_all_images(model_path, test_image_list, save_dir, device, category_list):
    model = load_model_endocer(model_path, device)
    for idx, image_path in enumerate(test_image_list):
        image_tensor = preprocess_image(image_path, device)
        synthetic_eeg = generate_eeg(model, image_tensor, device)
        category = category_list[idx]
        save_eeg_signal(synthetic_eeg, save_dir, idx, category)

    # # 加载已经保存的模型
    # # model_path = '/mnt/repo0/kyw/close-loop/sub_model/sub-08/generation/encoding-end_to_end/dnn-alexnet/modeled_time_points-all/pretrained-False/model_state_dict_250hz.pt'
    # model = load_model_endocer(model_path, device)
    #  # EEG 信号保存路径

    # # 遍历图像文件夹中的每一张图像
    # # image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    # # image_files = natsorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and not f.startswith('._')])
    # # image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and not f.startswith('._')]
    # # image_files.sort(key=extract_number)
    # image_files = [img for img in test_image_list]
    # image_files.sort(key=lambda x: extract_number(x))
    # # image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png')) and not f.startswith('._')])


    # for idx, image_file in enumerate(image_files):
    #     image_path = os.path.join(image_folder, image_file)
    #     # print(f"正在处理图像: {image_path}")

    #     # 预处理图像
    #     image_tensor = preprocess_image(image_path, device)

    #     # 生成 EEG 信号
    #     synthetic_eeg = generate_eeg(model, image_tensor, device)

    #     # 保存 EEG 信号
    #     save_eeg_signal(synthetic_eeg, save_dir, idx, image_gen)
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算额外的一个元素以适应奇数维度
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        # 使用切片确保不会溢出
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 5), (1, 1)),
            nn.AvgPool2d((1, 17), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (17, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1840, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATM_S_reconstruction_scale_0_1000(nn.Module):    
    def __init__(self, num_channels=17, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_reconstruction_scale_0_1000, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  
         
    def forward(self, x):
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        # print(f'After enc_eeg shape: {eeg_embedding.shape}')
        out = self.proj_eeg(eeg_embedding)
        return out  
    
def load_model_decoder(model_path, device):
    """
    加载预训练的 EEG 和图像模型以及优化器的状态
    """
    checkpoint = torch.load(model_path, map_location=device)
    eeg_model = ATM_S_reconstruction_scale_0_1000(17, 250)  # 例如 ATM_S_reconstruction_scale_0_1000 是 EEG 模型
    img_model = Proj_img()  # 假设使用的是 Proj_img 模型

    eeg_model.load_state_dict(checkpoint['eeg_model_state_dict'])
    img_model.load_state_dict(checkpoint['img_model_state_dict'])
    optimizer_state = checkpoint['optimizer_state_dict']
    
    return eeg_model.to(device), img_model.to(device), optimizer_state

def ImageEncoder(images, img_model, preprocess_train, device):
    batch_size = 20  # 设置为合适的值
    image_features_list = []
      
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

        with torch.no_grad():
            batch_image_features = vlmodel.encode_image(image_inputs)
            # batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

        image_features_list.append(batch_image_features)

    image_features = torch.cat(image_features_list, dim=0)
        
    return image_features

def load_all_eeg_signals(eeg_folder):
    """
    从文件夹加载 EEG 信号
    """
    eeg_paths = []
    for filename in os.listdir(eeg_folder):
        if filename.endswith('.npy'):
            eeg_paths.append(os.path.join(eeg_folder, filename))
    eeg_signals = []

    for path in eeg_paths:
        # 检查文件是否存在
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件未找到: {path}")

        # 加载单个脑电信号
        eeg_tensor = load_eeg_signals(path)  # 形状为 [17, 100]

        eeg_signals.append(eeg_tensor)

    # 堆叠所有脑电信号，形成形状为 [N, 17, 100] 的张量
    combined_eeg = torch.stack(eeg_signals, dim=0)  # 形状为 [81, 17, 100]
    
    return  combined_eeg

def load_eeg_signals(eeg_path):

    eeg_data = np.load(eeg_path)
    eeg_signals = torch.tensor(eeg_data)
    return eeg_signals

def get_eeg_features(eeg_model, eeg_signal, device):
    eeg_model.eval()
    eeg_model.to(device)
    eeg_signal = eeg_signal.to(device)
    eeg_embeds = eeg_model(eeg_signal.unsqueeze(0)).float()  # 将EEG信号传入模型
    return eeg_embeds

def get_img_features(img_model, preprocess_train,vlmodel,device,img_path):
    img_model.to(device)
    img_model.eval()
    image_input = torch.stack([preprocess_train(Image.open(img_path).convert("RGB"))]).to(device)
    img_embeds = vlmodel.encode_image(image_input)
    return img_embeds
    
def evaluate_eeg_signals(eeg_model, img_model, eeg_signals_truth, device, truth_folder, false_folder, truth, false):
    """
    对给定的EEG信号进行分类，并计算准确率
    """
    correct = 0
    total = 0
    correct_samples = []

    # 加载 truth 和 false 文件夹中的图像并生成特征
    img_truth_paths = [os.path.join(truth_folder, f"{truth}_{i+1}.jpg") for i in range(len(eeg_signals_truth))]
    img_truth = ImageEncoder(img_truth_paths, img_model, preprocess_train, device)

    img_false_paths = [os.path.join(false_folder, f"{false}_{i+1}.jpg") for i in range(1)]
    img_false = ImageEncoder(img_false_paths, img_model, preprocess_train, device)

    with torch.no_grad():
        # 遍历 truth 和 false 文件夹中的 EEG 信号
        for idx, eeg_data in enumerate(eeg_signals_truth):
            eeg_data = eeg_data.to(device)
            eeg_features = eeg_model(eeg_data.unsqueeze(0)).float()  # 将EEG信号传入模型
            logit_scale = eeg_model.logit_scale

            # 提取 truth 图片特征
            img_features_truth = img_truth[idx].unsqueeze(0).float()
            img_features_false = img_false[idx % len(img_false)].unsqueeze(0).float()  # 循环使用 false 特征

            # 计算与 truth 和 false 图片的 logits
            logits_truth = logit_scale * (eeg_features @ img_features_truth.T)
            logits_false = logit_scale * (eeg_features @ img_features_false.T)

            # 将 truth 和 false 的 logits 合并，判断是否正确分类
            logits = torch.cat([logits_truth, logits_false], dim=1)
            predicted_label = torch.argmax(logits)

            true_label = 0  # truth 的标签为 0

            if predicted_label == true_label:
                correct += 1
                correct_samples.append(idx)

            total += 1

        # 遍历 false 文件夹中的 EEG 信号
        # for idx, eeg_data in enumerate(eeg_signals_false):
        #     eeg_data = eeg_data.to(device)
        #     eeg_features = eeg_model(eeg_data.unsqueeze(0)).float()  # 将EEG信号传入模型
        #     logit_scale = eeg_model.logit_scale

        #     # 提取 truth 和 false 图片特征
        #     img_features_truth = img_truth[idx % len(img_truth)].unsqueeze(0).float()  # 循环使用 truth 特征
        #     img_features_false = img_false[idx].unsqueeze(0).float()

        #     # 计算与 truth 和 false 图片的 logits
        #     logits_truth = logit_scale * (eeg_features @ img_features_truth.T)
        #     logits_false = logit_scale * (eeg_features @ img_features_false.T)

        #     # 将 truth 和 false 的 logits 合并，判断是否正确分类
        #     logits = torch.cat([logits_truth, logits_false], dim=1)
        #     predicted_label = torch.argmax(logits)

        #     true_label = 1  # false 的标签为 1

        #     if predicted_label == true_label:
        #         correct += 1
        #         correct_samples.append(idx + len(eeg_signals_truth))  # 索引需要加上 truth 的数量

        #     total += 1

    accuracy = correct / total
    return accuracy, correct_samples




def classification(gene_image_embed, gene_eeg_embed, num_class, test_image_embeds, idx):
    num_test_samples = test_image_embeds.shape[0]
    all_idxes = list(range(num_test_samples))
    rest_idxes = [i for i in all_idxes if i != idx]

    select_idxes = random.sample(rest_idxes, num_class - 1)

    similarities = {}
    for select_idx in select_idxes:
        similarity = F.cosine_similarity(gene_eeg_embed, test_image_embeds[select_idx])
        similarities[select_idx] = similarity

    gene_sim = F.cosine_similarity(gene_image_embed, gene_eeg_embed)
    similarities[idx] = gene_sim

    max_idx = max(similarities, key=similarities.get)

    if max_idx == idx:
        return 1
    else:
        return 0

def get_image_pool(image_set_path):
    test_images_path = []
    labels = []
    for sub_test_image in sorted(os.listdir(image_set_path)):
        if sub_test_image.startswith('.'):
            continue
        sub_image_path = os.path.join(image_set_path, sub_test_image)
        for image in sorted(os.listdir(sub_image_path)):
            if image.startswith('.'):
                continue
            image_label = os.path.splitext(image)[0]
            labels.append(image_label)
            image_path = os.path.join(sub_image_path, image)
            test_images_path.append(image_path)
    return test_images_path, labels 

def sample_from_image_pool(image_paths, labels, k):
    idxes = random.sample(range(len(image_paths)), k)
    return [image_paths[idx] for idx in idxes], [labels[idx] for idx in idxes]