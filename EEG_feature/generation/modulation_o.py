import torch
import os
import numpy as np
from scipy.signal import spectrogram
from utils import get_image_pool, sample_from_image_pool, generate_and_save_eeg_for_all_images
from mne.time_frequency import psd_array_multitaper
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import open_clip
from utils import Proj_img
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

device = "cuda:2" if torch.cuda.is_available() else "cpu"
fs = 250
selected_channel_idxes = [3, 4, 5] # 'O1', 'Oz', 'O2'
# 17channels ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']

image_set_path = '/mnt/repo0/kyw/images_set/test_images'
test_images_path, labels = get_image_pool(image_set_path)
sample_images, sample_labels = sample_from_image_pool(test_images_path, labels, 10)
save_path = '/mnt/repo0/kyw/close-loop/modulation_o'

# target_path = '/mnt/repo0/kyw/close-loop/frequency_loop/sub-08/panda_1.npy'
# target_signal = np.load(target_path, allow_pickle=True)
# selected_target_signal = target_signal[selected_channel_idxes, :]

# target_psd, target_freqs = psd_array_multitaper(selected_target_signal, fs, adaptive=True, normalization='full', verbose=0) # psd(3, 126)
# target_psd = torch.from_numpy(target_psd.flatten())
# target_psd = target_psd.unsqueeze(0)

# sub-08
# model_path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet/sub-08/model_state_dict.pt'
# generate_and_save_eeg_for_all_images(model_path, sample_images, save_path, device, sample_labels)

# similarities = []
# for eeg in sorted(os.listdir(save_path)):
#     eeg_path = os.path.join(save_path, eeg)
#     eeg = np.load(eeg_path, allow_pickle=True)
#     selected_eeg = eeg[selected_channel_idxes, :]
#     psd, freqs = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
#     psd = torch.from_numpy(psd.flatten())
#     # psd = torch.from_numpy(psd)
#     cs = F.cosine_similarity(psd, target_psd)
#     cs_mean = cs.mean().item()
#     similarities.append(cs_mean)
# print(similarities)

# best = '/mnt/repo0/kyw/close-loop/modulation_o/omelet_10s_6.npy'
# signals = np.load(best, allow_pickle=True)
# average_signals = np.mean(signals, axis=0)
# frequencies, times, Sxx = spectrogram(average_signals, fs, nperseg=50)
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# plt.title('Time-Frequency Representation of EEG Signal (Averaged Over Channels)')
# plt.colorbar(label='Intensity (dB)')
# plt.ylim(0, fs / 2)  # 只显示到 Nyquist 频率
# plt.savefig('/mnt/repo0/kyw/close-loop/omelet_10s_6.jpg')
# plt.show()

import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
# sub-08
gene_eeg = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-08'
psds = []
for eeg in sorted(os.listdir(gene_eeg)):
    eeg_path = os.path.join(gene_eeg, eeg)
    eeg_signal = np.load(eeg_path, allow_pickle=True)
    selected_eeg_signal = eeg_signal[selected_channel_idxes, :]
    psd, freqs = psd_array_multitaper(selected_eeg_signal, fs, adaptive=True, normalization='full', verbose=0)
    psds.append(psd.flatten())
psds = np.array(psds)
correlation_matrix = np.corrcoef(psds)
distance_matrix = 1 - correlation_matrix
distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix, 0)
linkage_matrix = linkage(squareform(distance_matrix), method='ward')

num_clusters = 5
cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

order = np.argsort(cluster_labels)
sorted_corr_matrix = correlation_matrix[order, :][:, order]

plt.figure(figsize=(10, 8))
sns.heatmap(sorted_corr_matrix, cmap='Blues', cbar_kws={"label": "Correlation"}, square=True)

plt.title('Clustered Correlation Matrix')
plt.xlabel('PSD feature')
plt.ylabel('PSD feature')
plt.savefig('/mnt/repo0/kyw/close-loop/clustered.jpg')
plt.show()

num_elements = sorted_corr_matrix.shape[0]
unique_labels = np.unique(cluster_labels)

avg_diff = np.zeros(num_elements)

for i in range(num_elements):
    own_cluster = cluster_labels[i]
    other_clusters_mask = cluster_labels != own_cluster
    other_cluster_corr = sorted_corr_matrix[i, other_clusters_mask]
    
    avg_diff[i] = np.mean(other_cluster_corr)

max_diff_index = np.argmax(avg_diff)
print(f"与其他类别差距最大的元素索引是: {max_diff_index}")
print(f"该元素的平均差距值是: {avg_diff[max_diff_index]}")

num_elements = 200

avg_corr = np.zeros(num_elements)

for i in range(num_elements):
    other_elements_corr = np.delete(correlation_matrix[i], i)
    avg_corr[i] = np.mean(other_elements_corr)
min_corr_index = np.argmin(avg_corr)

print(f"与其他元素平均相关性最低的元素索引是: {min_corr_index}")
print(f"该元素的相关性均值是: {avg_corr[min_corr_index]}")

# min_corr_value = np.min(correlation_matrix)  # 相关性最低的值
# min_corr_indices = np.unravel_index(np.argmin(correlation_matrix), correlation_matrix.shape)  # 获取对应的索引
# # 打印相关性最低的信号对及其相关性值
# signal1_idx, signal2_idx = min_corr_indices
# print(f"相关性最低的信号对: 信号 {signal1_idx + 1} 和 信号 {signal2_idx + 1}")
# print(f"最低的相关性值: {min_corr_value}")

# psd_signal1 = psds[signal1_idx]
# psd_signal2 = psds[signal2_idx]
# print(psd_signal1, psd_signal2)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Standardize the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(psds)

# # PCA for dimensionality reduction
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(scaled_data)

# # KMeans clustering
# kmeans = KMeans(n_clusters=5, random_state=42)
# kmeans.fit(reduced_data)
# labels = kmeans.labels_

# # Calculate distances to cluster centers
# distances = np.linalg.norm(reduced_data - kmeans.cluster_centers_[labels], axis=1)

# # Set threshold for outliers (e.g., 95th percentile)
# threshold = np.percentile(distances, 95)

# # Identify outliers
# outliers = np.where(distances > threshold)[0]

# # Visualization
# plt.figure(figsize=(10, 8))
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)
# plt.scatter(reduced_data[outliers, 0], reduced_data[outliers, 1], color='red', label='Outliers', s=150, edgecolor='k')
# plt.title('PCA and KMeans Clustering with Outliers')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar(label='Cluster Label')
# plt.legend()
# plt.grid(True)
# plt.savefig('/mnt/repo0/kyw/close-loop/kmeans_with_outliers.jpg')
# plt.show()

# print("Outlier indices:", outliers)
# # Outlier indices: [ 33  72  86 109 115 132 162 172 183 188]

# max_outlier_index = outliers[np.argmax(distances[outliers])]

# # 可视化，标记最离群的点
# plt.figure(figsize=(10, 8))
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)
# plt.scatter(reduced_data[outliers, 0], reduced_data[outliers, 1], color='red', label='Outliers', s=150, edgecolor='k')
# plt.scatter(reduced_data[max_outlier_index, 0], reduced_data[max_outlier_index, 1], color='blue', label='Most Outlier', s=200, edgecolor='k')
# plt.title('PCA and KMeans Clustering with Outliers')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar(label='Cluster Label')
# plt.legend()
# plt.grid(True)
# plt.savefig('/mnt/repo0/kyw/close-loop/kmeans_with_most_outlier.jpg')
# plt.show()

# print("最离群的点索引:", max_outlier_index)



# device = "cuda" if torch.cuda.is_available() else "cpu"
# fs = 250
# selected_channel_idxes = [3, 4, 5] # 'O1', 'Oz', 'O2'

# target_path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-08/00163_seed_163.npy'
# target_signal = np.load(target_path, allow_pickle=True)
# selected_target_signal = target_signal[selected_channel_idxes, :]

# target_psd, target_freqs = psd_array_multitaper(selected_target_signal, fs, adaptive=True, normalization='full', verbose=0) # psd(3, 126)
# target_psd = torch.from_numpy(target_psd.flatten())
# target_psd = target_psd.unsqueeze(0)

# path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-08/00110_lightning_bug_110.npy'
# path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-08/00182_tiara_182.npy'
# eeg = np.load(path, allow_pickle=True)

# selected_signal = eeg[selected_channel_idxes, :]

# psd, freqs = psd_array_multitaper(selected_signal, fs, adaptive=True, normalization='full', verbose=0) # psd(3, 126)
# psd = torch.from_numpy(psd.flatten())
# psd = psd.unsqueeze(0)



# cs = F.cosine_similarity(target_psd, psd)
# print(cs) 

# tensor([0.8716], dtype=torch.float64)
# sub_8_path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-08'
# CS = []
# for eeg in sorted(os.listdir(sub_8_path)):
#     eeg_path = os.path.join(sub_8_path, eeg)
#     eeg = np.load(eeg_path, allow_pickle=True)
#     selected_eeg = eeg[selected_channel_idxes, :]
#     psd, freqs = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0) # psd(3, 126)
#     psd = torch.from_numpy(psd.flatten())
#     psd = psd.unsqueeze(0)
#     cs = F.cosine_similarity(psd, target_psd)
#     CS.append(cs)
# CS = torch.tensor(CS)
# top_result = torch.topk(CS, 5)
# print(top_result)

# best = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-08/00114_metal_detector_114.npy'
# best_signal = np.load(best, allow_pickle=True)
# average_signals = np.mean(best_signal, axis=0)
# frequencies, times, Sxx = spectrogram(average_signals, 250, nperseg=50)
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# plt.title('Time-Frequency Representation of EEG Signal (Averaged Over Channels)')
# plt.colorbar(label='Intensity (dB)')
# plt.ylim(0, 250 / 2)  # 只显示到 Nyquist 频率
# plt.savefig('/mnt/repo0/kyw/close-loop/best_signal.jpg')
# plt.show()

# target_path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-08/00163_seed_163.npy'
# target_signal = np.load(target_path, allow_pickle=True)
# average_signals = np.mean(target_signal, axis=0)
# frequencies, times, Sxx = spectrogram(average_signals, 250, nperseg=50)
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (s)')
# plt.title('Time-Frequency Representation of EEG Signal (Averaged Over Channels)')
# plt.colorbar(label='Intensity (dB)')
# plt.ylim(0, 250 / 2)  # 只显示到 Nyquist 频率
# plt.savefig('/mnt/repo0/kyw/close-loop/target_signal.jpg')
# plt.show()


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

# diffusion_model_path = "/mnt/repo0/kyw/close-loop/sub_model/sub-08/diffusion_250hz/ATM_S_reconstruction_scale_0_1000_40.pth"
# checkpoint = torch.load(diffusion_model_path, map_location=device)
# img_model = Proj_img() 
# img_model.load_state_dict(checkpoint['img_model_state_dict'])

# from PIL import Image
# from custom_pipeline_tjh import *
# from diffusion_prior_tjh import *

# generator = Generator4Embeds(num_inference_steps=4, device=device, guidance_scale=2.0)

# #从groudtruth得到n张图片和embed
# def gt_image_to_images(image_gt_path, num_images, device):
#     img_model.eval()
#     gt_image_input = torch.stack([preprocess_train(Image.open(image_gt_path).convert("RGB"))]).to(device)
#     vlmodel.to(device)  # 将模型加载到正确的设备上
#     img_embeds = vlmodel.encode_image(gt_image_input)
#     save_img_path = f'/mnt/repo0/kyw/close-loop/gene_images'
#     os.makedirs(save_img_path, exist_ok=True)
    
#     batch_size = 2  # 每批次处理的图片数量
#     for batch_start in range(0, num_images, batch_size):
#         batch_images = []
#         for idx in range(batch_start, min(batch_start + batch_size, num_images)):
#             # 生成图片并释放内存
#             with torch.no_grad():  # 禁用梯度计算以节省显存
#                 image = generator.generate(img_embeds, guidance_scale=2.0)
#             save_imgs_path = os.path.join(save_img_path, f'image_{idx}.jpg')  # 构造保存路径
#             image.save(save_imgs_path)
#             print(f"图片保存至: {save_imgs_path}")

#         # 清理临时变量，释放显存
#         del batch_images
#         torch.cuda.empty_cache()
    
#     # return image_embeddings

# gt_image_to_images('/mnt/repo0/kyw/images_set/test_images/00114_metal_detector/metal_detector_02s.jpg', 10, device)