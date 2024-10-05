import numpy as np
from utils import generate_and_save_eeg_for_all_images
import torch
from mne.time_frequency import psd_array_multitaper
import os
import torch.nn.functional as F
from scipy.signal import spectrogram
import random
import matplotlib.pyplot as plt
from scipy.special import softmax
from PIL import Image
import open_clip
from custom_pipeline_tjh import *
from diffusion_prior_tjh import *
from utils import Proj_img

model_weights_path = "/mnt/repo0/kyw/open_clip_pytorch_model.bin"
diffusion_model_path = "/mnt/repo0/kyw/close-loop/sub_model/sub-08/diffusion_250hz/ATM_S_reconstruction_scale_0_1000_40.pth"

vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
    model_name='ViT-H-14', pretrained=None, precision='fp32', device=device
)
vlmodel.load_state_dict(torch.load(model_weights_path, map_location=device))
vlmodel.eval()

checkpoint = torch.load(diffusion_model_path, map_location=device)
img_model = Proj_img()
img_model.load_state_dict(checkpoint['img_model_state_dict'])
img_model.eval()

generator = Generator4Embeds(num_inference_steps=4, device=device, guidance_scale=2.0)

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

def get_avg_signal(signal, name, save_path):
    average_signals = np.mean(signal, axis=0)
    plt.figure(figsize=(10, 3))
    plt.plot(average_signals)
    plt.title('Average Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig(save_path + f'{name}_avg_signal.jpg')
    plt.show()
    return average_signals

def get_time_freq(average_signals, fs, name, save_path):
    frequencies, times, Sxx = spectrogram(average_signals, fs, nperseg=50)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Time-Frequency')
    plt.colorbar(label='Intensity (dB)')
    plt.ylim(0, fs / 2)
    plt.savefig(save_path + f'{name}_time_freq.jpg')
    plt.show()

def get_eeg_pool(gene_eeg):
    eeg_paths = []
    for eeg in sorted(os.listdir(gene_eeg)):
        eeg_path = os.path.join(gene_eeg, eeg)
        eeg_paths.append(eeg_path)
    return eeg_paths

def calculate_similarity(eeg_path, target_psd, fs, selected_channel_idxes):
    eeg = np.load(eeg_path, allow_pickle=True)
    selected_eeg = eeg[selected_channel_idxes, :]
    psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
    psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
    return F.cosine_similarity(target_psd, psd).item()

def get_prob_random_sample(test_images_path, eeg_paths, fs, selected_channel_idxes, processed_paths, target_psd):
    available_paths = [path for path in test_images_path if path not in processed_paths]
    sample_image_paths = random.sample(available_paths, 10)
    processed_paths.update(sample_image_paths)
    idxes = [test_images_path.index(path) for path in sample_image_paths]
    sample_eeg_paths = [eeg_paths[idx] for idx in idxes]
    similarities = []
    for sample_eeg_path in sample_eeg_paths:
        sample_eeg = np.load(sample_eeg_path, allow_pickle=True)
        selected_eeg = sample_eeg[selected_channel_idxes, :]
        psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
        psd = torch.from_numpy(psd.flatten())
        psd = psd.unsqueeze(0)
        sim = F.cosine_similarity(target_psd, psd)
        similarities.append(sim.item())
    probabilities = softmax(similarities)
    chosen_indices = np.random.choice(len(probabilities), size=2, p=probabilities)
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_image_paths = [sample_image_paths[idx] for idx in chosen_indices.tolist()]
    chosen_eeg_paths = [sample_eeg_paths[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_image_paths, chosen_eeg_paths

def fusion_image_to_images(image_gt_paths, num_images, device, save_path, scale):
    img_embeds = []
    for image_gt_path in image_gt_paths:
        gt_image_input = torch.stack([preprocess_train(Image.open(image_gt_path).convert("RGB"))]).to(device)
        vlmodel.to(device)
        img_embed = vlmodel.encode_image(gt_image_input)
        img_embeds.append(img_embed)

    embed1, embed2 = img_embeds[0], img_embeds[1]
    embed_len = embed1.size(1)
    start_idx = random.randint(0, embed_len - scale - 1)
    end_idx = start_idx + scale
    temp = embed1[:, start_idx:end_idx].clone()
    embed1[:, start_idx:end_idx] = embed2[:, start_idx:end_idx]
    embed2[:, start_idx:end_idx] = temp

    save_img_path = save_path
    os.makedirs(save_img_path, exist_ok=True)
    batch_size = 2 
    for batch_start in range(0, num_images, batch_size):
        batch_images = []
        for idx in range(batch_start, min(batch_start + batch_size, num_images)):
            with torch.no_grad(): 
                image = generator.generate(embed1, guidance_scale=2.0)
            save_imgs_path = os.path.join(save_img_path, f'{scale}_{idx}.jpg') 
            image.save(save_imgs_path)
            print(f"图片保存至: {save_imgs_path}")
        del batch_images
        torch.cuda.empty_cache()