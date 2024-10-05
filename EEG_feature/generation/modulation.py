import numpy as np
import torch
import os
import random
from PIL import Image
import torch.nn.functional as F
from scipy.special import softmax
from mne.time_frequency import psd_array_multitaper
from utils import generate_and_save_eeg_for_all_images
import matplotlib.pyplot as plt
from scipy.special import softmax
import open_clip
from custom_pipeline_tjh import *
from diffusion_prior_tjh import *
# from utils import Proj_img

model_weights_path = "/mnt/repo0/kyw/open_clip_pytorch_model.bin"
# diffusion_model_path = "/mnt/repo0/kyw/close-loop/sub_model/sub-01/diffusion_250hz/ATM_S_reconstruction_scale_0_1000_40.pth"

vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
    model_name='ViT-H-14', pretrained=None, precision='fp32', device=device
)
vlmodel.load_state_dict(torch.load(model_weights_path, map_location=device))
vlmodel.eval()

# checkpoint = torch.load(diffusion_model_path, map_location=device)
# img_model = Proj_img()
# img_model.load_state_dict(checkpoint['img_model_state_dict'])
# img_model.eval()

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

def calculate_similarity(eeg_path, target_psd, fs, selected_channel_idxes):
    eeg = np.load(eeg_path, allow_pickle=True)
    selected_eeg = eeg[selected_channel_idxes, :]
    psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
    psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
    return F.cosine_similarity(target_psd, psd).item()

def calculate_loss(eeg_path, target_psd, fs, selected_channel_idxes):
    eeg = np.load(eeg_path, allow_pickle=True)
    selected_eeg = eeg[selected_channel_idxes, :]
    psd, _ = psd_array_multitaper(selected_eeg, fs, adaptive=True, normalization='full', verbose=0)
    psd = torch.from_numpy(psd.flatten()).unsqueeze(0)
    target_psd = torch.tensor(target_psd).view(1, 378)
    loss_fn = nn.MSELoss()
    loss = loss_fn(psd, target_psd)
    return loss

def select(probabilities, similarities, losses, sample_image_paths, sample_eeg_paths):
    chosen_indices = np.random.choice(len(probabilities), size=2, replace=False, p=probabilities)
    chosen_similarities = [similarities[idx] for idx in chosen_indices.tolist()] 
    chosen_losses = [losses[idx] for idx in chosen_indices.tolist()]
    chosen_image_paths = [sample_image_paths[idx] for idx in chosen_indices.tolist()]
    chosen_eeg_paths = [sample_eeg_paths[idx] for idx in chosen_indices.tolist()]
    return chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths

def get_prob_random_sample(test_images_path, model_path, save_path, fs, device, selected_channel_idxes, processed_paths, target_psd):
    available_paths = [path for path in test_images_path if path not in processed_paths]
    sample_image_paths = sorted(random.sample(available_paths, 10))
    processed_paths.update(sample_image_paths)
    sample_image_name = []
    for sample_image_path in sample_image_paths:
        filename = os.path.basename(sample_image_path).split('.')[0]
        sample_image_name.append(filename)
    generate_and_save_eeg_for_all_images(model_path, sample_image_paths, save_path, device, sample_image_name)
    similarities = []
    sample_eeg_paths = []
    losses = []
    for eeg in sorted(os.listdir(save_path)):
        filename = eeg.split('.')[0]
        eeg_path = os.path.join(save_path, eeg)
        sample_eeg_paths.append(eeg_path)
        cs = calculate_similarity(eeg_path, target_psd, fs, selected_channel_idxes)
        similarities.append(cs)
        loss = calculate_loss(eeg_path, target_psd, fs, selected_channel_idxes)
        losses.append(loss)
    probabilities = softmax(similarities)
    chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths = select(probabilities, similarities, losses, sample_image_paths, sample_eeg_paths)
    return chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths

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

device = "cuda:4" if torch.cuda.is_available() else "cpu"
fs = 250
selected_channel_idxes = [3, 4, 5]  # 'O1', 'Oz', 'O2'

def load_target_psd(target_path, fs, selected_channel_idxes):
    target_signal = np.load(target_path, allow_pickle=True)
    selected_target_signal = target_signal[selected_channel_idxes, :]
    target_psd, _ = psd_array_multitaper(selected_target_signal, fs, adaptive=True, normalization='full', verbose=0)
    return torch.from_numpy(target_psd.flatten()).unsqueeze(0)

def main_experiment_loop(seed, save_path):
    print(seed)

    model_path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet/sub-01/model_state_dict.pt'
    # model_path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_pretrain/sub-01/model_state_dict.pt'
    # target_eeg_path = '/mnt/repo0/kyw/close-loop/target_pretrain_1.npy'
    target_eeg_path = '/mnt/repo0/kyw/close-loop/sub_encoder_alexnet_test/sub-01/00183_tick_183.npy'
    image_set_path = '/mnt/repo0/kyw/images_set/test_images'

    target_psd = load_target_psd(target_eeg_path, fs, selected_channel_idxes)

    test_images_path, _ = get_image_pool(image_set_path)
    target_image_path = '/mnt/repo0/kyw/images_set/test_images/00183_tick/tick_06s.jpg'
    test_images_path.remove(target_image_path)

    num_loops = 90
    processed_paths = set()
    
    all_chosen_similarities = []
    all_chosen_losses = []
    all_chosen_image_paths = []
    all_chosen_eeg_paths = []
    history_cs = []
    history_loss = []

    for i in range(num_loops):
        print(f"Loop {i + 1}/{num_loops}")
        round_save_path = os.path.join(save_path, f'loop{i + 1}')
        loop_sample_ten = []
        loop_cs_ten = []
        loop_eeg_ten = []
        loop_loss_ten = []
        os.makedirs(save_path, exist_ok=True)

        if i == 0:
            first_ten = os.path.join(round_save_path, 'first_ten')
            os.makedirs(first_ten, exist_ok=True)
            chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths = get_prob_random_sample(test_images_path, 
                                                                                                              model_path, 
                                                                                                              first_ten, 
                                                                                                              fs, device, 
                                                                                                              selected_channel_idxes,
                                                                                                              processed_paths, 
                                                                                                              target_psd)
        else:
            chosen_similarities = all_chosen_similarities[-2:]
            chosen_image_paths = all_chosen_image_paths[-2: ]
        for chosen_image_path in chosen_image_paths:
            loop_sample_ten.append(chosen_image_path)
        for chosen_similarity in chosen_similarities:
            loop_cs_ten.append(chosen_similarity)
        for chosen_eeg_path in chosen_eeg_paths:
            loop_eeg_ten.append(chosen_eeg_path)
        for chosen_loss in chosen_losses:
            loop_loss_ten.append(chosen_loss)
        print(chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths)
        fusion_image_to_images(chosen_image_paths, 6, device, round_save_path, 256)
        image_path_list = []
        label_list = []
        for image in sorted(os.listdir(round_save_path)):
            if image.endswith('jpg'):
                image_path = os.path.join(round_save_path, image)
                loop_sample_ten.append(image_path)
                image_path_list.append(image_path)
                file_name = os.path.splitext(image)[0]
                label_list.append(file_name)
        generate_and_save_eeg_for_all_images(model_path, image_path_list, round_save_path, device, label_list)
        for eeg in sorted(os.listdir(round_save_path)):
            if eeg.endswith('npy'):
                eeg_path = os.path.join(round_save_path, eeg)
                loop_eeg_ten.append(eeg_path)
                cs = calculate_similarity(eeg_path, target_psd, fs, selected_channel_idxes)
                loss = calculate_loss(eeg_path, target_psd, fs, selected_channel_idxes)
                loop_cs_ten.append(cs)
                loop_loss_ten.append(loss)
        available_paths = [path for path in test_images_path if path not in processed_paths]
        print(len(available_paths))
        sample_image_paths = sorted(random.sample(available_paths, min(2, len(available_paths))))
        new_sample_list = []
        new_sample_label = []
        for sample_image_path in sample_image_paths:
            filename = os.path.basename(sample_image_path).split('.')[0]
            new_sample_label.append(filename)
            loop_sample_ten.append(sample_image_path)
            new_sample_list.append(sample_image_path)
        print(loop_sample_ten)
        processed_paths.update(sample_image_paths)
        print(len(processed_paths))
        new_save_path = os.path.join(round_save_path, 'new_sample_signal')
        os.makedirs(new_save_path, exist_ok=True)
        generate_and_save_eeg_for_all_images(model_path, new_sample_list, new_save_path, device, new_sample_label)
        for new_sample_eeg in sorted(os.listdir(new_save_path)):
            new_sample_eeg_path = os.path.join(new_save_path, new_sample_eeg)
            loop_eeg_ten.append(new_sample_eeg_path)
            cs = calculate_similarity(new_sample_eeg_path, target_psd, fs, selected_channel_idxes)
            loss = calculate_loss(new_sample_eeg_path, target_psd, fs, selected_channel_idxes)
            loop_cs_ten.append(cs)
            loop_loss_ten.append(loss)
        print(loop_cs_ten)
        print(loop_loss_ten)
        probabilities = softmax(loop_cs_ten)
        print(probabilities)

        chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths = select(probabilities, loop_cs_ten, loop_loss_ten, loop_sample_ten,loop_eeg_ten)
        
        print("Chosen similarities:", chosen_similarities)
        print("Chosen losses:", chosen_losses)
        print("Chosen image paths:", chosen_image_paths)
        print("Chosen eeg paths:", chosen_eeg_paths)

        for chosen_similarity in chosen_similarities:
            all_chosen_similarities.append(chosen_similarity)
        for chosen_loss in chosen_losses:
            all_chosen_losses.append(chosen_loss)
        for chosen_image_path in chosen_image_paths:
            all_chosen_image_paths.append(chosen_image_path)
        for chosen_eeg_path in chosen_eeg_paths:
            all_chosen_eeg_paths.append(chosen_eeg_path)      

        max_similarity = max(chosen_similarities)
        print('max_similarity:', max_similarity)
        max_index = chosen_similarities.index(max_similarity)
        corresponding_loss = chosen_losses[max_index]
        print("Corresponding loss:", corresponding_loss)

        if len(history_cs) == 0:
            history_cs.append(max_similarity)
            history_loss.append(corresponding_loss) 
        else:
            max_history = max(history_cs)
            if max_similarity > max_history:
                history_cs.append(max_similarity)
                history_loss.append(corresponding_loss)
            else:
                history_cs.append(max_history)
                history_loss.append(history_loss[-1])

        print(history_cs)
        print(history_loss)

        if len(history_cs) >= 2:
            if history_cs[-1] != history_cs[-2]:
                diff = abs(history_cs[-1] - history_cs[-2])
                print(history_cs[-1], history_cs[-2], diff)
                if diff <= 1e-4:
                    print("The difference is within 10e-4, stopping.")
                    break

    print(all_chosen_similarities)
    print(all_chosen_losses)
    print(all_chosen_image_paths)
    print(all_chosen_eeg_paths)

    plt.figure(figsize=(10, 5))
    plt.plot(history_cs, marker='o', markersize=3, label='Similarity')
    # plt.plot(history_cs, marker='o', markersize=5, label='Similarity')
    # plt.plot(history_loss, marker='x', markersize=5, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend() 
    path = os.path.join(save_path, 'similarities.jpg')
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    num_run = 6
    for i in range(1, num_run + 1):
        base_save_path = f'/mnt/repo1/jiahua/exp_sub1/loop_random_{i}'
        os.makedirs(base_save_path, exist_ok=True)
        print(f'run{i}/{num_run}')
        seed = 10100101 + i
        np.random.seed(seed) 
        random.seed(seed)
        main_experiment_loop(seed, base_save_path)