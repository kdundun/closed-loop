import numpy as np
import torch
import os
import sys
import random
from PIL import Image
from scipy.special import softmax
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from eeg_encoding.utils import generate_and_save_eeg_for_all_images
import matplotlib.pyplot as plt
from custom_pipeline import *
from modulation_utils import get_image_pool, calculate_loss, calculate_similarity, select, load_target_psd, get_prob_random_sample, load_vlmodel, get_target_eeg

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_weights_path = '/mnt/dataset0/jiahua/open_clip_pytorch_model.bin'
fs = 250
selected_channel_idxes = [3, 4, 5]  # 'O1', 'Oz', 'O2'

vlmodel, preprocess_train, feature_extractor = load_vlmodel(model_weights_path=model_weights_path, device=device)
generator = Generator4Embeds(guidance_scale=2.0, num_inference_steps=4, device=device)

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
            # print(f"图片保存至: {save_imgs_path}")
        del batch_images
        torch.cuda.empty_cache()

def main_experiment_loop(seed, subject_id, save_path, target_image_path, device):
    print(seed)
    sub = 'sub-' + (str(subject_id) if subject_id >= 10 else format(subject_id, '02'))
    model_path = f'/mnt/dataset0/jiahua/eeg_encoding/results/{sub}/synthetic_eeg_data/encoding-end_to_end/dnn-alexnet/modeled_time_points-all/pretrained-True/lr-1e-05__wd-0e+00__bs-064/model_state_dict.pt'
    save_target_path = f'/mnt/dataset0/jiahua/closed-loop/alexnet-pretrained-similarity/{sub}-target'
    image_set_path = '/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images'
    target_eeg_path = get_target_eeg(model_path, target_image_path, save_target_path, device)
    target_psd = load_target_psd(target_eeg_path, fs, selected_channel_idxes)
    test_images_path, _ = get_image_pool(image_set_path)
    processed_paths = set([target_image_path])
    print(len(processed_paths))
    num_loops = 60
    # processed_paths = set()
    
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
            print(len(processed_paths))
        else:
            chosen_similarities = all_chosen_similarities[-2:]
            chosen_image_paths = all_chosen_image_paths[-2: ]
        loop_sample_ten = [chosen_image_path for chosen_image_path in chosen_image_paths]
        loop_cs_ten = [chosen_similarity for chosen_similarity in chosen_similarities]
        loop_eeg_ten = [chosen_eeg_path for chosen_eeg_path in chosen_eeg_paths]
        loop_loss_ten = [chosen_loss for chosen_loss in chosen_losses]
        # print(chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths)
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
        # print(loop_sample_ten)
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
        # print(loop_cs_ten)
        # print(loop_loss_ten)
        probabilities = softmax(loop_cs_ten)
        # print(probabilities)

        chosen_similarities, chosen_losses, chosen_image_paths, chosen_eeg_paths = select(probabilities, loop_cs_ten, loop_loss_ten, loop_sample_ten,loop_eeg_ten)
        
        # print("Chosen similarities:", chosen_similarities)
        # print("Chosen losses:", chosen_losses)
        # print("Chosen image paths:", chosen_image_paths)
        # print("Chosen eeg paths:", chosen_eeg_paths)

        for chosen_similarity in chosen_similarities:
            all_chosen_similarities.append(chosen_similarity)
        for chosen_loss in chosen_losses:
            all_chosen_losses.append(chosen_loss)
        for chosen_image_path in chosen_image_paths:
            all_chosen_image_paths.append(chosen_image_path)
        for chosen_eeg_path in chosen_eeg_paths:
            all_chosen_eeg_paths.append(chosen_eeg_path)      

        max_similarity = max(chosen_similarities)
        # print('max_similarity:', max_similarity)
        max_index = chosen_similarities.index(max_similarity)
        corresponding_loss = chosen_losses[max_index]
        # print("Corresponding loss:", corresponding_loss)

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
    # plt.plot(history_loss, marker='x', markersize=3, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend() 
    path = os.path.join(save_path, 'similarities.jpg')
    plt.savefig(path)
    plt.show()
    

if __name__ == "__main__":
    num_run = 3
    num_subjects = 10
    target_image_paths = ['/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images/00039_cheetah/cheetah_05s.jpg'
                        ,'/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images/00140_popcorn/popcorn_07s.jpg'
                        ,'/mnt/dataset0/ldy/4090_Workspace/4090_THINGS/images_set/test_images/00163_seed/seed_07s.jpg']
    for subject_id in range(2, 3):
        base_seed = 100000 * subject_id 
        for i in range(1, num_run + 1):
            target_image_path = target_image_paths[i - 1]
            print(target_image_path)
            base_save_path = f'/mnt/dataset0/jiahua/closed-loop/test_alexnet-pretrained-similarity/exp_sub{subject_id}/loop_random_{i}'
            os.makedirs(base_save_path, exist_ok=True)
            seed = base_seed + i 
            np.random.seed(seed)
            random.seed(seed)
            print(f'Subject {subject_id}/{num_subjects} - Run {i}/{num_run} - Seed: {seed}')
            main_experiment_loop(seed, subject_id, base_save_path, target_image_path, device)
