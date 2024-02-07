import os
import sys
import tempfile
import time
import argparse
from pathlib import Path
from collections import namedtuple
import random
from datetime import datetime
import json

import torch
#Following is needed to avoid running out of FileDescriptors leak/limits 
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.cuda.amp import GradScaler, autocast
# from torch.mps import autocast, GradScaler
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from monai import transforms
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler

from colorama import Fore
from tqdm import tqdm

#for image comparison
import piqa
from skimage.metrics import structural_similarity

import AdniUtils
from CustomTransforms import Random2DSliceTransformd, UnsqueezeTransformd, Sequential2DSliceTransformd

NUM_CLASSES = 5
class_mapping = {
  'CN': 1,
  'EMCI': 2,
  'MCI': 3,
  'LMCI': 4,
  'AD': 5
}

TrainingParams = namedtuple('TrainingParams', [
    'num_iter', 'batch_sz',
    'val_interval', 'val_frac',
    'conditional_dropout',
    'spatial_size'
])
def _getTrainingParams(args):
    if args.debug:
        ret = TrainingParams(
            num_iter=10, batch_sz=4,
            val_interval=1, val_frac=0.2,
            conditional_dropout=0.15,
            spatial_size=128
        )
    else:
        ret = TrainingParams(
            num_iter=2e4, batch_sz=32,
            val_interval=100, val_frac=0.2,
            conditional_dropout=0.15,
            spatial_size=128
        )
    print(f'Training Params: \n\
        {Fore.RED}num_iter                  : {Fore.BLUE}{ret.num_iter} \n\
        {Fore.RED}batch_sz                  : {Fore.BLUE}{ret.batch_sz} \n\
        {Fore.RED}val_interval              : {Fore.BLUE}{ret.val_interval} \n\
        {Fore.RED}val_frac                  : {Fore.BLUE}{ret.val_frac} \n\
        {Fore.RED}conditional_dropout       : {Fore.BLUE}{ret.conditional_dropout}\n\
        {Fore.RED}spatial_size              : {Fore.BLUE}{ret.spatial_size}\n\
        {Fore.WHITE}\
    ')
    return ret

def _getAdniTransforms(training_params, training_mode:bool):
    if training_mode:
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            transforms.EnsureTyped(keys=['image']),
            transforms.Orientationd(keys=['image'], axcodes="RAS"),
            # transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
            transforms.CropForegroundd(keys=['image'], source_key="image"),
            transforms.Resized(keys=['image'], spatial_size=(training_params.spatial_size, training_params.spatial_size, training_params.spatial_size)),
            Random2DSliceTransformd(keys=['image'], lo=0.35, hi=0.75),
            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=0, upper=99.5, b_min=0, b_max=1),
            UnsqueezeTransformd(keys=["image"], axis=0),
        ])
    else:
        return transforms.Compose([
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            transforms.EnsureTyped(keys=['image']),
            transforms.Orientationd(keys=['image'], axcodes="RAS"),
            # transforms.Spacingd(keys=["image", "label"], pixdim=(3.0, 3.0, 2.0), mode=("bilinear", "nearest")),
            transforms.CropForegroundd(keys=['image'], source_key="image"),
            transforms.Resized(keys=['image'], spatial_size=(training_params.spatial_size, training_params.spatial_size, training_params.spatial_size)),
            # Sequential2DSliceTransformd(keys=['image'], lo=0.35, hi=0.355), #debug
            Sequential2DSliceTransformd(keys=['image'], lo=0.35, hi=0.75),
            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=0, upper=99.5, b_min=0, b_max=1),
            UnsqueezeTransformd(keys=["image"], axis=0),
        ])

def perform_training(args):
    #create model_dir for saving all models for this training session
    dir_name = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
    if args.modelNameHint is not None:
        dir_name += f'_{args.modelNameHint}'
    model_dir_path = Path(f'{args.modelsDir}/{dir_name}')
    model_dir_path.mkdir(parents=True, exist_ok=True)

    training_params = _getTrainingParams(args=args)

    device = AdniUtils.getDevice(args=args)
    print(f"Using {device}")

    data_loaders = AdniUtils.get_adni_dataloaders(
        args=args, 
        batch_sz=training_params.batch_sz,
        adniTrnsrfms=_getAdniTransforms(training_params, training_mode=True), 
        val_frac=training_params.val_frac
    )

    # ## Define network, scheduler, optimizer, and inferer
    #
    # At this step, we instantiate the MONAI components to create:
    #       a DDIM Scheduler,
    #       UNET with conditioning, 
    #       noise scheduler, 
    #       inferer used for training and sampling. 
    # We are using the deterministic DDIM scheduler containing 1000 timesteps, and a 2D UNET with attention mechanisms.
    #
    # The `attention` mechanism is essential for ensuring good conditioning and images manipulation here.
    #
    # An `embedding layer`, which is also optimised during training, is used in the original work because 
    # it was empirically shown to improve conditioning compared to a single scalar information.
    #
    embedding_dimension = 64
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=16,
        with_conditioning=True,
        cross_attention_dim=embedding_dimension,
    ).to(device)
    embed = torch.nn.Embedding(num_embeddings=NUM_CLASSES+1, embedding_dim=embedding_dimension, padding_idx=0).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(params=list(model.parameters()) + list(embed.parameters()), lr=1e-5)
    inferer = DiffusionInferer(scheduler)

    iter_loss_list = []
    val_iter_loss_list = []
    iterations = []
    iteration = 0
    iter_loss = 0

    scaler = GradScaler()
    total_start = time.time()

    while iteration < training_params.num_iter:
        progress_bar = tqdm(enumerate(data_loaders.train_loader), total=len(data_loaders.train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {iteration}")
        if args.device == 'mps':
            torch.mps.profiler.start(mode="interval")
        for step, batch in progress_bar:
            iteration += 1
            model.train() #opposite of model.eval()
            images, classes = batch["image"].float().to(device), batch["slice_label"].float().to(device)
            # 15% of the time, class conditioning dropout
            classes = classes * (torch.rand_like(classes) > training_params.conditional_dropout)
            # cross attention expects shape [batch size, sequence length, channels]
            class_embedding = embed(classes.long().to(device)).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            # pick a random time step t
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)

            # with autocast(device_type="mps", dtype=torch.bfloat16):
            with autocast(enabled=True): #For Cuda
                # Generate random noise
                noise = torch.randn_like(images).to(device)
                # Get model prediction
                noise_pred = inferer(
                    inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, condition=class_embedding
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter_loss += loss.item()
            sys.stdout.write(f"Iteration {iteration}/{training_params.num_iter} - train Loss {loss.item():.4f}" + "\r")
            sys.stdout.flush()

            if (iteration) % training_params.val_interval == 0:
                model.eval()
                val_iter_loss = 0
                for val_step, val_batch in enumerate(data_loaders.val_loader):
                    images, classes = val_batch["image"].float().to(device), val_batch["slice_label"].float().to(device)
                    # cross attention expects shape [batch size, sequence length, channels]
                    class_embedding = embed(classes.long().to(device)).unsqueeze(1)
                    timesteps = torch.randint(0, 1000, (len(images),)).to(device)
                    with torch.no_grad():
                        with autocast(enabled=True):
                            noise = torch.randn_like(images).float().to(device)
                            noise_pred = inferer(
                                inputs=images,
                                diffusion_model=model,
                                noise=noise,
                                timesteps=timesteps,
                                condition=class_embedding,
                            )
                            val_loss = F.mse_loss(noise_pred.float(), noise.float())
                    val_iter_loss += val_loss.item()
                iter_loss_list.append(iter_loss / training_params.val_interval)
                val_iter_loss_list.append(val_iter_loss / (val_step + 1))
                iterations.append(iteration)
                iter_loss = 0
                print(
                    f"Train Loss {loss.item():.4f}, Interval Loss {iter_loss_list[-1]:.4f}, Interval Loss Val {val_iter_loss_list[-1]:.4f}"
                )
        if args.device == 'mps':
            torch.mps.profiler.stop()

    total_time = time.time() - total_start

    print(f"train diffusion completed, total time: {total_time}.")
    modelFname = args.diffusionModelFileName if args.diffusionModelFileName is not None else 'adni_128x128_diffusion.pt'
    AdniUtils.save_model(model, model_dir_path, args, modelFname)
    embFname = args.embFileName if args.embFileName is not None else 'adni_128x128_embedding.pt'
    AdniUtils.save_model(embed, model_dir_path, args, embFname)

    plt.ion()
    plt.style.use("seaborn-v0_8")
    plt.figure('metrics')
    plt.title("Learning Curves Diffusion Model", fontsize=20)
    plt.plot(iterations, iter_loss_list, color="C0", linewidth=2.0, label="Train")
    plt.plot(
        iterations, val_iter_loss_list, color="C1", linewidth=2.0, label="Validation"
    )  # np.linspace(1, n_iterations, len(val_iter_loss_list))
    plt.yticks(fontsize=12), plt.xticks(fontsize=12)
    plt.xlabel("Iterations", fontsize=16), plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    plt.show()
    plt.savefig(f"{model_dir_path}/metrics.png")
    if args.debug:
        input("Press Enter to continue...")

    # ## Generate synthetic data
    # (Only for verifying the model and classifier guidance is working)
    model.eval()
    scheduler.clip_sample = True
    guidance_scale = 3
    conditioning = torch.cat([torch.zeros(1).long(), 2 * torch.ones(1).long()], dim=0).to(
        device
    )  # 2*torch.ones(1).long() is the class label for the UNHEALTHY class
    class_embedding = embed(conditioning).unsqueeze(1)  # cross attention expects shape [batch size, sequence length, channels]
    noise = torch.randn((1, 1, training_params.spatial_size, training_params.spatial_size), device=device).float()
    noise = noise.float().to(device)
    scheduler.set_timesteps(num_inference_steps=100)
    progress_bar = tqdm(scheduler.timesteps)
    for t in progress_bar:
        with autocast(enabled=True):
            with torch.no_grad():
                noise_input = torch.cat([noise] * 2)
                model_output = model(noise_input, timesteps=torch.Tensor((t,)).to(noise.device), context=class_embedding)
                noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        noise, _ = scheduler.step(noise_pred, t, noise)

    plt.style.use("default")
    plt.figure('Sample Image from Model')
    plt.imshow(noise[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
    plt.savefig(f"{model_dir_path}/sample_image_from_model.png")
    if args.debug:
        input("Press Enter to continue...")

    # # Image-to-Image Translation to a Healthy Subject
    # We pick a diseased subject of the validation set as input image. 
    # We want to translate it to its healthy reconstruction.
    val_batch = next(iter(data_loaders.val_loader))
    idx_unhealthy = np.argwhere(val_batch["slice_label"].numpy() != 1).squeeze()
    if len(idx_unhealthy.shape) > 0:
        idx = random.choice(idx_unhealthy)  # Pick a random slice of the validation set to be transformed
    else:
        idx = idx_unhealthy

    inputting = val_batch["image"][idx]  # Pick an input slice of the validation set to be transformed
    inputlabel = val_batch["slice_label"][idx]  # Check whether it is healthy or diseased

    plt.figure("val_input" + str(inputlabel))
    plt.imshow(inputting[0], vmin=0, vmax=1, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{model_dir_path}/val_input" + str(inputlabel) + ".png")
    print("input label: ", inputlabel)
    if args.debug:
        input("Press Enter to continue...")

def _get_trained_models(device, args) -> (DiffusionModelUNet, torch.nn.Embedding):
    diffModelPath = Path(f'{args.modelsDir}/{args.diffusionModelFileName}')
    embeddingModelPath = Path(f'{args.modelsDir}/{args.embFileName}')
    if diffModelPath.is_file() and embeddingModelPath.is_file():
        embedding_dimension = 64
        model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 64, 64),
            attention_levels=(False, True, True),
            num_res_blocks=1,
            num_head_channels=16,
            with_conditioning=True,
            cross_attention_dim=embedding_dimension,
        ).to(device)
        embed = torch.nn.Embedding(num_embeddings=NUM_CLASSES+1, 
                                    embedding_dim=embedding_dimension, 
                                    padding_idx=0).to(device)
        model.load_state_dict(torch.load(diffModelPath, map_location=device))
        embed.load_state_dict(torch.load(embeddingModelPath, map_location=device))
        return (model, embed)
    print(f'Pretrained models not found at:\n{Fore.GREEN}{diffModelPath}\n{embeddingModelPath}\n{Fore.WHITE}')
    return None

def perform_inference(args):
    training_params = _getTrainingParams(args=args)
    device = AdniUtils.getDevice(args=args)
    print(f"Using {device}")

    model, embed = _get_trained_models(device, args)

    if args.infDict:
        inf_tx = _getAdniTransforms(training_params, training_mode=False)
        seq2DSliceTx: Sequential2DSliceTransformd = inf_tx.transforms[inf_tx.get_index_of_first(lambda t: isinstance(t, Sequential2DSliceTransformd))]
        infJson = json.loads(args.infDict)
        
        #create output_dir for the ImageId
        image_id = AdniUtils.get_imageid_from_path(infJson['image'])
        model_dir_path = Path(f'{args.inferenceOutDir}/{image_id}')
        model_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        print('infDict(inference Dictionary) not specified for inference: use "--infDict"')
        return
    
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    # ### The image-to-image translation has two steps
    #
    # 1. Encoding the input image into a latent space with the reversed DDIM sampling scheme
    # 2. Sampling from the latent space using gradient guidance towards the desired class label `y=1` (healthy)
    #
    # In order to sample using gradient guidance, we first need to encode the input image in noise by using the reversed DDIM sampling scheme.
    # We define the number of steps in the noising and denoising process by `L`.

    model.eval()

    guidance_scale = 3.0
    total_timesteps = 500
    latent_space_depth = int(total_timesteps * 0.25)

    yN = 'y'
    slice_sim_scores = []
    while (yN == 'y' or yN == 'Y' or yN == '') and (seq2DSliceTx.is_next_slice_available == True):
        srcImg = inf_tx(infJson)['image']

        current_img = srcImg[None, ...].float().to(device)
        scheduler.set_timesteps(num_inference_steps=total_timesteps)

        ## Encoding
        print(f'\n\n{Fore.MAGENTA}Encoding/Noising input Image to Latent Space{Fore.WHITE} {Path(infJson["image"]).stem} ...')
        scheduler.clip_sample = False
        class_embedding = embed(torch.zeros(1).long().to(device)).unsqueeze(1) # ONE is the class-label for healthy/CN (which is the conditioning we want)
        progress_bar = tqdm(range(latent_space_depth))
        for i in progress_bar:  # go through the noising process
            t = i
            with torch.no_grad():
                model_output = model(current_img, timesteps=torch.Tensor((t,)).to(current_img.device), context=class_embedding)
            current_img, _ = scheduler.reversed_step(model_output, t, current_img)
            progress_bar.set_postfix({"timestep input": t})

        latent_img = current_img

        ## Decoding
        print(f'\n\n{Fore.MAGENTA}Sampling from the latent-space using Gradient Guidance towards the desired class label `y=1` (healthy/CN) ...{Fore.WHITE}')
        conditioning = torch.cat([torch.zeros(1).long(), torch.ones(1).long()], dim=0).to(device) #(Contioning, Unconditioning)
        class_embedding = embed(conditioning).unsqueeze(1)

        progress_bar = tqdm(range(latent_space_depth))
        for i in progress_bar:  # go through the denoising process
            t = latent_space_depth - i
            current_img_double = torch.cat([current_img] * 2)
            with torch.no_grad():
                model_output = model(
                    current_img_double, timesteps=torch.Tensor([t, t]).to(current_img.device), context=class_embedding
                )
            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            current_img, _ = scheduler.step(noise_pred, t, current_img)
            progress_bar.set_postfix({"timestep input": t})
            if args.device == 'mps':
                torch.mps.empty_cache()
            elif args.device == 'cuda':
                torch.cuda.empty_cache()

        # ### Visualize anomaly map
        def visualize(img):
            _min = img.min()
            _max = img.max()
            normalized_img = (img - _min) / (_max - _min)
            return normalized_img


        diff = abs(srcImg.cpu() - current_img[0].cpu()).detach().numpy()

        x = srcImg.cpu().squeeze(0)
        y = current_img[0].cpu().squeeze(0)

        #Perform Contrast normalization
        x = (x-x.min())/(x.max()-x.min())
        y = (y-y.min())/(y.max()-y.min())        

        scikit_ssim = structural_similarity(x.numpy().astype('float32'), 
                                            y.numpy().astype('float32'), 
                                            multichannel=False, 
                                            win_size=3, data_range=1.0)
        #piqa
        piqa_ssim=piqa.SSIM(n_channels=1)(x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0))
        piqa_psnr = piqa.PSNR()(x, y)

        #Compute FID
        piqa_fid = piqa.FID()
        fid_score = piqa_fid(x, y)

        #Compute MSID
        piqa_msssim = piqa.MS_SSIM(n_channels=1, window_size=8)
        msssim_score = piqa_msssim(x.unsqueeze(0).unsqueeze(0), y.unsqueeze(0).unsqueeze(0))

        #LPIPS
        lpips_score = piqa.LPIPS()(
            torch.cat([x.unsqueeze(0)]*3).unsqueeze(0),
            torch.cat([y.unsqueeze(0)]*3).unsqueeze(0)
        )

        #Visual Saliency Indicator
        vsi_score = piqa.VSI()(
            torch.cat([x.unsqueeze(0)]*3).unsqueeze(0),
            torch.cat([y.unsqueeze(0)]*3).unsqueeze(0)
        )
        
        print(f'scikit-ssim: {scikit_ssim}')
        print(f'piqa: ssim: {piqa_ssim}, psnr: {piqa_psnr}, fid: {fid_score}, mssim: {msssim_score}, lpips: {lpips_score}, vsi: {vsi_score}')

        num_cols = 7
        num_rows = 2
        plt.style.use("default")

        fig = plt.figure(figsize=(8*3, num_cols))
        sns.set_palette('pastel')
        
        plt.suptitle(f'{seq2DSliceTx.curr_slice_str_rep} : sksim={scikit_ssim:.2f}', 
                     fontsize=15,
                     color='#c167d1')
        
        plt.tight_layout()

        ax = plt.subplot(num_rows, num_cols, 1)
        ax.imshow(srcImg[0], vmin=0, vmax=1, cmap="gray")
        ax.set_title("Original Unhealthy Image"), plt.tight_layout()#, plt.axis("off")

        ax = plt.subplot(num_rows, num_cols, 2)
        ax.imshow(x, vmin=0, vmax=1, cmap="gray")
        ax.set_title("Original(Contrast Normalized)"), plt.tight_layout()#, plt.axis("off")

        ax = plt.subplot(num_rows, num_cols, 3)
        ax.imshow(latent_img[0, 0].cpu().detach().numpy(), vmin=0, vmax=1, cmap="gray")
        ax.set_title("Latent Image"), plt.tight_layout()#, plt.axis("off")

        ax = plt.subplot(num_rows, num_cols, 4)
        ax.imshow(current_img[0, 0].cpu().detach().numpy(), vmin=0, vmax=1, cmap="gray")
        ax.set_title("Reconstructed Healthy Image"), plt.tight_layout()#, plt.axis("off")

        ax = plt.subplot(num_rows, num_cols, 5)
        ax.imshow(y, vmin=0, vmax=1, cmap="gray")
        ax.set_title("Reconstructed(contrast normalized)"), plt.tight_layout()#, plt.axis("off")

        ax = plt.subplot(num_rows, num_cols, 6)
        ax.imshow(diff[0], cmap="inferno")
        ax.set_title(f"Anomaly Map: \nsksim={scikit_ssim:.2f}"), plt.tight_layout()#, plt.axis("off")

        ax = plt.subplot(num_rows, num_cols, 7)
        ax.imshow(visualize(diff[0]), cmap="inferno")
        ax.set_title(f"(viz)Anomaly Map: \nsksim={scikit_ssim:.2f}"), plt.tight_layout()#, plt.axis("off")


        #--- row-2
        ax = plt.subplot(num_rows, num_cols, 8)
        # sns.histplot(srcImg[0], bins=25, kde=True, ax=ax, element='poly', legend=False)
        sns.histplot(srcImg[0], ax=ax, element='poly', legend=False)
        ax.set_title("Original Unhealthy Image"), plt.tight_layout()

        ax = plt.subplot(num_rows, num_cols, 9)
        sns.histplot(x, ax=ax, element='poly', legend=False)
        ax.set_title("Original(Contrast Normalized)"), plt.tight_layout()

        ax = plt.subplot(num_rows, num_cols, 10)
        sns.histplot(latent_img[0, 0].cpu().detach().numpy(),ax=ax, element='poly', legend=False)
        ax.set_title("Latent Image"), plt.tight_layout()

        ax = plt.subplot(num_rows, num_cols, 11)
        sns.histplot(current_img[0, 0].cpu().detach().numpy(), ax=ax, element='poly', legend=False)
        ax.set_title("Reconstructed Healthy Image"), plt.tight_layout()

        ax = plt.subplot(num_rows, num_cols, 12)
        sns.histplot(y, ax=ax, element='poly', legend=False)
        ax.set_title("Reconstructed(Contrast Normalized)"), plt.tight_layout()

        ax = plt.subplot(num_rows, num_cols, 13)
        sns.histplot(diff[0], ax=ax, element='poly', legend=False)
        ax.set_title("Anomaly Map"), plt.tight_layout()

        ax = plt.subplot(num_rows, num_cols, 14)
        sns.histplot(visualize(diff[0]), ax=ax, element='poly', legend=False)
        ax.set_title("(viz)Anomaly Map"), plt.tight_layout()

        save_fname = seq2DSliceTx.curr_slice_str_rep.replace("[", "").replace("]", "").replace(", ", "_")
        plt.savefig(f"{model_dir_path}/{save_fname}.png")
        slice_sim_scores.append([
            save_fname, 
            scikit_ssim, 
            piqa_ssim.item(), 
            piqa_psnr.item(), 
            fid_score.item(), 
            msssim_score.item(),
            lpips_score.item(),
            vsi_score.item()
        ])

        if args.debug:
            plt.show()
        if args.debug:
            yN = input("Sample one more slice? Yy/nN ?: ")
    print(slice_sim_scores)
    with open(f"{model_dir_path}/slice_sim_scores.json", 'w') as f:
        json.dump(slice_sim_scores, f)

    slice_sim_scores_df = pd.DataFrame(columns=['fname', "sksim", "piqa_sim", "psnr", "fid", "mssim", "lpips", "vsi"])
    #plot metrics for all slices
    slice_sim_scores_df.reset_index(inplace=True)
    metrics_names_df = pd.DataFrame(slice_sim_scores_df.columns[2:], columns=["metric"])
    metrics_names_df.reset_index(inplace=True)

    #seaborn-scatterplot
    sns_graph = sns.FacetGrid(metrics_names_df, col="metric", col_wrap=4, sharex=False, sharey=False)
    for ax, y_var in zip(sns_graph.axes, metrics_names_df["metric"]):
        print(f'ax={ax}, y_var={y_var}')
        sns.scatterplot(data=slice_sim_scores_df, ax=ax, x="index", y=y_var, legend=False, hue=y_var, palette="husl", alpha=1, size=1)
        sns_graph.tight_layout()
    plt.savefig(f"{model_dir_path}/metrics_sns.png")

    #matplotlib-scatterplot
    nrows = 2
    ncols = int(len(metrics_names_df)/nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    plt_colors=['red', 'blue', 'green', 'purple', 'orange', 'brown', 'olive', 'pink']

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            idx = r * ncols + c
            ax.scatter(slice_sim_scores_df["index"], slice_sim_scores_df[metrics_names_df["metric"][idx]], alpha=0.5, c=plt_colors[idx])
            ax.set_title(metrics_names_df["metric"][idx])
            plt.tight_layout()
    plt.savefig(f"{model_dir_path}/metrics_matplotlib.png")
    if args.debug:
        plt.show()


def main(args):
    if args.training:
        perform_training(args)
    elif args.inference:
        perform_inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2D Diffusion Anomoly detection on Adni dataset')
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help='Run in DEBUG mode')
    parser.add_argument("--device", help='Device to run on: cpu|mps|cuda', default='mps')
    parser.add_argument("-d", "--datasetDir", help='Directory containing ADNI MRI files')
    parser.add_argument("-m", "--modelsDir", help='Directory to store models. If a model already exists, training script can bypass that part.')
    
    parser.add_argument("-t", "--training", action=argparse.BooleanOptionalAction, help='Perform Training')
    parser.add_argument("--modelNameHint", help='Helpful hint to attach to trained-model names')

    parser.add_argument("-i", "--inference", action=argparse.BooleanOptionalAction, help='Perform inference')
    parser.add_argument("--inferenceOutDir", type=str, help='Root Dir for storing inference-output artifacts', required=True)
    parser.add_argument("--diffusionModelFileName", help='Diffusion model Name', default='adni_128x128_diffusion.pt')
    parser.add_argument("--embFileName", help='Embedding File Name', default='adni_128x128_embedding.pt')
    parser.add_argument("--infDict", 
                        help='"{\"image\": \"${HOME}/adni/data/adni_go_mci_mri/ADNI/137_S_0722/MPR-R__GradWarp/2010-11-19_13_20_56.0/I218253/ADNI_137_S_0722_MR_MPR-R__GradWarp_Br_20110218130435170_S95085_I218253.nii\"}"'
                        )
    parser.add_argument("--make_symlinks", action=argparse.BooleanOptionalAction, 
                        help='Create Symlinks for generated models')
    args = parser.parse_args()

    print("\n-------------------")
    for k, v in class_mapping.items():
        print(f'{Fore.BLUE}{k}\t:{Fore.MAGENTA}{v}{Fore.RESET}')
    print("-------------------")

    main(args)

    print("\n-------------------")
    for k, v in class_mapping.items():
        print(f'{Fore.BLUE}{k}\t:{Fore.MAGENTA}{v}{Fore.RESET}')
    print("-------------------")
