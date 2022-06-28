"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from PIL import Image


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure('./logSampling')
    logger.log(f"pid   : {os.getpid()}")
    logger.log(f"cwd   : {os.getcwd()}")
    logger.log(f"torch.initial_seed(): {th.initial_seed()}")
    logger.log(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.log(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log(f"model.load_state_dict: {args.model_path}")
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cuda"))
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    # arr shape: [B, H, W, C], such as [100, 32, 32, 3]
    # and arr element value is from [0, 255]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        folder = logger.get_dir()
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
            save_image_one_by_one(folder, arr, label_arr)
        else:
            np.savez(out_path, arr)
            save_image_one_by_one(folder, arr)

    dist.barrier()
    logger.log("sampling complete")


def save_image_one_by_one(folder, arr, label_arr=None):
    """
    save image one by one
    :param folder:
    :param arr:       shape should be [B, H, W, C]
    :param label_arr:
    :return:
    """
    for idx in range(len(arr)):
        if label_arr:
            out_path = os.path.join(folder, f"sample_{idx:03d}_{label_arr[idx]}.png")
        else:
            out_path = os.path.join(folder, f"sample_{idx:03d}.png")
        img = arr[idx]  # [H, W, C]
        save_image(out_path, img)


def save_image(filename, image_255):
    """
    save image into file
    :param image_255: Tensor with shape: 32x32x3; value range [0, 255]
    :param filename
    :return:
    """
    image_np = image_255
    image_np = image_np.astype(np.uint8)
    res = Image.fromarray(image_np)
    res.save(filename)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        # model_path="./model_pt/cifar10_ema_0.9999_050000.pt",
        model_path="./model_pt/imagenet64_uncond_100M_1500K.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults['image_size'] = 64
    defaults['num_channels'] = 128
    defaults['num_res_blocks'] = 3
    defaults['diffusion_steps'] = 4000
    defaults['noise_schedule'] = 'cosine'
    defaults['learn_sigma'] = True
    defaults['timestep_respacing'] = ''

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
