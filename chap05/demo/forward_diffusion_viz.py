"""
前向扩散过程可视化脚本
Forward diffusion process visualization script

这个脚本用于可视化扩散模型的前向过程，展示图像如何随着时间步逐渐被噪声化。
This script visualizes the forward process of diffusion models, showing how images
gradually become noisier over timesteps.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


def load_image(image_path, image_size=64):
    """
    加载并预处理图像
    Load and preprocess image

    Args:
        image_path: 图像路径 / Path to image
        image_size: 目标图像尺寸 / Target image size

    Returns:
        img_tensor: 预处理后的图像张量 / Preprocessed image tensor
    """
    img = Image.open(image_path)
    img = img.resize((image_size, image_size))
    img_array = np.array(img)

    # 转换为张量并归一化到 [-1, 1]
    # Convert to tensor and normalize to [-1, 1]
    img_tensor = torch.from_numpy(img_array).float() / 127.5 - 1
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return img_tensor


def create_diffusion_model(image_size=64, num_timesteps=1000):
    """
    创建扩散模型
    Create diffusion model

    Args:
        image_size: 图像尺寸 / Image size
        num_timesteps: 扩散时间步数 / Number of diffusion timesteps

    Returns:
        diffusion: 扩散模型 / Diffusion model
    """
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=num_timesteps
    )

    return diffusion


def add_noise_at_timesteps(diffusion, img_tensor, timesteps):
    """
    在指定时间步添加噪声
    Add noise at specified timesteps

    Args:
        diffusion: 扩散模型 / Diffusion model
        img_tensor: 输入图像张量 / Input image tensor
        timesteps: 时间步列表 / List of timesteps

    Returns:
        noisy_images: 噪声图像列表 / List of noisy images
    """
    noisy_images = []

    for t in timesteps:
        t_tensor = torch.tensor([t])
        noisy = diffusion.q_sample(img_tensor, t_tensor)
        noisy_images.append(noisy)

    return noisy_images


def visualize_forward_process(img_tensor, noisy_images, timesteps, output_path):
    """
    可视化前向扩散过程
    Visualize forward diffusion process

    Args:
        img_tensor: 原始图像张量 / Original image tensor
        noisy_images: 噪声图像列表 / List of noisy images
        timesteps: 时间步列表 / List of timesteps
        output_path: 输出路径 / Output path
    """
    num_images = len(noisy_images)
    fig, axes = plt.subplots(1, num_images, figsize=(18, 3))

    for ax, t, noisy in zip(axes, timesteps, noisy_images):
        # 转换为图像格式 / Convert to image format
        noisy_img = noisy[0].permute(1, 2, 0).cpu().numpy()
        noisy_img = ((noisy_img + 1) * 127.5).astype(np.uint8)

        ax.imshow(noisy_img)
        ax.set_title(f't={t}')
        ax.axis('off')

    plt.tight_layout()

    # 保存图像 / Save image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"前向扩散过程已保存到 / Forward process saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='前向扩散过程可视化 / Visualize forward diffusion process'
    )

    parser.add_argument(
        '--image_path',
        type=str,
        default='demo/data/000000.png',
        help='输入图像路径 / Path to input image'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='demo/visualizations/forward_process.png',
        help='输出图像路径 / Path to output image'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=64,
        help='图像尺寸 / Image size'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        nargs='+',
        default=[0, 100, 300, 500, 700, 999],
        help='要可视化的时间步 / Timesteps to visualize'
    )
    parser.add_argument(
        '--num_timesteps',
        type=int,
        default=1000,
        help='扩散总时间步数 / Total number of diffusion timesteps'
    )

    args = parser.parse_args()

    # 打印配置信息 / Print configuration
    print("=" * 60)
    print("前向扩散过程可视化 / Forward Diffusion Process Visualization")
    print("=" * 60)
    print(f"输入图像 / Input image: {args.image_path}")
    print(f"输出路径 / Output path: {args.output_path}")
    print(f"图像尺寸 / Image size: {args.image_size}")
    print(f"时间步 / Timesteps: {args.timesteps}")
    print(f"总时间步数 / Total timesteps: {args.num_timesteps}")
    print("=" * 60)

    # 检查输入图像是否存在 / Check if input image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在 / Image file not found: {args.image_path}")

    # 加载图像 / Load image
    print("\n加载图像 / Loading image...")
    img_tensor = load_image(args.image_path, args.image_size)
    print(f"图像张量形状 / Image tensor shape: {img_tensor.shape}")

    # 创建扩散模型 / Create diffusion model
    print("\n创建扩散模型 / Creating diffusion model...")
    diffusion = create_diffusion_model(args.image_size, args.num_timesteps)

    # 添加噪声 / Add noise
    print("\n添加噪声 / Adding noise...")
    noisy_images = add_noise_at_timesteps(diffusion, img_tensor, args.timesteps)

    # 可视化 / Visualize
    print("\n生成可视化 / Generating visualization...")
    visualize_forward_process(img_tensor, noisy_images, args.timesteps, args.output_path)

    print("\n完成！/ Done!")


if __name__ == '__main__':
    main()
