"""
扩散模型可视化脚本
Diffusion model visualization script

这个脚本用于可视化扩散模型的训练过程、生成结果等。
This script is used to visualize the training process and generation results of diffusion models.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from torchvision.utils import make_grid


def visualize_training_results(results_folder, output_path='./demo/visualizations'):
    """
    可视化训练结果
    Visualize training results

    Args:
        results_folder: 结果文件夹 / Results folder
        output_path: 输出路径 / Output path
    """
    results_path = Path(results_folder)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有样本图像 / Find all sample images
    sample_files = sorted(results_path.glob('sample-*.png'))

    if not sample_files:
        print(f"未找到样本图像 / No sample images found in {results_folder}")
        return

    print(f"找到 {len(sample_files)} 个样本图像 / Found {len(sample_files)} sample images")

    # 创建训练进度可视化 / Create training progress visualization
    num_samples = min(9, len(sample_files))
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        img = Image.open(sample_files[i])
        axes[i].imshow(img)
        axes[i].set_title(f'Step {i * 1000}', fontsize=10)
        axes[i].axis('off')

    # 隐藏多余的子图 / Hide extra subplots
    for i in range(num_samples, 9):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / 'training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"训练进度可视化已保存到 / Training progress visualization saved to: {output_path / 'training_progress.png'}")


def visualize_data_samples(data_folder, num_samples=16, output_path='./demo/visualizations'):
    """
    可视化数据集样本
    Visualize dataset samples

    Args:
        data_folder: 数据文件夹 / Data folder
        num_samples: 样本数量 / Number of samples
        output_path: 输出路径 / Output path
    """
    data_path = Path(data_folder)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取图像文件 / Get image files
    image_files = sorted(data_path.glob('*.png')) + sorted(data_path.glob('*.jpg')) + sorted(data_path.glob('*.jpeg'))

    if not image_files:
        print(f"未找到图像文件 / No image files found in {data_folder}")
        return

    # 随机选择样本 / Randomly select samples
    num_samples = min(num_samples, len(image_files))
    selected_files = np.random.choice(image_files, num_samples, replace=False)

    # 创建网格可视化 / Create grid visualization
    images = []
    for file in selected_files:
        img = Image.open(file)
        images.append(img)

    # 计算网格大小 / Calculate grid size
    nrow = int(np.ceil(np.sqrt(num_samples)))
    ncol = int(np.ceil(num_samples / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)

    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        axes[row, col].imshow(img)
        axes[row, col].set_title(selected_files[i].name, fontsize=8)
        axes[row, col].axis('off')

    # 隐藏多余的子图 / Hide extra subplots
    for i in range(num_samples, nrow * ncol):
        row = i // ncol
        col = i % ncol
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / 'data_samples.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"数据集样本可视化已保存到 / Dataset samples visualization saved to: {output_path / 'data_samples.png'}")


def compare_real_vs_generated(data_folder, generated_folder, num_samples=8, output_path='./demo/visualizations'):
    """
    比较真实图像和生成图像
    Compare real and generated images

    Args:
        data_folder: 真实数据文件夹 / Real data folder
        generated_folder: 生成数据文件夹 / Generated data folder
        num_samples: 样本数量 / Number of samples
        output_path: 输出路径 / Output path
    """
    data_path = Path(data_folder)
    generated_path = Path(generated_folder)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取图像文件 / Get image files
    real_files = sorted(data_path.glob('*.png')) + sorted(data_path.glob('*.jpg')) + sorted(data_path.glob('*.jpeg'))
    generated_files = sorted(generated_path.glob('sample_*.png'))

    if not real_files or not generated_files:
        print(f"未找到足够的图像 / Not enough images found")
        return

    # 随机选择样本 / Randomly select samples
    num_samples = min(num_samples, len(real_files), len(generated_files))
    selected_real = np.random.choice(real_files, num_samples, replace=False)
    selected_generated = np.random.choice(generated_files, num_samples, replace=False)

    # 创建对比可视化 / Create comparison visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    for i in range(num_samples):
        # 真实图像 / Real image
        real_img = Image.open(selected_real[i])
        axes[0, i].imshow(real_img)
        axes[0, i].set_title('Real', fontsize=10)
        axes[0, i].axis('off')

        # 生成图像 / Generated image
        gen_img = Image.open(selected_generated[i])
        axes[1, i].imshow(gen_img)
        axes[1, i].set_title('Generated', fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path / 'real_vs_generated.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"对比可视化已保存到 / Comparison visualization saved to: {output_path / 'real_vs_generated.png'}")


def visualize_noise_schedule(diffusion, output_path='./demo/visualizations'):
    """
    可视化噪声调度
    Visualize noise schedule

    Args:
        diffusion: 扩散模型 / Diffusion model
        output_path: 输出路径 / Output path
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取噪声调度参数 / Get noise schedule parameters
    betas = diffusion.betas.cpu().numpy()
    alphas = 1 - betas
    alphas_cumprod = diffusion.alphas_cumprod.cpu().numpy()

    # 创建可视化 / Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Beta schedule
    axes[0].plot(betas)
    axes[0].set_title('Beta Schedule', fontsize=12)
    axes[0].set_xlabel('Timestep', fontsize=10)
    axes[0].set_ylabel('Beta', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Alpha schedule
    axes[1].plot(alphas)
    axes[1].set_title('Alpha Schedule', fontsize=12)
    axes[1].set_xlabel('Timestep', fontsize=10)
    axes[1].set_ylabel('Alpha', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Alpha cumulative product
    axes[2].plot(alphas_cumprod)
    axes[2].set_title('Alpha Cumulative Product', fontsize=12)
    axes[2].set_xlabel('Timestep', fontsize=10)
    axes[2].set_ylabel('Alpha Cumprod', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'noise_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"噪声调度可视化已保存到 / Noise schedule visualization saved to: {output_path / 'noise_schedule.png'}")


def main():
    parser = argparse.ArgumentParser(description='扩散模型可视化 / Diffusion model visualization')

    parser.add_argument('--mode', type=str, required=True,
                        choices=['training', 'data', 'compare', 'noise_schedule'],
                        help='可视化模式 / Visualization mode')
    parser.add_argument('--data_folder', type=str, default='./demo/data',
                        help='数据文件夹 / Data folder')
    parser.add_argument('--results_folder', type=str, default='./demo/results',
                        help='结果文件夹 / Results folder')
    parser.add_argument('--generated_folder', type=str, default='./demo/samples',
                        help='生成图像文件夹 / Generated images folder')
    parser.add_argument('--output_path', type=str, default='./demo/visualizations',
                        help='输出路径 / Output path')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='样本数量 / Number of samples')

    args = parser.parse_args()

    if args.mode == 'training':
        visualize_training_results(args.results_folder, args.output_path)
    elif args.mode == 'data':
        visualize_data_samples(args.data_folder, args.num_samples, args.output_path)
    elif args.mode == 'compare':
        compare_real_vs_generated(
            args.data_folder,
            args.generated_folder,
            args.num_samples,
            args.output_path
        )
    elif args.mode == 'noise_schedule':
        # 需要先加载扩散模型 / Need to load diffusion model first
        from denoising_diffusion_pytorch import Unet, GaussianDiffusion

        model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=3)
        diffusion = GaussianDiffusion(
            model,
            image_size=64,
            timesteps=1000,
            beta_schedule='sigmoid'
        )
        visualize_noise_schedule(diffusion, args.output_path)

    print("\n完成！/ Done!")


if __name__ == '__main__':
    main()
