"""
扩散模型采样/测试脚本
Diffusion model sampling/testing script

这个脚本用于加载训练好的扩散模型并生成样本。
This script is used to load a trained diffusion model and generate samples.
"""

import torch
from pathlib import Path
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import argparse
from torchvision.utils import save_image
import math


def load_model(checkpoint_path, device='cuda'):
    """
    加载训练好的模型
    Load trained model

    Args:
        checkpoint_path: 模型检查点路径 / Path to model checkpoint
        device: 设备 / Device

    Returns:
        diffusion: 加载的扩散模型 / Loaded diffusion model
    """
    print(f"加载模型 / Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 从checkpoint中获取模型配置 / Get model config from checkpoint
    # 注意：这里需要根据实际的checkpoint结构调整
    # Note: This needs to be adjusted based on actual checkpoint structure

    # 创建模型 / Create model
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    )

    # 创建扩散过程 / Create diffusion process
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=250,
        beta_schedule='sigmoid',
        objective='pred_v'
    )

    # 加载模型权重 / Load model weights
    # 处理可能带有 'model.' 前缀的权重 / Handle weights possibly prefixed with 'model.'
    state_dict = checkpoint['model']
    # 移除 'model.' 前缀 / Remove 'model.' prefix
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

    # 过滤掉GaussianDiffusion的参数，只保留UNet模型的权重
    # Filter out GaussianDiffusion parameters, keep only UNet model weights
    diffusion_params = {
        'betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod',
        'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod',
        'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod',
        'posterior_variance', 'posterior_log_variance_clipped',
        'posterior_mean_coef1', 'posterior_mean_coef2', 'loss_weight'
    }
    model_state_dict = {k: v for k, v in state_dict.items() if k not in diffusion_params}

    model.load_state_dict(model_state_dict)
    model.eval()

    # 将整个diffusion对象移动到设备 / Move entire diffusion object to device
    diffusion = diffusion.to(device)

    print(f"模型加载成功！/ Model loaded successfully!")
    print(f"训练步数 / Training steps: {checkpoint.get('step', 'unknown')}")

    return diffusion


def sample_images(diffusion, batch_size=16, output_path='./demo/samples', device='cuda'):
    """
    生成样本图像
    Generate sample images

    Args:
        diffusion: 扩散模型 / Diffusion model
        batch_size: 批次大小 / Batch size
        output_path: 输出路径 / Output path
        device: 设备 / Device
    """
    print(f"\n开始采样 / Starting sampling...")
    print(f"批次大小 / Batch size: {batch_size}")
    print(f"设备 / Device: {device}")

    # 生成样本 / Generate samples
    with torch.no_grad():
        sampled_images = diffusion.sample(batch_size=batch_size)

    print(f"采样完成！/ Sampling completed!")
    print(f"生成图像形状 / Generated image shape: {sampled_images.shape}")

    # 保存图像 / Save images
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 将图像保存为网格 / Save images as grid
    nrow = int(math.sqrt(batch_size))
    save_image(sampled_images, output_path / 'sampled_images.png', nrow=nrow)

    print(f"图像已保存到 / Images saved to: {output_path / 'sampled_images.png'}")

    # 保存单独的图像 / Save individual images
    for i, img in enumerate(sampled_images):
        save_image(img, output_path / f'sample_{i:04d}.png')

    print(f"单独图像已保存到 / Individual images saved to: {output_path}")

    return sampled_images


def interpolate_images(diffusion, x1, x2, num_steps=5, output_path='./demo/interpolation'):
    """
    在两个图像之间进行插值
    Interpolate between two images

    Args:
        diffusion: 扩散模型 / Diffusion model
        x1: 第一个图像 / First image
        x2: 第二个图像 / Second image
        num_steps: 插值步数 / Number of interpolation steps
        output_path: 输出路径 / Output path
    """
    print(f"\n开始插值 / Starting interpolation...")
    print(f"插值步数 / Interpolation steps: {num_steps}")

    # 生成插值图像 / Generate interpolated images
    with torch.no_grad():
        interpolated = []
        for lam in torch.linspace(0, 1, num_steps):
            result = diffusion.interpolate(x1, x2, lam=lam.item())
            interpolated.append(result)

        interpolated = torch.cat(interpolated, dim=0)

    # 保存插值图像 / Save interpolated images
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    save_image(interpolated, output_path / 'interpolation.png', nrow=num_steps)

    print(f"插值图像已保存到 / Interpolated images saved to: {output_path / 'interpolation.png'}")


def visualize_diffusion_process(diffusion, num_steps=10, output_path='./demo/diffusion_process'):
    """
    可视化扩散过程
    Visualize diffusion process

    Args:
        diffusion: 扩散模型 / Diffusion model
        num_steps: 可视化的步数 / Number of steps to visualize
        output_path: 输出路径 / Output path
    """
    print(f"\n可视化扩散过程 / Visualizing diffusion process...")
    print(f"可视化步数 / Visualization steps: {num_steps}")

    # 生成样本并返回所有中间步骤 / Generate samples and return all intermediate steps
    with torch.no_grad():
        sampled_images = diffusion.sample(
            batch_size=1,
            return_all_timesteps=True
        )

    # sampled_images shape: (batch_size, timesteps+1, channels, height, width)
    # 选择要可视化的步骤 / Select steps to visualize
    total_steps = sampled_images.shape[1]
    step_indices = torch.linspace(0, total_steps - 1, num_steps, dtype=torch.long)

    selected_images = sampled_images[0, step_indices]

    # 保存扩散过程 / Save diffusion process
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    save_image(selected_images, output_path / 'diffusion_process.png', nrow=num_steps)

    print(f"扩散过程已保存到 / Diffusion process saved to: {output_path / 'diffusion_process.png'}")


def main():
    parser = argparse.ArgumentParser(description='扩散模型采样 / Diffusion model sampling')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径 / Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 / Batch size')
    parser.add_argument('--output_path', type=str, default='./demo/samples',
                        help='输出路径 / Output path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 / Device')
    parser.add_argument('--visualize_process', action='store_true',
                        help='可视化扩散过程 / Visualize diffusion process')
    parser.add_argument('--num_process_steps', type=int, default=10,
                        help='可视化的扩散步数 / Number of diffusion steps to visualize')

    args = parser.parse_args()

    # 加载模型 / Load model
    diffusion = load_model(args.checkpoint, args.device)

    # 生成样本 / Generate samples
    sample_images(
        diffusion,
        batch_size=args.batch_size,
        output_path=args.output_path,
        device=args.device
    )

    # 可选：可视化扩散过程 / Optional: visualize diffusion process
    if args.visualize_process:
        visualize_diffusion_process(
            diffusion,
            num_steps=args.num_process_steps,
            output_path='./demo/diffusion_process'
        )

    print("\n完成！/ Done!")


if __name__ == '__main__':
    main()
