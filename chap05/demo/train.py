"""
扩散模型训练脚本
Diffusion model training script

这个脚本用于训练扩散模型，支持自定义配置和多种训练选项。
This script is used to train diffusion models with custom configurations and various training options.
"""

import torch
from pathlib import Path
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import argparse


def train_diffusion_model(
    data_folder,
    image_size=64,
    num_timesteps=1000,
    sampling_timesteps=250,
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=10000,
    results_folder='./demo/results',
    model_dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False,
    amp=True,
    ema_decay=0.995,
    gradient_accumulate_every=1,
    save_and_sample_every=1000,
    num_samples=25,
    calculate_fid=False,
    beta_schedule='sigmoid'
):
    """
    训练扩散模型
    Train diffusion model

    Args:
        data_folder: 数据文件夹路径 / Path to data folder
        image_size: 图像尺寸 / Image size
        num_timesteps: 扩散步数 / Number of diffusion timesteps
        sampling_timesteps: 采样步数 / Number of sampling timesteps (for DDIM)
        train_batch_size: 训练批次大小 / Training batch size
        train_lr: 学习率 / Learning rate
        train_num_steps: 训练步数 / Number of training steps
        results_folder: 结果保存文件夹 / Results folder
        model_dim: 模型基础维度 / Model base dimension
        dim_mults: 维度倍数 / Dimension multipliers
        flash_attn: 是否使用Flash Attention / Whether to use Flash Attention
        amp: 是否使用混合精度训练 / Whether to use mixed precision training
        ema_decay: EMA衰减率 / EMA decay rate
        gradient_accumulate_every: 梯度累积步数 / Gradient accumulation steps
        save_and_sample_every: 保存和采样间隔 / Save and sample interval
        num_samples: 每次采样的样本数 / Number of samples per save
        calculate_fid: 是否计算FID分数 / Whether to calculate FID score
        beta_schedule: 噪声调度类型 / Noise schedule type
    """
    print("=" * 60)
    print("扩散模型训练 / Diffusion Model Training")
    print("=" * 60)
    print(f"数据文件夹 / Data folder: {data_folder}")
    print(f"图像尺寸 / Image size: {image_size}")
    print(f"训练步数 / Training steps: {train_num_steps}")
    print(f"批次大小 / Batch size: {train_batch_size}")
    print(f"学习率 / Learning rate: {train_lr}")
    print(f"扩散步数 / Diffusion timesteps: {num_timesteps}")
    print(f"采样步数 / Sampling timesteps: {sampling_timesteps}")
    print(f"噪声调度 / Beta schedule: {beta_schedule}")
    print(f"混合精度 / Mixed precision: {amp}")
    print("=" * 60)

    # 检查数据文件夹 / Check data folder
    data_path = Path(data_folder)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件夹不存在 / Data folder not found: {data_folder}")

    # 创建UNet模型 / Create UNet model
    model = Unet(
        dim=model_dim,
        dim_mults=dim_mults,
        channels=3,
        flash_attn=flash_attn
    )

    # 创建扩散过程 / Create diffusion process
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=num_timesteps,
        sampling_timesteps=sampling_timesteps,
        beta_schedule=beta_schedule,
        objective='pred_v'  # 使用v-parameterization，通常效果更好
    )

    # 创建训练器 / Create trainer
    trainer = Trainer(
        diffusion,
        data_folder,
        train_batch_size=train_batch_size,
        train_lr=train_lr,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=gradient_accumulate_every,
        ema_decay=ema_decay,
        amp=amp,
        results_folder=results_folder,
        save_and_sample_every=save_and_sample_every,
        num_samples=num_samples,
        calculate_fid=calculate_fid
    )

    # 开始训练 / Start training
    print("\n开始训练 / Starting training...\n")
    trainer.train()

    print("\n训练完成！/ Training completed!")
    print(f"模型和结果保存在 / Models and results saved to: {results_folder}")


def main():
    parser = argparse.ArgumentParser(description='训练扩散模型 / Train diffusion model')

    # 数据相关参数 / Data-related parameters
    parser.add_argument('--data_folder', type=str, default='./demo/data',
                        help='数据文件夹路径 / Path to data folder')
    parser.add_argument('--image_size', type=int, default=64,
                        help='图像尺寸 / Image size')

    # 模型相关参数 / Model-related parameters
    parser.add_argument('--model_dim', type=int, default=64,
                        help='模型基础维度 / Model base dimension')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='维度倍数 / Dimension multipliers')
    parser.add_argument('--flash_attn', action='store_true',
                        help='使用Flash Attention / Use Flash Attention')

    # 扩散相关参数 / Diffusion-related parameters
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='扩散步数 / Number of diffusion timesteps')
    parser.add_argument('--sampling_timesteps', type=int, default=250,
                        help='采样步数 / Number of sampling timesteps')
    parser.add_argument('--beta_schedule', type=str, default='sigmoid',
                        choices=['linear', 'cosine', 'sigmoid'],
                        help='噪声调度类型 / Noise schedule type')

    # 训练相关参数 / Training-related parameters
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='训练批次大小 / Training batch size')
    parser.add_argument('--train_lr', type=float, default=8e-5,
                        help='学习率 / Learning rate')
    parser.add_argument('--train_num_steps', type=int, default=10000,
                        help='训练步数 / Number of training steps')
    parser.add_argument('--gradient_accumulate_every', type=int, default=1,
                        help='梯度累积步数 / Gradient accumulation steps')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='EMA衰减率 / EMA decay rate')

    # 保存和采样相关参数 / Save and sample-related parameters
    parser.add_argument('--results_folder', type=str, default='./demo/results',
                        help='结果保存文件夹 / Results folder')
    parser.add_argument('--save_and_sample_every', type=int, default=1000,
                        help='保存和采样间隔 / Save and sample interval')
    parser.add_argument('--num_samples', type=int, default=25,
                        help='每次采样的样本数 / Number of samples per save')

    # 其他参数 / Other parameters
    parser.add_argument('--amp', action='store_true', default=True,
                        help='使用混合精度训练 / Use mixed precision training')
    parser.add_argument('--calculate_fid', action='store_true',
                        help='计算FID分数 / Calculate FID score')

    args = parser.parse_args()

    # 训练模型 / Train model
    train_diffusion_model(
        data_folder=args.data_folder,
        image_size=args.image_size,
        num_timesteps=args.num_timesteps,
        sampling_timesteps=args.sampling_timesteps,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        results_folder=args.results_folder,
        model_dim=args.model_dim,
        dim_mults=tuple(args.dim_mults),
        flash_attn=args.flash_attn,
        amp=args.amp,
        ema_decay=args.ema_decay,
        gradient_accumulate_every=args.gradient_accumulate_every,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        calculate_fid=args.calculate_fid,
        beta_schedule=args.beta_schedule
    )


if __name__ == '__main__':
    main()
