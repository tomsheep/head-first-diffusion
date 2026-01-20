"""
生成合成数据用于扩散模型学习
Generate synthetic data for diffusion model learning

这个脚本生成简单的合成图像数据，用于学习扩散模型的基础概念。
This script generates simple synthetic image data for learning the basics of diffusion models.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm


def generate_shapes(num_images, image_size, output_dir, shape_type='circles'):
    """
    生成简单的几何形状图像
    Generate simple geometric shape images

    Args:
        num_images: 生成的图像数量 / Number of images to generate
        image_size: 图像尺寸 / Image size
        output_dir: 输出目录 / Output directory
        shape_type: 形状类型 / Shape type (circles, squares, mixed)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} {shape_type} images of size {image_size}x{image_size}...")
    print(f"输出目录: {output_path}")

    for i in tqdm(range(num_images)):
        # 创建空白图像 / Create blank image
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        img.fill(255)  # 白色背景 / White background

        # 随机选择颜色 / Randomly select color
        color = np.random.randint(50, 255, size=3)

        # 随机选择形状 / Randomly select shape
        if shape_type == 'mixed':
            current_shape = np.random.choice(['circle', 'square'])
        else:
            current_shape = shape_type[:-1]  # Remove 's' from 'circles' or 'squares'

        # 随机位置和大小 / Random position and size
        center_x = np.random.randint(image_size // 4, 3 * image_size // 4)
        center_y = np.random.randint(image_size // 4, 3 * image_size // 4)
        size = np.random.randint(image_size // 8, image_size // 3)

        # 绘制形状 / Draw shape
        if current_shape == 'circle':
            # 绘制圆形 / Draw circle
            y, x = np.ogrid[:image_size, :image_size]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= size ** 2
            img[mask] = color
        else:
            # 绘制正方形 / Draw square
            x1 = max(0, center_x - size)
            x2 = min(image_size, center_x + size)
            y1 = max(0, center_y - size)
            y2 = min(image_size, center_y + size)
            img[y1:y2, x1:x2] = color

        # 保存图像 / Save image
        pil_img = Image.fromarray(img)
        pil_img.save(output_path / f"{i:06d}.png")

    print(f"完成！生成了 {num_images} 张图像")
    print(f"Done! Generated {num_images} images")


def generate_gradient_images(num_images, image_size, output_dir):
    """
    生成渐变色图像
    Generate gradient color images

    Args:
        num_images: 生成的图像数量 / Number of images to generate
        image_size: 图像尺寸 / Image size
        output_dir: 输出目录 / Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} gradient images of size {image_size}x{image_size}...")
    print(f"输出目录: {output_path}")

    for i in tqdm(range(num_images)):
        # 随机选择两个颜色 / Randomly select two colors
        color1 = np.random.randint(50, 255, size=3)
        color2 = np.random.randint(50, 255, size=3)

        # 创建渐变 / Create gradient
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # 水平或垂直渐变 / Horizontal or vertical gradient
        if np.random.rand() > 0.5:
            # 水平渐变 / Horizontal gradient
            for x in range(image_size):
                ratio = x / image_size
                img[:, x] = color1 * (1 - ratio) + color2 * ratio
        else:
            # 垂直渐变 / Vertical gradient
            for y in range(image_size):
                ratio = y / image_size
                img[y, :] = color1 * (1 - ratio) + color2 * ratio

        # 保存图像 / Save image
        pil_img = Image.fromarray(img)
        pil_img.save(output_path / f"{i:06d}.png")

    print(f"完成！生成了 {num_images} 张渐变图像")
    print(f"Done! Generated {num_images} gradient images")


def generate_noise_patterns(num_images, image_size, output_dir):
    """
    生成噪声模式图像
    Generate noise pattern images

    Args:
        num_images: 生成的图像数量 / Number of images to generate
        image_size: 图像尺寸 / Image size
        output_dir: 输出目录 / Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} noise pattern images of size {image_size}x{image_size}...")
    print(f"输出目录: {output_path}")

    for i in tqdm(range(num_images)):
        # 随机选择噪声类型 / Randomly select noise type
        noise_type = np.random.choice(['gaussian', 'perlin-like', 'checkerboard'])

        if noise_type == 'gaussian':
            # 高斯噪声 / Gaussian noise
            img = np.random.randint(50, 255, (image_size, image_size, 3), dtype=np.uint8)

        elif noise_type == 'perlin-like':
            # 类似Perlin噪声（简化版） / Perlin-like noise (simplified)
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            for c in range(3):
                # 使用多个频率的正弦波叠加 / Superimpose multiple sine waves
                noise = np.zeros((image_size, image_size))
                for freq in [1, 2, 4, 8]:
                    phase = np.random.rand() * 2 * np.pi
                    noise += np.sin(np.linspace(0, freq * 2 * np.pi, image_size)[:, None] +
                                  np.linspace(0, freq * 2 * np.pi, image_size)[None, :] + phase)
                noise = (noise - noise.min()) / (noise.max() - noise.min())
                img[:, :, c] = (noise * 200 + 55).astype(np.uint8)

        else:  # checkerboard
            # 棋盘格模式 / Checkerboard pattern
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            color1 = np.random.randint(50, 200, size=3)
            color2 = np.random.randint(50, 200, size=3)
            block_size = np.random.randint(4, 16)
            for y in range(image_size):
                for x in range(image_size):
                    if ((x // block_size) + (y // block_size)) % 2 == 0:
                        img[y, x] = color1
                    else:
                        img[y, x] = color2

        # 保存图像 / Save image
        pil_img = Image.fromarray(img)
        pil_img.save(output_path / f"{i:06d}.png")

    print(f"完成！生成了 {num_images} 张噪声模式图像")
    print(f"Done! Generated {num_images} noise pattern images")


def main():
    parser = argparse.ArgumentParser(description='生成合成数据用于扩散模型学习 / Generate synthetic data for diffusion model learning')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='生成的图像数量 / Number of images to generate')
    parser.add_argument('--image_size', type=int, default=64,
                        help='图像尺寸 / Image size')
    parser.add_argument('--output_dir', type=str, default='./demo/data',
                        help='输出目录 / Output directory')
    parser.add_argument('--data_type', type=str, default='circles',
                        choices=['circles', 'squares', 'mixed', 'gradient', 'noise'],
                        help='数据类型 / Data type')

    args = parser.parse_args()

    if args.data_type in ['circles', 'squares', 'mixed']:
        generate_shapes(args.num_images, args.image_size, args.output_dir, args.data_type)
    elif args.data_type == 'gradient':
        generate_gradient_images(args.num_images, args.image_size, args.output_dir)
    elif args.data_type == 'noise':
        generate_noise_patterns(args.num_images, args.image_size, args.output_dir)


if __name__ == '__main__':
    main()
