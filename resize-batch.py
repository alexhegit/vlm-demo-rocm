import os
import argparse
from PIL import Image

def resize_images(input_dir, output_dir, target_size=(448, 364)):
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    total_files = 0
    resized_files = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            total_files += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with Image.open(input_path) as img:
                    # 调整大小
                    img = img.resize(target_size, Image.LANCZOS)
                    # 保存为 JPG
                    img.save(output_path, "JPEG")
                    resized_files += 1
                    print(f"已调整大小: {filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    print(f"\n处理完成。")
    print(f"总共处理了 {total_files} 张图片")
    print(f"成功调整大小并保存了 {resized_files} 张图片")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将指定目录中的所有 JPG 图像调整为 484x484 大小并保存到输出目录")
    parser.add_argument("--input", type=str, required=True, help="输入目录（包含 JPG 图像）")
    parser.add_argument("--output", type=str, required=True, help="输出目录（保存调整大小后的 JPG 图像）")
    args = parser.parse_args()

    resize_images(args.input, args.output)
