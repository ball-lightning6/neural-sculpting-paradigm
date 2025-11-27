import os
import subprocess

# --- 配置区 ---

# 1. 存放 eval image 的目录
# IMAGE_DIR = r"F:\data\shuiliu\checkpoints_arc_shuiliu"
# IMAGE_DIR = r"F:\data\tess\checkpoints_arc_tessellation_video"
# IMAGE_DIR = r"F:\data\ca\checkpoints_qwen2_text2image_ca_video\eval_predictions"
IMAGE_DIR = r"F:\data\cube\checkpoints_cube_text2image_video\eval_predictions"
# 2. 视频输出目录 (如果不存在，脚本会自动创建)
# OUTPUT_DIR = r"F:\data\shuiliu\videos_output"
OUTPUT_DIR = r"F:\data\cube\videos_output"

# 3. 样本数量 (0 到 15，共 16 个)
NUM_SAMPLES = 32#16

# 4. 起始和结束的 step，以及步长
START_STEP = 20
END_STEP = 99080#1200#59800#19200
STEP_INTERVAL = 20

# 5. FFmpeg 相关配置
FFMPEG_PATH = "ffmpeg"  # 如果 ffmpeg 在系统路径中，直接写 "ffmpeg" 即可。
# 否则，请提供完整路径，例如 r"C:\ffmpeg\bin\ffmpeg.exe"

VIDEO_FRAMERATE = 30#6#30#10  # 视频帧率 (每秒播放多少张图片)，可以调整以改变播放速度
VIDEO_CODEC = "libx264"  # 视频编码器，libx264 兼容性最好
PIXEL_FORMAT = "yuv420p"  # 像素格式，确保在各种播放器上都能正常显示颜色
CRF = 23  # 视频质量参数 (Constant Rate Factor)，数值越小质量越高，文件越大。18-28 是一个合理的范围。


# --- 脚本主逻辑 ---

def create_video_for_sample(sample_idx):
    """为指定的样本编号生成演化视频"""

    print(f"--- 开始处理样本 {sample_idx} ---")

    # 1. 收集该样本的所有图片文件路径
    image_files = []
    for step in range(START_STEP, END_STEP + 1, STEP_INTERVAL):
        filename = f"step_{step}_sample_{sample_idx}.png"
        filepath = os.path.join(IMAGE_DIR, filename)

        if os.path.exists(filepath):
            image_files.append(filepath)
        else:
            print(f"警告：找不到文件 {filepath}，将跳过此帧。")

    if not image_files:
        print(f"错误：样本 {sample_idx} 没有任何有效的图片文件，无法生成视频。")
        return

    print(f"找到 {len(image_files)} 张图片，将用于生成视频。")

    # 2. 创建一个临时的文件列表供 ffmpeg 读取
    list_filename = f"temp_filelist_sample_{sample_idx}.txt"
    with open(list_filename, 'w') as f:
        for filepath in image_files:
            # FFmpeg 的 filelist 格式需要对特殊字符进行转义，并使用单引号
            # Python 的 os.path.join 会自动处理路径分隔符，这里我们做一些处理确保兼容性
            escaped_path = filepath.replace('\\', '/')
            f.write(f"file '{escaped_path}'\n")

    # 3. 构建并执行 FFmpeg 命令
    output_video_path = os.path.join(OUTPUT_DIR, f"evolution_sample_{sample_idx}.mp4")

    command = [
        FFMPEG_PATH,
        '-y',  # 如果输出文件已存在，直接覆盖
        '-f', 'concat',
        '-safe', '0',
        '-r', str(VIDEO_FRAMERATE),  # 设置输入帧率
        '-i', list_filename,
        '-c:v', VIDEO_CODEC,
        '-pix_fmt', PIXEL_FORMAT,
        '-crf', str(CRF),
        '-framerate', str(VIDEO_FRAMERATE),  # 设置输出帧率
        output_video_path
    ]

    print(f"正在执行 FFmpeg 命令...")
    # print(" ".join(command)) # 如果需要调试，可以取消这行注释来查看完整命令

    try:
        # 使用 subprocess.run 来执行命令，并捕获输出
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"成功为样本 {sample_idx} 生成视频: {output_video_path}")
        # print("FFmpeg 输出:\n", result.stdout) # 调试时可以查看 ffmpeg 的详细输出
    except FileNotFoundError:
        print(f"错误：找不到 FFmpeg 程序。请确保 '{FFMPEG_PATH}' 是正确的路径或已添加到系统环境变量中。")
    except subprocess.CalledProcessError as e:
        print(f"错误：FFmpeg 在处理样本 {sample_idx} 时返回错误。")
        print(f"返回码: {e.returncode}")
        print(f"FFmpeg 的错误输出:\n{e.stderr}")
    finally:
        # 4. 清理临时文件
        if os.path.exists(list_filename):
            os.remove(list_filename)
            print(f"已删除临时文件: {list_filename}")


if __name__=="__main__":
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    # 为所有样本生成视频
    for i in range(NUM_SAMPLES):
        create_video_for_sample(i)
        print("\n" + "=" * 50 + "\n")

    print("--- 所有任务已完成 ---")