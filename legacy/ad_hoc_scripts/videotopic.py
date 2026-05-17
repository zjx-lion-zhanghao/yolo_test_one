import cv2
import os
import argparse


def video_to_frames(video_path, output_dir, frame_interval=1, start_time=0, end_time=None, prefix='frame'):
    """
    将视频按指定间隔抽帧并保存为图片

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_interval: 每隔多少帧抽取一帧 (默认1，即每帧都抽取)
        start_time: 从视频的哪个时间开始(秒) (默认0)
        end_time: 到视频的哪个时间结束(秒) (默认None表示到结尾)
        prefix: 输出图片的前缀名 (默认'frame')
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"视频信息: {fps} FPS, 总帧数: {total_frames}, 时长: {duration:.2f}秒")

    # 设置起始和结束帧
    start_frame = int(start_time * fps)
    if end_time is not None:
        end_frame = min(int(end_time * fps), total_frames)
    else:
        end_frame = total_frames

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    saved_count = 0

    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔frame_interval帧保存一次
        if frame_count % frame_interval == 0:
            # 构造输出文件名
            frame_time = frame_count / fps
            output_filename = f"{prefix}_{frame_count:06d}_{frame_time:.2f}s.jpg"
            output_path = os.path.join(output_dir, output_filename)

            # 保存帧
            cv2.imwrite(output_path, frame)
            saved_count += 1

            print(f"已保存: {output_filename}")

        frame_count += 1

    cap.release()
    print(f"抽帧完成! 共保存了 {saved_count} 张图片到 {output_dir}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='视频抽帧工具')
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--interval', type=int, default=1, help='每隔多少帧抽取一帧 (默认1)')
    parser.add_argument('--start', type=float, default=0, help='起始时间(秒) (默认0)')
    parser.add_argument('--end', type=float, help='结束时间(秒) (默认到结尾)')
    parser.add_argument('--prefix', default='frame', help='输出图片前缀 (默认"frame")')

    args = parser.parse_args()

    # 调用抽帧函数
    video_to_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        frame_interval=args.interval,
        start_time=args.start,
        end_time=args.end,
        prefix=args.prefix
    )