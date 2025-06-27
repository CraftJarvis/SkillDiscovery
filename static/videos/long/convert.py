import os
import av

def convert_to_avc1(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'converted')
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.mp4')

            print(f"Converting {filename} to avc1 (H.264)...")
            try:
                # 使用 PyAV 打开视频
                container = av.open(input_path)
                output_container = av.open(output_path, mode='w')

                # 创建 H.264 编码器
                stream = output_container.add_stream('h264', rate=container.streams.video[0].average_rate)
                stream.width = container.streams.video[0].width
                stream.height = container.streams.video[0].height
                stream.pix_fmt = 'yuv420p'  # H.264 常用的像素格式

                # 转码视频
                for frame in container.decode(video=0):
                    frame = frame.reformat(stream.width, stream.height, 'yuv420p')
                    for packet in stream.encode(frame):
                        output_container.mux(packet)

                # 刷新编码缓冲区
                for packet in stream.encode():
                    output_container.mux(packet)

                print(f"Saved to {output_path}")
                container.close()
                output_container.close()
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

    print("Conversion completed.")

# 使用示例
convert_to_avc1('./', './converted')