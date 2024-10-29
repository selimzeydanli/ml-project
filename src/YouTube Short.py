import os
import subprocess


def create_video_short(input_filename, output_filename, start_time, end_time):
    """
    Extract a short clip from an MP4 video file.

    Parameters:
    input_filename (str): Filename of the input MP4 video file
    output_filename (str): Filename of the output short video file
    start_time (float): Start time of the clip in seconds
    end_time (float): End time of the clip in seconds
    """
    # Construct the full file paths
    input_file = os.path.join(r'C:\Users\Selim\Desktop\Youtube\Video Contents\Turkce\1 - Aktif Dinleme\Video and Short',
                              input_filename)
    output_file = os.path.join(
        r'C:\Users\Selim\Desktop\Youtube\Video Contents\Turkce\1 - Aktif Dinleme\Video and Short', output_filename)

    # Locate the ffmpeg executable
    ffmpeg_executable = find_ffmpeg_executable()

    # Construct the ffmpeg command
    ffmpeg_cmd = [
        ffmpeg_executable,
        '-i', input_file,
        '-ss', str(start_time),
        '-t', str(end_time - start_time),
        '-c', 'copy',
        output_file
    ]

    # Execute the ffmpeg command
    subprocess.run(ffmpeg_cmd, check=True)

    print(f"Short video created: {output_file}")


def find_ffmpeg_executable():
    """
    Locate the ffmpeg executable on the system.

    Returns:
    str: The full path to the ffmpeg executable.
    """
    ffmpeg_path = r"C:\Users\Selim\Desktop\ffmpeg\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"
    if os.path.exists(ffmpeg_path):
        return ffmpeg_path
    else:
        raise FileNotFoundError("ffmpeg executable not found at specified path.")


# Example usage
import os

print(os.getcwd())
create_video_short('Aktif Dinleme Video.mp4', 'Aktif Dinleme Short.mp4', 30.0, 45.0)