#!/usr/bin/env python3
import ffmpeg
import argparse
import json
import re
import os
import subprocess
import string
import whisperx
import torch

# Function to load a list of swear words from a file
def load_swear_words(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return set(words)

# Function to get video name and extension from file path
def get_video_name(video_file):
    video_name, video_ext = os.path.splitext(os.path.basename(video_file))
    input_video = f"input_{video_name}{video_ext}"
    out_video_name = video_name
    out_video_ext = video_ext
    os.rename(video_file, input_video)
    in_video_name, in_video_ext = os.path.splitext(os.path.basename(input_video))
    return video_name, video_ext, input_video, in_video_name, in_video_ext, out_video_name, out_video_ext

# Function to list audio streams in the input video
def list_audio_streams(input_video):
    probe = ffmpeg.probe(input_video)
    return [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']

# Function to get information about audio streams
def get_audio_streams_info(input_video):
    audio_streams = list_audio_streams(input_video)
    audio_streams_info = []

    for stream in audio_streams:
        language = stream.get('tags', {}).get('language', 'Unknown')
        stream_info = {
            'codec_name': stream['codec_name'],
            'channels': stream['channels'],
            'language': language
        }
        audio_streams_info.append(stream_info)
    return audio_streams_info

# Function to prompt user to select an audio stream
def prompt_user_to_select_stream(audio_streams):
    num_streams = len(audio_streams)
    if num_streams == 1:
        selected_index = 0
        return selected_index
    else:
        print("Available audio streams:")
        for idx, stream in enumerate(audio_streams, start=1):
            language = stream.get('tags', {}).get('language', 'Unknown')
            print(f"{idx}: {stream['codec_long_name']} - {stream['channels']} channels - {language}")
        while True:
            try:
                selected_index = int(input("Select the audio stream index to process: "))
                if 1 <= selected_index <= len(audio_streams):
                    return selected_index - 1
                else:
                    print("Invalid stream index. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

# Function to match channel layout to a list of channel abbreviations
def match_channel_layout(layout):
    layouts = {
    "mono": ["FC"],
    "stereo": ["FL", "FR"],
    "2.1": ["FL", "FR", "LFE"],
    "3.0": ["FL", "FR", "FC"],
    "3.0(back)": ["FL", "FR", "BC"],
    "4.0": ["FL", "FR", "FC", "BC"],
    "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"],
    "3.1": ["FL", "FR", "FC", "LFE"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "4.1": ["FL", "FR", "FC", "LFE", "BC"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.0": ["FL", "FR", "FC", "BC", "SL", "SR"],
    "6.0(front)": ["FL", "FR", "FLC", "FRC", "SL", "SR"],
    "hexagonal": ["FL", "FR", "FC", "BL", "BR", "BC"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "6.1(back)": ["FL", "FR", "FC", "LFE", "BL", "BR", "BC"],
    "6.1(front)": ["FL", "FR", "LFE", "FLC", "FRC", "SL", "SR"],
    "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.0(front)": ["FL", "FR", "FC", "FLC", "FRC", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
    "7.1(wide-side)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
    "octagonal": ["FL", "FR", "FC", "BL", "BR", "BC", "SL", "SR"],
    "hexadecagonal": ["FL", "FR", "FC", "BL", "BR", "BC", "SL", "SR", "TFL", "TFC", "TFR", "TBL", "TBC", "TBR", "WL", "WR"],
    "downmix": ["DL", "DR"],
    "22.2": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC", "BC", "SL", "SR", "TC", "TFL", "TFC", "TFR", "TBL", "TBC", "TBR", "LFE2", "TSL", "TSR", "BFC", "BFL", "BFR"]
    }
    if layout in layouts:
        return layouts[layout]
    else:
        return None

# Function to extract individual audio channels from the selected stream
def extract_channels(in_video_name, input_video, selected_stream_index, output_prefix=None, output_files=None):
    output_prefix = in_video_name
    audio_streams = list_audio_streams(input_video)
    selected_stream = audio_streams[selected_stream_index]
    # Get audio stream individual values
    layout = selected_stream['channel_layout']
    num_channels = selected_stream['channels']
    channels = match_channel_layout(layout)
    codec_name = selected_stream['codec_name']
    audio_start_time = selected_stream['start_time']
    audio_bit_rate = selected_stream['bit_rate']

    if not channels:
        print(f"Unknown or unsupported channel layout: {layout}")
        return None, []

    try:
        if layout in ["mono", "stereo", "2.1", "3.0", "3.0(back)"]:
            # For mono or stereo, copy the entire stream without splitting
            output_file = f"{output_prefix}_{layout}.{codec_name}"
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", input_video,
                "-map", f"0:a:{selected_stream_index}",
                "-c:a", "copy",
                "-y",
                output_file
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            channel_file = output_file
            print(f"Extracted {layout} audio stream to {output_file}")
            return layout, num_channels, channels, codec_name, audio_start_time, audio_bit_rate, channel_file, []
            
        else:
            if output_files is None:
                output_files = [f"{output_prefix}_{channel}.{codec_name}" for channel in channels]
                print(f"Output files: {', '.join(output_files)}")

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", input_video,
                "-filter_complex", f"[0:a:{selected_stream_index}]channelsplit=channel_layout={layout}[{']['.join(channels)}]"
            ]
            for i, channel in enumerate(channels):
                ffmpeg_cmd.extend(["-map", f"[{channel}]", "-b:a", audio_bit_rate,  "-ac", "1", output_files[i]])
            # Debug - print(f"ffmpeg command: {ffmpeg_cmd}") 
            subprocess.run(ffmpeg_cmd, check=True)
            
            for file in output_files:
                if "_FC" in os.path.basename(file):
                    channel_file = file
                    break

            if channel_file:
                print(f"Channel file: {channel_file}")
            else:
                print("Center channel file not found - no channel file name contains _FC")
                exit(1)
            print(f"Extracted channels to {', '.join(output_files)}")
            return layout, num_channels, channels, codec_name, audio_start_time, audio_bit_rate, channel_file, output_files

    except subprocess.CalledProcessError as e:
        print(f"Error extracting channels: {e}")
        return

# Function to call whisperx to generate transcript and word timestamps
def run_whisperx(channel_file, in_video_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    language = "en"
    batch_size = 16
    compute_type = "float16"
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    
    audio = whisperx.load_audio(channel_file)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Save the transcript and word timestamps to a JSON file
    json_file = f"{in_video_name}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=4)

    # Save the transcript and word timestamps to an SRT file
    srt_file = f"{in_video_name}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    return in_video_name

# Function to format timestamp to SRT format
def format_timestamp(seconds):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Function to get audio properties
def get_audio_properties(channel_file):
    result = subprocess.run(
        ['ffprobe', '-loglevel', '0', '-print_format', 'json', '-show_format', '-show_streams', channel_file],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    sample_rate = int(info['streams'][0]['sample_rate'])
    codec_name = info['streams'][0]['codec_name']
    channels = int(info['streams'][0]['channels'])
    duration = float(info['streams'][0]['duration'])
    format_name = info['format']['format_name']
    format_bitrate = int(info['format']['bit_rate'])
    return sample_rate, codec_name, channels, duration, format_name, format_bitrate

# Function to load an SRT file
def load_srt(in_video_name):
    input_srt = (f"{in_video_name}.srt")
    with open(input_srt, 'r', encoding='utf-8') as file:
        return file.readlines()

# Function to replace filter words in SRT file
def replace_words_in_srt(lines, swear_words):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in swear_words) + r')\b', re.IGNORECASE)
    modified_lines = []
    for line in lines:
        if not line.strip().isdigit() and '-->' not in line:
            line = pattern.sub('***', line)
        modified_lines.append(line)
    return modified_lines

# Function to save the redacted SRT file
def save_srt(out_video_name, lines):
    output_srt = (f"{out_video_name}_modified.srt")
    print (f"{output_srt}")
    with open(output_srt, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    return output_srt

# Function to collect timestamps of all filter words from JSON file
def identify_swearing_timestamps(in_video_name, swear_words):
    timestamp_file = (f"{in_video_name}.json")
    with open(timestamp_file) as f:
        timestamps_data = json.load(f)
    swear_timestamps = []
    for segment in timestamps_data["segments"]:
        for word in segment["words"]:
            word_without_punct = ''.join(char for char in word["word"].lower() if char not in set(string.punctuation))            
            if word_without_punct in (sw.lower().strip(string.punctuation) for sw in swear_words):
                start_time = word["start"] - 0.02
                end_time = word["end"] + 0.02
                swear_timestamps.append({"start": start_time, "end": end_time})
    return swear_timestamps

# Function to get the inverse of the filter words timestamps
def get_non_swearing_intervals(swear_timestamps, duration):
    non_swearing_intervals = []
    if not swear_timestamps:
        return [{'start': 0, 'end': duration}]
    if swear_timestamps[0]['start'] > 0:
        non_swearing_intervals.append({'start': 0, 'end': swear_timestamps[0]['start']})
    for i in range(len(swear_timestamps) - 1):
        non_swearing_intervals.append({                                                                                                                                                                                                                                                                                                                                                 
            'start': swear_timestamps[i]['end'],
            'end': swear_timestamps[i + 1]['start']
        })
    if swear_timestamps and isinstance(duration, str):
        duration = float(duration)
        print(duration)
        print(swear_timestamps[-1]['end'])
    if swear_timestamps[-1]['end'] < duration:
        non_swearing_intervals.append({'start': swear_timestamps[-1]['end'], 'end': duration})
    return non_swearing_intervals

# Function to generate the ffmpeg complex filter, to redact filter words
def create_ffmpeg_filter(swear_timestamps, non_swearing_intervals):
    dipped_vocals_conditions = '+'.join([f"between(t,{b['start']},{b['end']})" for b in swear_timestamps])
    dipped_vocals_filter = f"[0]volume=0:enable='{dipped_vocals_conditions}'[main]"

    no_bleeps_conditions = '+'.join([f"between(t,{segment['start']},{segment['end']})" for segment in non_swearing_intervals])
    dipped_bleep_filter = f"sine=f=800,pan=stereo|FL=c0|FR=c0,volume=0:enable='{no_bleeps_conditions}'[beep]"

    amix_filter = "[main][beep]amix=inputs=2:duration=first"

    filter_complex = ';'.join([
        dipped_vocals_filter,
        dipped_bleep_filter,
        amix_filter
    ])
    return filter_complex

# Function to apply complext filter to voice audio stream
def apply_complex_filter(channel_file, filter_complex, format_bitrate=None):
    bitrate = format_bitrate
    channel_file_name, channel_file_ext = os.path.splitext(os.path.basename(channel_file))
    output_file = f"{channel_file_name}_modified{channel_file_ext}"
    ffmpeg_cmd = [
        "ffmpeg", "-i", channel_file,
        "-y", "-filter_complex", filter_complex,
        "-bitexact", "-ac", "1",  
    ]
    if bitrate:
        ffmpeg_cmd.extend(["-b:a", str(bitrate)])
    ffmpeg_cmd.append(output_file)
    subprocess.run(ffmpeg_cmd, check=True)
    modified_fc_channel_file = output_file
    modified_audio_stream = modified_fc_channel_file
    return modified_fc_channel_file, modified_audio_stream

# Function to combine modified center channel audio file with the remaining channel audio files into a multi-channel audio stream
def combine_multi_channel_audio(layout, out_video_name, audio_bit_rate, output_files, modified_fc_channel_file):
    modified_audio_stream = f"{out_video_name}_modified.ac3"
    center_channel_file = next((f for f in output_files if "_FC" in os.path.basename(f)), None)

    if center_channel_file:
        channel_mapping = match_channel_layout(layout)
        if channel_mapping:
            channel_indices = "|".join([str(i) for i in range(len(channel_mapping))])
            channel_names = "|".join(channel_mapping)
            channel_mapping_str = f"{channel_indices}:{channel_names}"
            ffmpeg_cmd = [
                "ffmpeg", "-y"
            ]

            for file in output_files:
                if file == center_channel_file:
                    ffmpeg_cmd.extend(["-ac", "1", "-i", modified_fc_channel_file])
                else:
                    ffmpeg_cmd.extend(["-ac", "1", "-i", file])

            input_stream_mapping = []
            for i in range(len(output_files)):
                input_stream_mapping.append(f"[{i}:a]")

            amerge_filter = "".join(input_stream_mapping) + f"amerge=inputs={len(output_files)},channelmap={channel_mapping_str}"

            ffmpeg_cmd.extend([
                "-filter_complex", amerge_filter,
                "-b:a", audio_bit_rate,
                "-ac", str(len(output_files)),
                modified_audio_stream
            ])
            
            subprocess.run(ffmpeg_cmd, check=True)
            return modified_audio_stream
        else:
            print("Unsupported channel layout")
            return None
    else:
        print("Channel file not found")
        return None    

# Function to add the updated audio and subtitile streams to the container file
def replace_channel(out_video_name, out_video_ext, input_video, modified_audio_stream, audio_start_time, output_srt):
    final_video = f"{out_video_name}.mkv"
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ss", audio_start_time,
        "-i", modified_audio_stream,
        "-i", output_srt,
        "-map", "0:v",
        "-map", "1:a",
        "-map", "0:a", 
        "-map", "2:s", 
        "-c:v", "copy",  
        "-c:a", "copy",
        "-c:s", "srt", 
        "-disposition:a:0", "default", 
        "-disposition:a:1", "0",
        "-disposition:s:0", "default", 
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:0", "title=Family", 
        "-metadata:s:a:1", "language=eng", 
        "-metadata:s:a:1", "title=Original",
        "-metadata:s:s:0", "language=eng", 
        "-metadata:s:s:0", "title=English", 
        "-y",
        final_video
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return final_video

# Function to clean up temporary files
def cleanup_temp_files(input_video, in_video_name, output_srt, channel_file, modified_fc_channel_file, modified_audio_stream, output_files=None):

    # List of files and directories to remove
    temp_files = [
        f"{in_video_name}.json",
        #f"{in_video_name}.srt",
        input_video,
        #output_srt,
        channel_file,
        modified_fc_channel_file,
        modified_audio_stream
    ]

    # Add channel files and directories from output_files if exist
    if output_files:
        temp_files.extend(output_files)

    # Delete temp files
    for item in temp_files:
        try:
            if os.path.isfile(item):
                os.remove(item)
            elif os.path.isdir(item):
                shutil.rmtree(item)
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")

# Main functiom
def main(video_file):
    swear_words_file = "filter_words.txt"
    swear_words = load_swear_words(swear_words_file)
    video_name, video_ext, input_video, in_video_name, in_video_ext, out_video_name, out_video_ext = get_video_name(video_file)
    audio_streams = list_audio_streams(input_video)
    audio_streams_info = get_audio_streams_info(input_video)
    selected_stream_index = prompt_user_to_select_stream(audio_streams)
    
    layout, num_channels, channels, codec_name, audio_start_time, audio_bit_rate, channel_file, output_files = extract_channels(in_video_name, input_video, selected_stream_index, 0)
    sample_rate, codec_name, channels, duration, format_name, format_bitrate = get_audio_properties(channel_file)
    run_whisperx(channel_file, in_video_name) # Debug
    swear_timestamps = identify_swearing_timestamps(in_video_name, swear_words)
    non_swearing_intervals = get_non_swearing_intervals(swear_timestamps, duration)
    filter_complex = create_ffmpeg_filter(swear_timestamps, non_swearing_intervals)
    print (f"Complex filter: {filter_complex}")
    modified_fc_channel_file, modified_audio_stream = apply_complex_filter(channel_file, filter_complex, format_bitrate)
    lines = load_srt(in_video_name)
    modified_lines = replace_words_in_srt(lines, swear_words)
    output_srt = save_srt(out_video_name, modified_lines)
    print(f"Processed audio channel saved as {modified_fc_channel_file} and SRT subtitle file as {output_srt}.")

    # Combine multi-channel audio stream and replace channel in the container file
    if layout in ["mono", "stereo", "2.1", "3.0", "3.0(back)"]:
        final_video = replace_channel(out_video_name, out_video_ext, input_video, modified_audio_stream, audio_start_time, output_srt)
        print(f"Completed {final_video} conversion.")
        cleanup_temp_files(input_video, in_video_name, output_srt, channel_file, modified_fc_channel_file, modified_audio_stream)
    else:
        modified_audio_stream = combine_multi_channel_audio(layout, out_video_name, audio_bit_rate, output_files, modified_fc_channel_file)
        final_video = replace_channel(out_video_name, out_video_ext, input_video, modified_audio_stream, audio_start_time, output_srt)
        cleanup_temp_files(input_video, in_video_name, output_srt, channel_file, modified_fc_channel_file, modified_audio_stream, output_files)
        print (f"Video creation modified channel file: {modified_audio_stream}")
        print(f"Completed {final_video} conversion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Identify audio stream and extract channels.')
    parser.add_argument('video_file', type=str, help='Path to the original video file')
    args = parser.parse_args()

    main(args.video_file)
