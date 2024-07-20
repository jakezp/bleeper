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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

def get_video_name(video_file):
    video_name, video_ext = os.path.splitext(os.path.basename(video_file))
    input_video = f"input_{video_name}{video_ext}"
    out_video_name = video_name
    out_video_ext = video_ext
    os.rename(video_file, input_video)
    in_video_name, in_video_ext = os.path.splitext(os.path.basename(input_video))
    return video_name, video_ext, input_video, in_video_name, in_video_ext, out_video_name, out_video_ext

def list_audio_streams(input_video):
    probe = ffmpeg.probe(input_video)
    return [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']

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

def prompt_user_to_select_stream(audio_streams):
    num_streams = len(audio_streams)
    if num_streams == 1:
        # If there is only one stream, automatically select it
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

def match_channel_layout(layout):
    # Strip any extra information from the channel layout string
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

def extract_channels(in_video_name, input_video, selected_stream_index, output_prefix=None, output_files=None):
    output_prefix = in_video_name
    audio_streams = list_audio_streams(input_video)
    selected_stream = audio_streams[selected_stream_index]
    print(selected_stream)
    # Get audio stream individual values
    layout = selected_stream['channel_layout']
    num_channels = selected_stream['channels']
    channels = match_channel_layout(layout)
    codec_name = selected_stream['codec_name']
    audio_start_time = selected_stream['start_time']
    audio_bit_rate = selected_stream['bit_rate']

    '''    
    print(selected_stream)  # Debug
    print(f"Channels: {channels}")  # Debug
    print(f"Start time: {audio_start_time}")  # Debug
    print(f"Bit rate: {audio_bit_rate}")  # Debug
    #input("Press any key to continue...")   #Debug
    {'index': 2, 
     'codec_name': 
     'ac3', 
     'codec_long_name': 
     'ATSC A/52A (AC-3)', 
     'codec_type': 'audio', 
     'codec_tag_string': '[0][0][0][0]', 
     'codec_tag': '0x0000', 
     'sample_fmt': 'fltp', 
     'sample_rate': '48000', 
     'channels': 6, 
     'channel_layout': '5.1(side)', 
     'bits_per_sample': 0, 
     'dmix_mode': '-1', 
     'ltrt_cmixlev': '-1.000000', 
     'ltrt_surmixlev': '-1.000000', 
     'loro_cmixlev': '-1.000000', 
     'loro_surmixlev': '-1.000000', 
     'r_frame_rate': '0/0', 
     'avg_frame_rate': '0/0', 
     'time_base': '1/1000', 
     'start_pts': 5, 
     'start_time': 
     '0.005000', 
     'bit_rate': '640000', 
     'disposition': {'default': 0, 
                     'dub': 0, 
                     'original': 0, 
                     'comment': 0, 
                     'lyrics': 0, 
                     'karaoke': 0, 
                     'forced': 0, 
                     'hearing_impaired': 0, 
                     'visual_impaired': 0, 
                     'clean_effects': 0, 
                     'attached_pic': 0, 
                     'timed_thumbnails': 0}, 
                     'tags': {'language': 'eng', 
                              'title': 'DD 5.1',
                              'BPS-eng': '640000', 
                              'DURATION-eng': '00:47:41.088000000', 
                              'NUMBER_OF_FRAMES-eng': '89409', 
                              'NUMBER_OF_BYTES-eng': '228887040', 
                              '_STATISTICS_WRITING_APP-eng': "mkvpropedit v45.0.0 ('Heaven in Pennies') 64-bit", 
                              '_STATISTICS_WRITING_DATE_UTC-eng': '2023-05-20 10:07:39', 
                              '_STATISTICS_TAGS-eng': 'BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES', 
                              'DURATION': '00:47:41.093000000'}}
    '''
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
                #input("Press any key to continue...")

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", input_video,
                "-filter_complex", f"[0:a:{selected_stream_index}]channelsplit=channel_layout={layout}[{']['.join(channels)}]"
            ]
            for i, channel in enumerate(channels):
                ffmpeg_cmd.extend(["-map", f"[{channel}]", "-b:a", audio_bit_rate,  "-ac", "1", output_files[i]])
            print(f"ffmpeg command: {ffmpeg_cmd}") # Debug
            #input("Press any key to continue...") # Debug
            subprocess.run(ffmpeg_cmd, check=True)
            
            #channel_file = output_files[2]
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
            #input("Press any key to continue...")   #Debug
            return layout, num_channels, channels, codec_name, audio_start_time, audio_bit_rate, channel_file, output_files

    except subprocess.CalledProcessError as e:
        print(f"Error extracting channels: {e}")
        return

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
    
    # Save the transcription result as a JSON file
    json_file = f"{in_video_name}.json"
    with open(json_file, "w") as f:
        json.dump(result, f, indent=4)

    # Save the transcription result as an SRT file
    srt_file = f"{in_video_name}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    return in_video_name

def format_timestamp(seconds):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

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

def load_srt(in_video_name):
    input_srt = (f"{in_video_name}.srt")
    """Read the content of the SRT file."""
    with open(input_srt, 'r', encoding='utf-8') as file:
        return file.readlines()

def replace_words_in_srt(lines, swear_words):
    """Replace words in the SRT file content based on the swear words set."""
    # Compile a regex pattern to match the words
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in swear_words) + r')\b', re.IGNORECASE)
    modified_lines = []
    for line in lines:
        # Replace words only in text lines (not timestamps)
        if not line.strip().isdigit() and '-->' not in line:
            # Replace each match with ***
            line = pattern.sub('***', line)
        modified_lines.append(line)
    return modified_lines

def save_srt(out_video_name, lines):
    output_srt = (f"{out_video_name}_modified.srt")
    print (f"{output_srt}")
    #input("SRT Input... Press enter")
    with open(output_srt, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    return output_srt

def identify_swearing_timestamps(in_video_name, swear_words):
    # Construct the expected filename for timestamps file
    timestamp_file = (f"{in_video_name}.json")
    # Load timestamps data
    with open(timestamp_file) as f:
        timestamps_data = json.load(f)
    swear_timestamps = []
    for segment in timestamps_data["segments"]:
        for word in segment["words"]:
            # Remove punctuation from the word
            word_without_punct = ''.join(char for char in word["word"].lower() if char not in set(string.punctuation))            
            # Check if the word (without punctuation and case-insensitive) is in the swear_words set
            if word_without_punct in (sw.lower().strip(string.punctuation) for sw in swear_words):
                start_time = word["start"] - 0.02
                end_time = word["end"] + 0.02
                swear_timestamps.append({"start": start_time, "end": end_time})
    return swear_timestamps

    #swear_timestamps = []
    #for segment in timestamps_data["segments"]:
    #    for word in segment["words"]:
    #        # Remove punctuation from the word
    #        word_without_punct = ''.join(char for char in word["word"].lower() if char not in set(string.punctuation))
    #        
    #        # Check if the word (without punctuation and case-insensitive) is in the swear_words set
    #        if word_without_punct in (sw.lower().strip(string.punctuation) for sw in swear_words):
    #            swear_timestamps.append({"start": word["start"], "end": word["end"]})
    #return swear_timestamps

def get_non_swearing_intervals(swear_timestamps, duration):
    non_swearing_intervals = []
    if not swear_timestamps:
        return [{'start': 0, 'end': duration}]
    if swear_timestamps[0]['start'] > 0:
        non_swearing_intervals.append({'start': 0, 'end': swear_timestamps[0]['start']})
    # Add intervals between the swearing intervals
    for i in range(len(swear_timestamps) - 1):
        non_swearing_intervals.append({                                                                                                                                                                                                                                                                                                                                                 
            'start': swear_timestamps[i]['end'],
            'end': swear_timestamps[i + 1]['start']
        })
    # Add interval from the end of the last swearing interval to the end of the audio
    if swear_timestamps and isinstance(duration, str):
        duration = float(duration)
        print(duration)
        print(swear_timestamps[-1]['end'])
    if swear_timestamps[-1]['end'] < duration:
        non_swearing_intervals.append({'start': swear_timestamps[-1]['end'], 'end': duration})
    return non_swearing_intervals

def create_ffmpeg_filter(swear_timestamps, non_swearing_intervals):
    """                                                 
    Creates the full FFmpeg filter complex string for muting swearing intervals and adding a bleep sound to non-swearing parts.
    Parameters:
    - swear_timestamps: A list of dictionaries with "start" and "end" keys indicating the intervals of swearing.
    - duration: The total duration of the audio in seconds.

    Returns:
    - filter_complex: A string representing the FFmpeg filter complex command.
    """
    # Create the 'dippedVocals' filter part
    dipped_vocals_conditions = '+'.join([f"between(t,{b['start']},{b['end']})" for b in swear_timestamps])
    dipped_vocals_filter = f"[0]volume=0:enable='{dipped_vocals_conditions}'[main]"

    # Create the 'dippedBleep' filter part
    no_bleeps_conditions = '+'.join([f"between(t,{segment['start']},{segment['end']})" for segment in non_swearing_intervals])
    dipped_bleep_filter = f"sine=f=800,pan=stereo|FL=c0|FR=c0,volume=0:enable='{no_bleeps_conditions}'[beep]"

    # Combine the dipped vocals and dipped bleep sounds
    amix_filter = "[main][beep]amix=inputs=2:duration=first"

    # Join all parts into a single filter complex string
    filter_complex = ';'.join([
        dipped_vocals_filter,
        dipped_bleep_filter,
        amix_filter
    ])
    return filter_complex

def apply_complex_filter(channel_file, filter_complex, format_bitrate=None):
    bitrate = format_bitrate
    # Get the input audio file name and extension
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
    modified_channel_file = output_file
    return modified_channel_file

def combine_multi_channel_audio(layout, out_video_name, audio_bit_rate, output_files, modified_channel_file):
    '''
    Combine multiple audio channel files into a single 5.1 channel audio file.

    Args:
        channel_files (list): A list of file paths for the individual audio channel files.
        output_file (str): Path to the output 5.1 channel audio file.

    Returns:
        subprocess.CompletedProcess: The completed FFmpeg process.
    '''
    #Output file name
    modified_output_file = f"{out_video_name}_modified.ac3"
    
    # Find the file that contains "_FC" in its name
    center_channel_file = next((f for f in output_files if "_FC" in os.path.basename(f)), None)

    if center_channel_file:
        # Get the channel mapping from the match_channel_layout function
        channel_mapping = match_channel_layout(layout)
        if channel_mapping:
            # Create the channel mapping string
            channel_indices = "|".join([str(i) for i in range(len(channel_mapping))])
            channel_names = "|".join(channel_mapping)
            channel_mapping_str = f"{channel_indices}:{channel_names}"
            # Build the FFmpeg command
            ffmpeg_cmd = [
                "ffmpeg", "-y"
            ]

            for file in output_files:
                if file == center_channel_file:
                    ffmpeg_cmd.extend(["-ac", "1", "-i", modified_channel_file])
                else:
                    ffmpeg_cmd.extend(["-ac", "1", "-i", file])

            # Create the input stream mapping for the amerge filter
            input_stream_mapping = []
            for i in range(len(output_files)):
                input_stream_mapping.append(f"[{i}:a]")

            # Join the input stream mapping with the amerge filter
            amerge_filter = "".join(input_stream_mapping) + f"amerge=inputs={len(output_files)},channelmap={channel_mapping_str}"

            ffmpeg_cmd.extend([
                "-filter_complex", amerge_filter,
                "-b:a", audio_bit_rate,
                "-ac", str(len(output_files)),
                modified_output_file
            ])
            
            '''
            ffmpeg_cmd.extend([
                "-filter_complex", f"[0:a][1:a][2:a][3:a][4:a][5:a]amerge=inputs={len(output_files)},channelmap={channel_mapping_str}",
                "-b:a", audio_bit_rate,
                "-ac", str(len(output_files)),
                modified_output_file
            ])
            '''
            print("FFmpeg command:", " ".join(ffmpeg_cmd))
            input("Go for it! Press enter...")
            subprocess.run(ffmpeg_cmd, check=True)
            return modified_output_file
        else:
            print("Unsupported channel layout")
            return None
    else:
        print("Channel file not found")
        return None    

def replace_channel(out_video_name, out_video_ext, input_video, modified_channel_file, audio_start_time, output_srt):
    # Get the input video file name and extension
    #video_name = os.path.splitext(os.path.basename(video_file))
    final_video = f"{out_video_name}{out_video_ext}"
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ss", audio_start_time,
        "-i", modified_channel_file,
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

def cleanup_temp_files(input_video, in_video_name, output_srt, channel_file, modified_channel_file, output_files):
    # List of files and directories to remove
    temp_files = [
        f"{in_video_name}.json",
        f"{in_video_name}.srt",
        input_video,
        output_srt,
        channel_file,
        modified_channel_file,
        #"temp_dir"  # Assuming you created a temporary directory named "temp_dir"
    ]

    for output_file in output_files:
        if os.path.isfile(output_file):
            temp_files.append(output_file)

    for item in temp_files:
        try:
            if os.path.isfile(item):
                os.remove(item)
            elif os.path.isdir(item):
                shutil.rmtree(item)
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")

def main(video_file):
    # Set of words to remove
    swear_words = {"Godspeed", "God", "Jesus", "Christ's", "Christ", "Goddamn", "shit", "shitty", "bullshit", "ass", "arse", "bitch", "bitching", "dick", "pussy", "cunt", "blowjob", "damn", "fuck", "fucked", "fucking", "fucker", "motherfucker", "fuckin", "motherfucking", "motherfuck", "face-fucks", "fucks"}    
    # Identify the audio stream
    video_name, video_ext, input_video, in_video_name, in_video_ext, out_video_name, out_video_ext = get_video_name(video_file)
    audio_streams = list_audio_streams(input_video)
    audio_streams_info = get_audio_streams_info(input_video)
    selected_stream_index = prompt_user_to_select_stream(audio_streams)
    
    # Extract audio stream
    layout, num_channels, channels, codec_name, audio_start_time, audio_bit_rate, channel_file, output_files = extract_channels(in_video_name, input_video, selected_stream_index, 0)
    # print (f"Number of channels: {num_channels}") # Debug
    # print (f"Channel file: {channel_file}") # Debug
    sample_rate, codec_name, channels, duration, format_name, format_bitrate = get_audio_properties(channel_file)
    # print (f"Channel audio sample rate: {sample_rate}") # Debug
    # print (f"Channel audio codec: {codec_name}") # Debug
    # print (f"Channel audio channels: {channels}") # Debug
    # print (f"Channel audio format name: {format_name}") # Debug
    # print (f"Channel audio format bitrate: {format_bitrate}") # Debug
    #input\("Press any key to continue...") # Debug
    
    # Transcribe audio stream and generate timestampped output files (json, SRT)
    run_whisperx(channel_file, in_video_name) # Debug
    # print (f"WhisperX channel file: {channel_file}") # Debug
    # print (f"WhisperX video file: {video_file}") # Debug
    # print (f"WhisperX video name: {video_name}") # Debug
    #input("Press any key to continue...") # Debug

    # Create ffmpeg complex filter
    swear_timestamps = identify_swearing_timestamps(in_video_name, swear_words)
    non_swearing_intervals = get_non_swearing_intervals(swear_timestamps, duration)
    filter_complex = create_ffmpeg_filter(swear_timestamps, non_swearing_intervals)
    # print (f"Swearing timestamps: {swear_timestamps}")
    # print (f"Inverse timestamps: {non_swearing_intervals}")
    print (f"Complex filter: {filter_complex}")
    # print (f"")
    # Redact (bleep) audio stream and SRT subtitle file
    modified_channel_file = apply_complex_filter(channel_file, filter_complex, format_bitrate)
    lines = load_srt(in_video_name)
    modified_lines = replace_words_in_srt(lines, swear_words)
    output_srt = save_srt(out_video_name, modified_lines)
    #print (f"Modified channel file: {modified_channel_file}")
    print(f"Processed audio channel saved as {modified_channel_file} and SRT subtitle file as {output_srt}.")
    
    '''
    # Add the modified audio stream to the original video file. If it's multi-channel audio, first combine the channels,
    # then add the modified audio stream to the original file.
    # The modified audio_stream and subtitle_stream is set as default
    '''
    # print(f"Nr channels: {num_channels}")
    # print(f"Audio start time: {audio_start_time}")
    # input("Nr of channels and audio start time... Enter")
    if layout in ["mono", "stereo", "2.1", "3.0", "3.0(back)"]:
        final_video = replace_channel(out_video_name, out_video_ext, input_video, modified_channel_file, audio_start_time, output_srt)
        print(f"Completed {final_video} conversion.")
    else:
        modified_channel_file = combine_multi_channel_audio(layout, out_video_name, audio_bit_rate, output_files, modified_channel_file)
        final_video = replace_channel(out_video_name, out_video_ext, input_video, modified_channel_file, audio_start_time, output_srt)
        print (f"Video creation modified channel file: {modified_channel_file}")
        print(f"Completed {final_video} conversion.")

    # Clean up temporary files.
    cleanup_temp_files(input_video, in_video_name, output_srt, channel_file, modified_channel_file, output_files)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Identify audio stream and extract channels.')
    parser.add_argument('video_file', type=str, help='Path to the original video file')
    args = parser.parse_args()

    main(args.video_file)
