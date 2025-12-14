# Adapted from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/03_bonus_pretraining_on_gutenberg/prepare_dataset.py

"""
Script that processes the Project Gutenberg files into fewer larger files.
"""

import argparse, os, re
from tqdm import tqdm
from data.gutenberg.src.cleanup import strip_headers
from langdetect import detect, DetectorFactory, LangDetectException

# Langdetect -- Google's language detection algorithm
# Uses statistical models with some randomness. Setting a seed makes each detect() call
# deterministic for a given text input, ensuring reproducible results regardless of file processing order.
DetectorFactory.seed = 0
def is_english(text, min_length=100):
    if not text or len(text.strip()) < min_length:
        # Too short to reliably detect, skip it
        return False
    
    try:
        detected_lang = detect(text).lower()
        return detected_lang == 'en'
    except LangDetectException:
        # If detection fails, skip the file
        return False

def combine_files(file_paths, target_dir, 
                  max_size_mb=200, 
                  separator="<|endoftext|>", 
                  fallback_encoding="latin1"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    current_content = []
    current_size = 0
    file_counter = 1
    separator_bytes = separator.encode("utf-8")

    for file_path in tqdm(file_paths):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # Attempt to read the file with a fallback encoding
            tqdm.write(f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        # Check if content is English before processing
        if not is_english(content):
            tqdm.write(f"Skipping {file_path} as it does not contain primarily English text.")
            continue
        
        content = strip_headers(content)

        # Normalize line endings to \n only (remove \r characters)
        # This prevents \r tokens from accumulating in the training data
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Collapse 3+ blank lines to 2 and strip trailing whitespace via one regex pass
        content = re.sub(
            r'\n{3,}|[ \t]+(?=\n)',
            lambda m: '\n\n' if m.group(0).startswith('\n') else '',
            content,
        )

        # Encode once for both size accounting and writing
        content_bytes = content.encode("utf-8")
        estimated_size = len(content_bytes)
        max_size_bytes = max_size_mb * 1024 * 1024

        # If current file alone exceeds max size, write it as its own file (exception case)
        if estimated_size > max_size_bytes:            
            # Write file as its own combined file
            tqdm.write(f"Warning: {file_path} ({estimated_size / 1024 / 1024:.2f} MB) exceeds max_size_mb ({max_size_mb} MB). Writing as standalone file.")
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "wb") as target_file:
                target_file.write(content_bytes + separator_bytes)
            file_counter += 1
        # If adding this file would exceed the limit, write out current batch first
        # This ensures files are only added if they fit fully
        elif current_content and current_size + estimated_size > max_size_bytes:
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "wb") as target_file:
                target_file.write(separator_bytes.join(current_content) + separator_bytes)
            file_counter += 1
            current_content = [content_bytes]
            current_size = estimated_size
        # If adding this file would not exceed the limit, add to current batch
        else:
            current_content.append(content_bytes)
            current_size += estimated_size

    # Write out any remaining content in current batch
    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "wb") as target_file:
            target_file.write(separator_bytes.join(current_content) + separator_bytes)
    return file_counter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess and combine text files for pretraining")

    parser.add_argument("--data_dir", type=str, default="data/gutenberg/data/raw",
                        help="Directory containing the downloaded raw training data")
    parser.add_argument("--max_size_mb", type=int, default=200,
                        help="The maximum file size for each concatenated file in megabytes")
    parser.add_argument("--output_dir", type=str, default="data/pretraining/gutenberg_preprocessed",
                        help="Directory where the preprocessed data will be saved")

    args = parser.parse_args()

    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]

    print(f"{len(all_files)} file(s) to process.")
    file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")