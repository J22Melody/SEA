#!/usr/bin/env python3
import os
import argparse
import re
import torch
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool

# Uncomment these lines if you have not already downloaded the resources.
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Global lemmatizer instance.
lemmatizer = WordNetLemmatizer()

# Example stop words set.
stop_words = {"the", "a", "an", "and", "or"}

def fix_contractions(text):
    """Fix contractions in the given text."""
    fixed_text = contractions.fix(text)
    return fixed_text

def remove_punctuation(text):
    """Remove punctuation from the given text."""
    punctuation_pattern = r'[^\w\s]'  # Matches any character that is not a word character or whitespace
    text_without_punctuation = re.sub(punctuation_pattern, '', text)
    return text_without_punctuation

def classify_have(sentence):
    """Classify whether 'have' is a verb or an auxiliary verb in the given sentence."""
    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    new_sentence = []
    for i, (word, tag) in enumerate(tagged_words):
        if word == 'have' and i < len(tagged_words) - 1:
            next_word, next_tag = tagged_words[i + 1]
            if next_tag.startswith('V'):
                # Skip 'have' when it appears as an auxiliary verb.
                continue
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)
    return new_sentence

def lemmatize_word(word):
    """Lemmatize the given word."""
    lemmatized_word = lemmatizer.lemmatize(word)
    return lemmatized_word

def convert_verb_tense(sentence, target_tense):
    """Convert the tense of verbs in the given sentence to the target tense."""
    local_lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)
    converted_sentence = []
    for word, tag in tagged_words:
        if tag.startswith('VB'):  # Check if the word is a verb
            lemma = local_lemmatizer.lemmatize(word, pos='v')
            if target_tense == 'present':
                converted_word = lemma
            elif target_tense == 'past':
                if tag == 'VBP':  # Present tense tag
                    converted_word = local_lemmatizer.lemmatize(word, pos='v') + 'ed'
                else:
                    converted_word = lemma
            else:
                converted_word = lemma
        else:
            converted_word = word
        converted_sentence.append(converted_word)
    return converted_sentence

def nearest_dilation_1d(input_tensor, dilation_factor):
    """
    Perform nearest dilation on a 1D tensor along its last dimension.
    Note: This function moves the tensor to GPU.
    """
    input_tensor = input_tensor.cuda()
    input_tensor = input_tensor.squeeze(-1)
    output_tensor = torch.zeros_like(input_tensor)
    for i, mat in enumerate(input_tensor):
        indices = (mat == 1).nonzero(as_tuple=False)
        if len(indices) > 0:
            start_idx = max(0, indices[0] - dilation_factor)
            end_idx = min(len(mat), indices[-1] + dilation_factor + 1)
            output_tensor[i, start_idx:end_idx] = 1
        else:
            output_tensor[i] = torch.ones_like(input_tensor[i])
    return output_tensor.unsqueeze(-1)

def clean_text(
    arr,
    remove_stopwords=True,
    lemmatize_words=False,
    stem_words=False,
    preprocess_words=False,
    remove_be=False,
    remove_have=False
):
    """
    Process an array of words from a subtitle unit according to various text cleaning options.
    """
    if remove_stopwords:
        arr = [w for w in arr if w not in stop_words]
        # Remove possessives.
        arr = [w.replace("'s", "").replace("'", "") for w in arr]

    if lemmatize_words:
        try:
            arr = [lemmatize_word(w) for w in arr]
        except Exception:
            pass

    if stem_words:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        arr = [stemmer.stem(w) for w in arr]

    if preprocess_words:
        refined_text = ' '.join(arr)
        refined_text = fix_contractions(refined_text)
        refined_text = remove_punctuation(refined_text)
        # Convert verbs to the target tense (here, present tense).
        arr = convert_verb_tense(refined_text, 'present')
        # Remove common articles.
        arr = [w for w in arr if w not in ['a', 'an', 'the']]

    if remove_be:
        arr = [w for w in arr if w not in ['be']]

    if remove_have:
        refined_text = ' '.join(arr)
        arr = classify_have(refined_text)

    return arr

def process_subtitle_text(text,
                          remove_stopwords=True,
                          lemmatize_words=False,
                          stem_words=False,
                          preprocess_words=False,
                          remove_be=False,
                          remove_have=False):
    """
    Process a single subtitle text line by tokenizing, cleaning, and rejoining it.
    """
    tokens = text.split()
    cleaned_tokens = clean_text(tokens,
                                remove_stopwords=remove_stopwords,
                                lemmatize_words=lemmatize_words,
                                stem_words=stem_words,
                                preprocess_words=preprocess_words,
                                remove_be=remove_be,
                                remove_have=remove_have)
    return ' '.join(cleaned_tokens)

def process_vtt_file(input_path, output_path,
                     remove_stopwords=True,
                     lemmatize_words=False,
                     stem_words=False,
                     preprocess_words=False,
                     remove_be=False,
                     remove_have=False):
    """
    Read a VTT file, process each subtitle unit text, and write the processed content.
    """
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    output_lines = []
    current_block = []
    is_header = True

    for line in lines:
        if line.strip() == "":
            if current_block:
                if is_header:
                    # Write header (e.g., "WEBVTT") without processing.
                    output_lines.extend(current_block)
                    output_lines.append('\n')
                    is_header = False
                else:
                    # Process subtitle cue block.
                    if '-->' in current_block[0]:
                        timing_line = current_block[0]
                        text_lines = current_block[1:]
                        processed_text_lines = [
                            process_subtitle_text(text_line.strip(),
                                                  remove_stopwords=remove_stopwords,
                                                  lemmatize_words=lemmatize_words,
                                                  stem_words=stem_words,
                                                  preprocess_words=preprocess_words,
                                                  remove_be=remove_be,
                                                  remove_have=remove_have) + '\n'
                            for text_line in text_lines if text_line.strip()
                        ]
                        new_block = [timing_line] + processed_text_lines
                    else:
                        identifier = current_block[0]
                        timing_line = current_block[1]
                        text_lines = current_block[2:]
                        processed_text_lines = [
                            process_subtitle_text(text_line.strip(),
                                                  remove_stopwords=remove_stopwords,
                                                  lemmatize_words=lemmatize_words,
                                                  stem_words=stem_words,
                                                  preprocess_words=preprocess_words,
                                                  remove_be=remove_be,
                                                  remove_have=remove_have) + '\n'
                            for text_line in text_lines if text_line.strip()
                        ]
                        new_block = [identifier, timing_line] + processed_text_lines
                    output_lines.extend(new_block)
                    output_lines.append('\n')
                current_block = []
        else:
            current_block.append(line)

    # Process any remaining block if file doesn't end with a blank line.
    if current_block:
        if is_header:
            output_lines.extend(current_block)
        else:
            if '-->' in current_block[0]:
                timing_line = current_block[0]
                text_lines = current_block[1:]
                processed_text_lines = [
                    process_subtitle_text(text_line.strip(),
                                          remove_stopwords=remove_stopwords,
                                          lemmatize_words=lemmatize_words,
                                          stem_words=stem_words,
                                          preprocess_words=preprocess_words,
                                          remove_be=remove_be,
                                          remove_have=remove_have) + '\n'
                    for text_line in text_lines if text_line.strip()
                ]
                new_block = [timing_line] + processed_text_lines
            else:
                identifier = current_block[0]
                timing_line = current_block[1]
                text_lines = current_block[2:]
                processed_text_lines = [
                    process_subtitle_text(text_line.strip(),
                                          remove_stopwords=remove_stopwords,
                                          lemmatize_words=lemmatize_words,
                                          stem_words=stem_words,
                                          preprocess_words=preprocess_words,
                                          remove_be=remove_be,
                                          remove_have=remove_have) + '\n'
                    for text_line in text_lines if text_line.strip()
                ]
                new_block = [identifier, timing_line] + processed_text_lines
            output_lines.extend(new_block)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(output_lines)

def process_file(task):
    """
    Wrapper function for processing a single file.
    """
    (input_path, output_path,
     remove_stopwords, lemmatize_words, stem_words,
     preprocess_words, remove_be, remove_have) = task
    print(f"Processing {input_path} -> {output_path}")
    process_vtt_file(
        input_path, output_path,
        remove_stopwords=remove_stopwords,
        lemmatize_words=lemmatize_words,
        stem_words=stem_words,
        preprocess_words=preprocess_words,
        remove_be=remove_be,
        remove_have=remove_have
    )

def main():
    parser = argparse.ArgumentParser(description="Preprocess subtitle (VTT) files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction",
        help="Directory where subtitle (VTT) files are stored."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/users/zifan/BOBSL/derivatives/subtitles_preprocessed",
        help="Directory where the processed VTT files will be stored."
    )
    parser.add_argument(
        "--video_ids",
        type=str,
        default="/users/zifan/subtitle_align/data/bobsl_align.txt",
        help="Path to text file containing video ids (one per line)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes to use for processing VTT files."
    )
    # Optional cleaning parameters with updated default values.
    parser.add_argument('--remove_stopwords', type=bool, default=False, help='Remove stopwords')
    parser.add_argument('--lemmatize_words', type=bool, default=False, help='Lemmatize words')
    parser.add_argument('--stem_words', type=bool, default=False, help='Stem words')
    parser.add_argument('--preprocess_words', type=bool, default=True, help='Preprocess words (fix contractions, remove punctuation, etc.)')
    parser.add_argument('--remove_be', type=bool, default=True, help='Remove the word "be"')
    parser.add_argument('--remove_have', type=bool, default=True, help='Process the word "have"')
    
    args = parser.parse_args()

    # Load the list of valid video IDs from the provided file.
    with open(args.video_ids, 'r', encoding='utf-8') as vid_file:
        valid_video_ids = {line.strip() for line in vid_file if line.strip()}

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Build a list of tasks to process only VTT files that match a video ID in the list.
    tasks = []
    for file_name in os.listdir(args.input_dir):
        if file_name.lower().endswith('.vtt'):
            video_id = os.path.splitext(file_name)[0]
            if video_id not in valid_video_ids:
                continue  # Skip files not in the id list.
            input_path = os.path.join(args.input_dir, file_name)
            output_path = os.path.join(args.output_dir, file_name)
            tasks.append((
                input_path, output_path,
                args.remove_stopwords, args.lemmatize_words, args.stem_words,
                args.preprocess_words, args.remove_be, args.remove_have
            ))

    # Process files concurrently using a pool of workers.
    with Pool(args.num_workers) as pool:
        pool.map(process_file, tasks)

if __name__ == '__main__':
    main()
