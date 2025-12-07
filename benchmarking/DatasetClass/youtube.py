import os
import glob
import re
import jiwer
from .base import BaseDataset

class YoutubeDataset(BaseDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self._scan()

    def _scan(self):
        # Youtube structure: root/channel/playlist/video_id.wav
        # Transcripts: root/channel/playlist/video_id.*.srt
        
        # We treat each video as a chapter containing 1 sample.
        
        for root, dirs, files in os.walk(self.root_dir):
            audio_files = [f for f in files if f.endswith('.wav')]
            for audio_file in audio_files:
                # audio_file: videoID.wav
                basename = os.path.splitext(audio_file)[0]
                
                candidates = [f for f in files if f.startswith(basename) and f.endswith('.srt')]
                
                if not candidates:
                    # Skip if no srt file is found
                    continue
                
                # Prioritize 'zh' in filename
                def sort_key(filename):
                    # Lower value means higher priority
                    if 'zh' in filename.lower():
                        return 0
                    return 1
                
                candidates.sort(key=sort_key)
                srt_file = candidates[0]
                srt_path = os.path.join(root, srt_file)
                audio_path = os.path.join(root, audio_file)
                
                try:
                    text = self._parse_srt(srt_path)
                    if text:
                        sample = {
                            'id': basename,
                            'audio_path': audio_path,
                            'text': text
                        }
                        # Each video is one chapter
                        self.samples.append([sample])
                except Exception as e:
                    print(f"Error reading/parsing SRT file {srt_path}: {e}")
        
        # # Sort samples by ID
        # self.samples.sort(key=lambda x: x[0]['id'])

    def _parse_srt(self, srt_path):
        """
        Robust SRT parser to extract text.
        """
        text_parts = []
        with open(srt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # State machine:
        # 0: Expecting index (numeric)
        # 1: Expecting timestamp (contains -->)
        # 2: Expecting text (until empty line)
        state = 0
        
        for line in lines:
            line = line.strip()
            
            if state == 0:
                if line.isdigit():
                    state = 1
                elif line:
                    if '-->' in line:
                        state = 2
            elif state == 1:
                if '-->' in line:
                    state = 2
                else:
                    if not line:
                        state = 0
            elif state == 2:
                if not line:
                    # Empty line indicates end of block
                    state = 0
                else:
                    line = re.sub(r'\[.*?\]', '', line)
                    line = re.sub(r'\(.*?\)', '', line)
                    line = re.sub(r'【.*?】', '', line)
                    
                    if line.strip():
                        text_parts.append(line.strip())
            
        return " ".join(text_parts)

    def normalize(self, text):
        """
        Normalize for Youtube data (Chinese/English mixed).
        Replace punctuation (both ASCII and CJK) with spaces.
        Convert Chinese numerals to Arabic numerals.
        """
        import string

        # 1. Replace punctuation (ASCII)
        for ch in string.punctuation:
            text = text.replace(ch, " ")

        # 2. Replace CJK punctuation with spaces (extended list)
        cjk_punct = "，。！？、；：「」『』【】（）《》〈〉…—～"
        for ch in cjk_punct:
            text = text.replace(ch, " ")

        # 3. Upper case
        text = text.upper()

        # 4. Tokenize (single spaces)
        tokens = self._tokenize_mixed(text)
        text = " ".join(tokens)
        return text

    def compute_error_ratio(self, norm_ref: str, norm_hyp: str) -> float:
        """
        Compute Mixed Error Rate (MER).
        MER = (EditDistance) / (ReferenceLength)
        """
        if len(norm_ref) == 0:
            return 1.0 if len(norm_hyp) > 0 else 0.0
        
        return jiwer.wer(norm_ref, norm_hyp)

    def _tokenize_mixed(self, text):
        """
        Tokenize mixed CJK and English text.
        - English words (consecutive alphanumeric) are kept as one token.
        - CJK characters are split into individual tokens.
        """
        tokens = []
        current_word = []
        
        for char in text:
            if self._is_cjk(char):
                if current_word:
                    tokens.append("".join(current_word))
                    current_word = []
                tokens.append(char)
            elif char.strip() == "":
                if current_word:
                    tokens.append("".join(current_word))
                    current_word = []
                # Ignore spaces
            else:
                # English/Number/Other
                current_word.append(char)
        
        if current_word:
            tokens.append("".join(current_word))
            
        return tokens

    def _is_cjk(self, char):
        """
        Check if character is CJK.
        """
        # Basic CJK ranges
        code = ord(char)
        return (
            (0x4E00 <= code <= 0x9FFF) or
            (0x3400 <= code <= 0x4DBF) or
            (0x20000 <= code <= 0x2A6DF) or
            (0x2A700 <= code <= 0x2B73F) or
            (0x2B740 <= code <= 0x2B81F) or
            (0x2B820 <= code <= 0x2CEAF) or
            (0xF900 <= code <= 0xFAFF) or
            (0x2F800 <= code <= 0x2FA1F)
        )

