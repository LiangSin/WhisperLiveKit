import os
import string

class LibriSpeechDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self._scan()

    def _scan(self):
        # LibriSpeech structure: root/speaker/chapter/speaker-chapter-index.flac
        # Transcripts: root/speaker/chapter/speaker-chapter.trans.txt
        # We walk through all directories.
        
        # Temporary storage to group by chapter
        chapters = {}
        
        for root, dirs, files in os.walk(self.root_dir):
            trans_files = [f for f in files if f.endswith('.trans.txt')]
            for trans_file in trans_files:
                trans_path = os.path.join(root, trans_file)
                try:
                    with open(trans_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                file_id, transcript = parts
                                # Audio file should have same id but .flac extension
                                audio_filename = file_id + '.flac'
                                if audio_filename in files:
                                    audio_path = os.path.join(root, audio_filename)
                                    
                                    # Parse file_id to get speaker and chapter
                                    # Format: SPEAKER-CHAPTER-INDEX
                                    id_parts = file_id.split('-')
                                    if len(id_parts) >= 2:
                                        speaker_id = id_parts[0]
                                        chapter_id = id_parts[1]
                                        group_key = f"{speaker_id}-{chapter_id}"
                                        
                                        if group_key not in chapters:
                                            chapters[group_key] = []
                                            
                                        chapters[group_key].append({
                                            'id': file_id,
                                            'audio_path': audio_path,
                                            'text': transcript
                                        })
                except Exception as e:
                    print(f"Error reading transcript file {trans_path}: {e}")
        
        # Convert dictionary values to list of lists
        # Sort samples within each chapter by ID to ensure correct order
        for key in chapters:
            chapters[key].sort(key=lambda x: x['id'])
            self.samples.append(chapters[key])
            
        # Sort chapters by the first sample's ID for consistent ordering
        self.samples.sort(key=lambda x: x[0]['id'] if x else "")

    def normalize(self, text):
        """
        Normalize text to match LibriSpeech format (UPPERCASE, no punctuation).
        """
        # Replace common punctuation with spaces or remove
        text = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
        text = text.replace('-', ' ') # Hyphens often split words in ASR output
        # Remove other punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        return " ".join(text.split()).upper()

    def get_group_id(self, samples):
        """
        Derive a group ID from the samples in a chapter.
        """
        if not samples:
            return "unknown"
            
        # Derive a group ID from the first sample
        first_id_parts = samples[0]['id'].split('-')
        if len(first_id_parts) >= 2:
            group_id = f"{first_id_parts[0]}-{first_id_parts[1]}"
        else:
            group_id = samples[0]['id']
        return group_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Alias for generic usage
Dataset = LibriSpeechDataset
