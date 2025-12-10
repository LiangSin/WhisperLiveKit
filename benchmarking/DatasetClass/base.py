import string
import jiwer

class BaseDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

    def _scan(self):
        raise NotImplementedError

    def normalize(self, text):
        """
        Default normalization (LibriSpeech style).
        """
        # Replace common punctuation with spaces or remove
        text = text.replace('.', ' ').replace(',', ' ').replace('?', ' ').replace('!', ' ')
        text = text.replace('-', ' ') # Hyphens often split words in ASR output
        # Remove other punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        return " ".join(text.split()).upper()

    def compute_error_ratio(self, norm_ref: str, norm_hyp: str) -> float:
        """
        Default: WER on normalized text.
        """
        # Avoid division-by-zero semantics inside WER when reference is empty
        if len(norm_ref) == 0:
            return 1.0 if len(norm_hyp) > 0 else 0.0

        return jiwer.wer(norm_ref, norm_hyp)

    def get_group_id(self, samples):
        """
        Derive a group ID from the samples in a chapter.
        """
        if not samples:
            return "unknown"
            
        # Default behavior: use the first sample's ID
        # Subclasses can override if they have specific ID structure logic
        return samples[0]['id']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

