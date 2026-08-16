import torch
import sys
class TokenBuffer:

    def __init__(self, text="", tokenizer=None, device=None, prefix_token_ids=[]):
        self.text = text
        self.prefix_token_ids = prefix_token_ids
        self.tokenizer = tokenizer
        self.device = device
        self.pending_token_ids = []
        # Cache of the encoded (prefix + text) ids and their device tensor.
        # trim_context/_current_tokens re-read the context every infer round;
        # without this each read costs a full BPE encode (+ an H2D copy for
        # the tensor form). Invalidated whenever self.text mutates.
        self._ids_cache = None
        self._tensor_cache = None

    def _invalidate(self):
        self._ids_cache = None
        self._tensor_cache = None

    def as_token_ids(self, tokenizer=None):

        if tokenizer is not None and tokenizer is not self.tokenizer:
            return self.prefix_token_ids + tokenizer.encode(self.text)
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not set.")
        if self._ids_cache is None:
            self._ids_cache = self.prefix_token_ids + tokenizer.encode(self.text)
        return self._ids_cache

    def as_tensor(self, device=None):
        if device is None:
            device = self.device
        if device is None:
            raise ValueError("Device is not set.")
        if self._tensor_cache is None or self._tensor_cache.device != torch.device(device):
            tok_ids = self.as_token_ids()
            self._tensor_cache = torch.tensor(
                tok_ids, dtype=torch.long, device=device).unsqueeze(0)
        return self._tensor_cache

    def as_tensor_beam(self, beam, device=None):
        t = self.as_tensor(device=device)
        if beam == 1:
            # repeat_interleave(1) would still copy; callers never mutate the
            # returned prompt tensor in place (it is only concatenated).
            return t
        return t.repeat_interleave(beam, dim=0)


    def as_text(self):
        return self.text

    @staticmethod
    def empty(*a, **kw):
        return TokenBuffer(*a,**kw)

    @staticmethod
    def from_text(text, *a, **kw):
        return TokenBuffer(*a, text=text, **kw)

    def is_empty(self):
        return self.text is None or self.text == ""

    def trim_words(self, num=1, after=0):
        '''
        num: how many words to trim from the beginning
        after: how many characters to skip (length of the static prompt)
        '''
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."

        ids = tokenizer.encode(self.text[after:])
        words, wids = self.tokenizer.split_to_word_tokens(ids)
        if not words:
            return 0
        self.text = self.text[:after] + "".join(words[num:])
        self._invalidate()
        return sum(len(wi) for wi in wids[:num])

    def append_token_ids(self, token_ids):
        tokenizer = self.tokenizer
        assert tokenizer is not None, "Tokenizer is not set."

        all_tokens = self.pending_token_ids + token_ids

        decoded = tokenizer.decode(all_tokens)
        replacement_char = "�"

        if replacement_char in decoded:
            if len(all_tokens) > 1:
                decoded_partial = tokenizer.decode(all_tokens[:-1])

                if replacement_char not in decoded_partial:
                    self.text += decoded_partial
                    self.pending_token_ids = [all_tokens[-1]]
                    self._invalidate()
                else:
                    self.pending_token_ids = all_tokens
            else:
                self.pending_token_ids = all_tokens
        else:
            self.text += decoded
            self.pending_token_ids = []
            self._invalidate()

    def as_split_word_tokens(self):
        tokenizer = self.tokenizer
        ids = self.as_token_ids()[len(self.prefix_token_ids):]
        assert tokenizer is not None, "Tokenizer is not set."
        return tokenizer.split_to_word_tokens(ids)
