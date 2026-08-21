from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch


@dataclass
class DecoderState:

    kv_cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    tokenizer: Any = None
    detected_language: Optional[str] = None
    reset_tokenizer_to_auto_next_call: bool = False
    
    tokens: List[torch.Tensor] = field(default_factory=list)
    initial_tokens: Optional[torch.Tensor] = None
    initial_token_length: int = 0
    sot_index: int = 0
    
    align_source: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    num_align_heads: int = 0
    
    segments: List[torch.Tensor] = field(default_factory=list)
    
    context: Any = None
    
    pending_incomplete_tokens: List[int] = field(default_factory=list)
    
    global_time_offset: float = 0.0
    cumulative_time_offset: float = 0.0
    first_timestamp: Optional[float] = None
    last_attend_frame: int = 0

    # Rewind-storm detection: absolute times (s) of recent rewind-guard hits.
    # Three hits within a 1s spread mean the decoder is stuck re-failing on
    # the same audio; rewind_storm signals the caller to refresh the context.
    recent_rewinds: List[float] = field(default_factory=list)
    rewind_storm: bool = False
    
    speaker: int = -1
    log_segments: int = 0
    
    CIFLinear: Optional[torch.nn.Module] = None
    always_fire: bool = False
    never_fire: bool = False
    
    suppress_tokens_fn: Any = None
    
    token_decoder: Any = None
    decoder_type: str = "greedy"
    
    inference: Any = None

    # Cross-session batched decode (batch_executor.py): slot index held for
    # the current round (assigned asynchronously by the executor worker),
    # and whether an adoption/round is pending with the executor.
    batch_slot: Optional[int] = None
    batch_pending: bool = False
    batch_executor: Any = None

    def clean_cache(self):
        """Clean the kv_cache after each inference step."""
        self.kv_cache = {}
        if self.batch_executor is not None and (
                self.batch_pending or self.batch_slot is not None):
            self.batch_pending = False
            # The worker clears batch_slot; FIFO ordering runs any queued
            # adoption for this round first, so the slot cannot leak.
            self.batch_executor.cancel(self)
        if self.decoder_type == "beam" and self.inference is not None:
            self.inference.kv_cache = self.kv_cache
            if self.token_decoder is not None:
                self.token_decoder.reset()
    
    def reset(self, rewind_threshold: int = 200):
        """
        Reset transient state for a new segment.
        
        Args:
            rewind_threshold: Value for resetting last_attend_frame
        """
        self.last_attend_frame = -rewind_threshold
        self.cumulative_time_offset = 0.0
        self.pending_incomplete_tokens = []
        self.recent_rewinds = []
        self.rewind_storm = False
        self.log_segments += 1
    
    def full_reset(self, rewind_threshold: int = 200):
        """
        Full reset including audio segments and tokens.
        
        Args:
            rewind_threshold: Value for resetting last_attend_frame
        """
        self.reset(rewind_threshold)
        self.segments = []
        self.tokens = []
        self.clean_cache()
        self.first_timestamp = None

