from copy import copy
from enum import Enum, auto
from itertools import count
import time

from nanovllm.sampling_params import SamplingParams

# Token ID used for image placeholder tokens in Qwen2.5-VL sequences.
IMAGE_TOKEN_ID = 151655


def _compute_image_token_ranges(token_ids: list, image_hashes: list) -> list:
    """Return [(start, end, hash), ...] for each image's consecutive token run.

    Each entry marks the half-open range [start, end) of token positions in
    *token_ids* that belong to one image, paired with that image's content hash.
    Images are matched to hash values in order of appearance.
    """
    ranges = []
    image_index = 0
    token_pos = 0
    while token_pos < len(token_ids) and image_index < len(image_hashes):
        if token_ids[token_pos] == IMAGE_TOKEN_ID:
            end_pos = token_pos
            while end_pos < len(token_ids) and token_ids[end_pos] == IMAGE_TOKEN_ID:
                end_pos += 1
            ranges.append((token_pos, end_pos, image_hashes[image_index]))
            image_index += 1
            token_pos = end_pos
        else:
            token_pos += 1
    return ranges


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), mm_inputs: dict = None, image_hashes: list[int] = None):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.mm_inputs = mm_inputs
        self.image_hashes = image_hashes

        # Token ranges for each image — used by BlockManager for position-agnostic
        # KV cache hashing.  List of (start, end, image_content_hash).
        self.image_token_ranges = (
            _compute_image_token_ranges(token_ids, image_hashes)
            if image_hashes else []
        )

        # Block indices (within this sequence's block_table) that were reused from
        # the KV cache via image-content hash even though the text prefix before
        # the image did not match (non-contiguous image KV reuse).
        self.image_reused_blocks: set = set()

        # Timing metrics
        self.start_time = time.time()
        self.vit_time = 0.0
        self.ttft = 0.0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    @property
    def image_hash(self):
        if not self.image_hashes:
            return None
        h = 0
        for x in self.image_hashes:
            h ^= x
        return h

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token,
                self.mm_inputs, self.start_time, self.vit_time, self.ttft, self.image_hashes,
                self.image_reused_blocks)

    def __setstate__(self, state):
        if len(state) == 11:
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs, self.start_time, self.vit_time, self.ttft,
             self.image_hashes, self.image_reused_blocks) = state
        elif len(state) == 10:
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs, self.start_time, self.vit_time, self.ttft,
             self.image_hashes) = state
            self.image_reused_blocks = set()
        elif len(state) == 9:
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs, self.start_time, self.vit_time, self.ttft) = state
            self.image_hashes = None
            self.image_reused_blocks = set()
        else:
            # Backward compatibility
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs) = state
            self.start_time = time.time()
            self.vit_time = 0.0
            self.ttft = 0.0
            self.image_hashes = None
            self.image_reused_blocks = set()

        # image_token_ranges is only needed in BlockManager (scheduler process);
        # worker model-runner processes only need image_reused_blocks.
        self.image_token_ranges = []

        if self.num_completion_tokens == 0:
            self.token_ids = token_data
        else:
            self.last_token = token_data
