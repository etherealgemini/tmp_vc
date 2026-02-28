from copy import copy
from enum import Enum, auto
from itertools import count
import time

from nanovllm.sampling_params import SamplingParams

# Token ID used for image placeholder tokens in Qwen2.5-VL sequences.
IMAGE_TOKEN_ID = 151655

# Padding token used to align image boundaries to block boundaries.
PAD_TOKEN_ID = 0


def _pad_for_block_alignment(
    token_ids: list, image_token_ranges: list, block_size: int,
) -> tuple[list[int], list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Pad *token_ids* so that each image's tokens occupy dedicated, complete blocks.

    Padding tokens (``PAD_TOKEN_ID``) are inserted:

    * **before** each image – to push the image start to the next block boundary,
    * **after** each image – to fill the remainder of the last image block.

    This guarantees that every KV-cache block that falls inside an image range
    contains *only* tokens from that single image (plus deterministic padding),
    enabling position-agnostic cross-request reuse of image KV blocks.

    Returns ``(padded_token_ids, padded_image_token_ranges, postpad_ranges)`` where:

    * *padded_image_token_ranges* are the ranges adjusted to the new, block-aligned
      positions.
    * *postpad_ranges* is a list of ``(start, end)`` half-open intervals in the
      padded sequence identifying the post-padding tokens appended at the end of
      each image's last block.  These positions sit inside image KV blocks but
      carry no real image content; their KV values must **not** be stored so that
      the KV cache is equivalent before and after image-block reuse.
    """
    if not image_token_ranges:
        return list(token_ids), [], []

    padded: list[int] = []
    padded_ranges: list[tuple[int, int, int]] = []
    postpad_ranges: list[tuple[int, int]] = []
    src = 0  # next uncopied position in the original token_ids

    for img_start, img_end, img_hash in image_token_ranges:
        # --- text segment preceding this image ---
        padded.extend(token_ids[src:img_start])

        # Pad text segment to the next block boundary so the image starts clean.
        remainder = len(padded) % block_size
        if remainder != 0:
            padded.extend([PAD_TOKEN_ID] * (block_size - remainder))

        # --- image segment ---
        p_img_start = len(padded)
        padded.extend(token_ids[img_start:img_end])

        # Record the end of the real image tokens before post-padding.
        real_img_end = len(padded)

        # Pad image segment to fill the last block completely.
        remainder = len(padded) % block_size
        if remainder != 0:
            padded.extend([PAD_TOKEN_ID] * (block_size - remainder))

        p_img_end = len(padded)

        # Track the post-pad range (may be empty when image already block-aligned).
        if p_img_end > real_img_end:
            postpad_ranges.append((real_img_end, p_img_end))

        padded_ranges.append((p_img_start, p_img_end, img_hash))
        src = img_end

    # --- remaining text after the last image ---
    padded.extend(token_ids[src:])

    return padded, padded_ranges, postpad_ranges


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

        # Pad token_ids so that every image occupies dedicated, complete blocks.
        # This ensures image KV-cache blocks contain only a single image's tokens,
        # enabling cross-request reuse regardless of surrounding text.
        # image_postpad_ranges records the half-open (start, end) intervals of the
        # post-padding tokens appended to the last block of each image.  These
        # positions must never have their KV values stored so that image KV blocks
        # contain only real image token KV, making cache usage equivalent before
        # and after reuse.
        if self.image_token_ranges:
            self.token_ids, self.image_token_ranges, self.image_postpad_ranges = (
                _pad_for_block_alignment(
                    self.token_ids, self.image_token_ranges, self.block_size,
                )
            )
            self.num_tokens = len(self.token_ids)
            self.num_prompt_tokens = self.num_tokens
        else:
            self.image_postpad_ranges: list[tuple[int, int]] = []

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

    @property
    def image_postpad_count(self) -> int:
        """Total number of post-padding tokens appended inside image KV blocks.

        These tokens fill the last block of each image to a multiple of
        ``block_size`` for cache-reuse alignment but carry no real image
        content.  Their KV values are never stored, so the KV cache used
        before and after image-block reuse remains equivalent.
        """
        return sum(e - s for s, e in self.image_postpad_ranges)

    @property
    def num_effective_tokens(self) -> int:
        """Number of tokens excluding image-block post-padding."""
        return self.num_tokens - self.image_postpad_count

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
                self.image_reused_blocks, self.image_postpad_ranges)

    def __setstate__(self, state):
        # Each branch corresponds to a historical state tuple length.
        # v5 (len=12): adds image_postpad_ranges (this PR).
        # v4 (len=11): adds image_reused_blocks.
        # v3 (len=10): adds image_hashes.
        # v2 (len=9):  adds start_time, vit_time, ttft.
        # v1 (len=6):  original format.
        if len(state) == 12:          # v5
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs, self.start_time, self.vit_time, self.ttft,
             self.image_hashes, self.image_reused_blocks, self.image_postpad_ranges) = state
        elif len(state) == 11:        # v4
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs, self.start_time, self.vit_time, self.ttft,
             self.image_hashes, self.image_reused_blocks) = state
            self.image_postpad_ranges = []
        elif len(state) == 10:        # v3
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs, self.start_time, self.vit_time, self.ttft,
             self.image_hashes) = state
            self.image_reused_blocks = set()
            self.image_postpad_ranges = []
        elif len(state) == 9:         # v2
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs, self.start_time, self.vit_time, self.ttft) = state
            self.image_hashes = None
            self.image_reused_blocks = set()
            self.image_postpad_ranges = []
        else:                         # v1 (len=6)
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
             token_data, self.mm_inputs) = state
            self.start_time = time.time()
            self.vit_time = 0.0
            self.ttft = 0.0
            self.image_hashes = None
            self.image_reused_blocks = set()
            self.image_postpad_ranges = []

        # image_token_ranges is only needed in BlockManager (scheduler process);
        # worker model-runner processes only need image_reused_blocks.
        self.image_token_ranges = []

        if self.num_completion_tokens == 0:
            self.token_ids = token_data
        else:
            self.last_token = token_data
