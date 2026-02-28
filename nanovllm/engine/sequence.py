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
) -> tuple[list[int], list[tuple[int, int, int]], set[int]]:
    """Pad *token_ids* so that each image's tokens occupy dedicated, complete blocks.

    Padding tokens (``PAD_TOKEN_ID``) are inserted:

    * **before** each image – to push the image start to the next block boundary,
    * **after** each image – to fill the remainder of the last image block.

    This guarantees that every KV-cache block that falls inside an image range
    contains *only* tokens from that single image (plus deterministic padding),
    enabling position-agnostic cross-request reuse of image KV blocks.

    Returns ``(padded_token_ids, padded_image_token_ranges, padding_positions)``
    where the ranges are adjusted to the new, block-aligned positions and
    *padding_positions* is a set of indices in *padded_token_ids* that are padding.
    """
    if not image_token_ranges:
        return list(token_ids), [], set()

    padded: list[int] = []
    padded_ranges: list[tuple[int, int, int]] = []
    padding_positions: set[int] = set()
    src = 0  # next uncopied position in the original token_ids

    for img_start, img_end, img_hash in image_token_ranges:
        # --- text segment preceding this image ---
        padded.extend(token_ids[src:img_start])

        # Pad text segment to the next block boundary so the image starts clean.
        remainder = len(padded) % block_size
        if remainder != 0:
            pad_start = len(padded)
            pad_count = block_size - remainder
            padded.extend([PAD_TOKEN_ID] * pad_count)
            padding_positions.update(range(pad_start, pad_start + pad_count))

        # --- image segment ---
        p_img_start = len(padded)
        padded.extend(token_ids[img_start:img_end])

        # Pad image segment to fill the last block completely.
        remainder = len(padded) % block_size
        if remainder != 0:
            pad_start = len(padded)
            pad_count = block_size - remainder
            padded.extend([PAD_TOKEN_ID] * pad_count)
            padding_positions.update(range(pad_start, pad_start + pad_count))

        p_img_end = len(padded)
        padded_ranges.append((p_img_start, p_img_end, img_hash))
        src = img_end

    # --- remaining text after the last image ---
    padded.extend(token_ids[src:])

    return padded, padded_ranges, padding_positions


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

        # Positions in the padded token_ids that are block-alignment padding.
        # Populated by _pad_for_block_alignment; cleared after KV compaction.
        self.padding_positions: set[int] = set()

        # Pad token_ids so that every image occupies dedicated, complete blocks.
        # This ensures image KV-cache blocks contain only a single image's tokens,
        # enabling cross-request reuse regardless of surrounding text.
        if self.image_token_ranges:
            self.token_ids, self.image_token_ranges, self.padding_positions = _pad_for_block_alignment(
                self.token_ids, self.image_token_ranges, self.block_size,
            )
            self.num_tokens = len(self.token_ids)
            self.num_prompt_tokens = self.num_tokens

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

    def get_non_padding_indices(self) -> list[int]:
        """Return sorted list of non-padding token positions in the padded token_ids."""
        if not self.padding_positions:
            return list(range(self.num_tokens))
        return sorted(set(range(self.num_tokens)) - self.padding_positions)

    def apply_compaction(self, new_block_table: list[int]):
        """Update sequence metadata after KV cache compaction removes padding."""
        non_padding = self.get_non_padding_indices()
        self.token_ids = [self.token_ids[i] for i in non_padding]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = self.num_tokens
        self.num_cached_tokens = self.num_tokens
        self.block_table = new_block_table
        self.padding_positions = set()
        self.image_token_ranges = []
        self.image_reused_blocks = set()

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
        # padding_positions is only needed in the scheduler process for compaction
        # planning; it is cleared after compaction and not serialized.
        self.padding_positions = set()

        if self.num_completion_tokens == 0:
            self.token_ids = token_data
        else:
            self.last_token = token_data
