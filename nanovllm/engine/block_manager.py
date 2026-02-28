from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= self._count_required_free_blocks(seq)

    def _count_required_free_blocks(self, seq: Sequence) -> int:
        h = -1
        text_cache_miss = False
        required_free_blocks = 0
        image_token_ranges = getattr(seq, 'image_token_ranges', [])

        for i in range(seq.num_blocks):
            block_start = i * self.block_size
            token_ids = seq.block(i)
            full_block = len(token_ids) == self.block_size
            img_content_hash = None
            is_first_of_image = False
            if full_block and image_token_ranges:
                block_end = block_start + self.block_size
                for (img_start, img_end, img_hash) in image_token_ranges:
                    if block_start >= img_start and block_end <= img_end:
                        img_content_hash = img_hash
                        is_first_of_image = (block_start == img_start)
                        break
            if img_content_hash is not None:
                if is_first_of_image:
                    h = img_content_hash
                h = self.compute_hash(token_ids, h)
            else:
                h = self.compute_hash(token_ids, h) if full_block else -1

            block_id = self.hash_to_block_id.get(h, -1)
            cache_hit = block_id != -1 and self.blocks[block_id].token_ids == token_ids
            block_is_used = block_id in self.used_block_ids

            if img_content_hash is not None:
                if cache_hit and block_is_used:
                    continue
                required_free_blocks += 1
            elif cache_hit and not text_cache_miss:
                if not block_is_used:
                    required_free_blocks += 1
            else:
                text_cache_miss = True
                required_free_blocks += 1
        return required_free_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        # Use -1 as the initial chain seed for text blocks.  Image blocks receive
        # a position-agnostic seed derived from the image content hash so that the
        # same image can be found in the KV cache regardless of what text precedes
        # it in the sequence.
        h = -1
        text_cache_miss = False
        image_reused_blocks: set = set()
        image_token_ranges = getattr(seq, 'image_token_ranges', [])

        for i in range(seq.num_blocks):
            block_start = i * self.block_size
            token_ids = seq.block(i)
            full_block = len(token_ids) == self.block_size

            # --- Determine whether this is a pure-image block ---
            # A block qualifies when it is a full block whose entire token range
            # falls within one image's token range.
            img_content_hash = None
            is_first_of_image = False
            if full_block and image_token_ranges:
                block_end = block_start + self.block_size
                for (img_start, img_end, img_hash) in image_token_ranges:
                    if block_start >= img_start and block_end <= img_end:
                        img_content_hash = img_hash
                        is_first_of_image = (block_start == img_start)
                        break

            # --- Compute this block's hash ---
            if img_content_hash is not None:
                # Image block: reset chain to the image content hash at the start
                # of each image so that the hash is independent of the text prefix.
                if is_first_of_image:
                    h = img_content_hash
                h = self.compute_hash(token_ids, h)
            else:
                h = self.compute_hash(token_ids, h) if full_block else -1

            # --- Cache lookup ---
            # h == -1 is the "no hash" sentinel; xxhash intdigest() is always >= 0,
            # so hash_to_block_id will never contain -1 as a key.
            block_id = self.hash_to_block_id.get(h, -1)
            cache_hit = block_id != -1 and self.blocks[block_id].token_ids == token_ids

            if img_content_hash is not None:
                # Image block: attempt reuse regardless of text prefix cache state.
                if cache_hit:
                    if text_cache_miss:
                        # Non-contiguous image KV reuse: text prefix before the
                        # image missed, but the image block itself is cached.
                        image_reused_blocks.add(i)
                    else:
                        seq.num_cached_tokens += self.block_size
                    if block_id in self.used_block_ids:
                        block = self.blocks[block_id]
                        block.ref_count += 1
                    else:
                        block = self._allocate_block(block_id)
                else:
                    block_id = self.free_block_ids[0]
                    block = self._allocate_block(block_id)
            elif cache_hit and not text_cache_miss:
                # Normal text prefix cache hit.
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            else:
                # Text block miss.
                text_cache_miss = True
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)

            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        seq.image_reused_blocks = image_reused_blocks

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.image_reused_blocks.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            if len(block_table) > 1:
                prefix = self.blocks[block_table[-2]].hash
            else:
                prefix = -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
