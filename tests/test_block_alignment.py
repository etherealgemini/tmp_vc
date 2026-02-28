"""Tests for image-token block-alignment padding and KV cache block allocation."""
import pickle
import sys
import os
import importlib.util

# ---- direct file-based imports (avoids nanovllm/__init__.py & heavy dependencies) ----
_ROOT = os.path.join(os.path.dirname(__file__), "..")

def _load_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Stub out the nanovllm package so sub-modules can import each other.
import types
sys.modules.setdefault("nanovllm", types.ModuleType("nanovllm"))
sys.modules.setdefault("nanovllm.engine", types.ModuleType("nanovllm.engine"))

_sp_mod = _load_module("nanovllm.sampling_params", "nanovllm/sampling_params.py")
_seq_mod = _load_module("nanovllm.engine.sequence", "nanovllm/engine/sequence.py")
_bm_mod = _load_module("nanovllm.engine.block_manager", "nanovllm/engine/block_manager.py")

IMAGE_TOKEN_ID = _seq_mod.IMAGE_TOKEN_ID
PAD_TOKEN_ID = _seq_mod.PAD_TOKEN_ID
Sequence = _seq_mod.Sequence
_compute_image_token_ranges = _seq_mod._compute_image_token_ranges
_pad_for_block_alignment = _seq_mod._pad_for_block_alignment
BlockManager = _bm_mod.BlockManager

# Use a small block size for readable tests.
BLOCK = 16


# ---- helpers ---------------------------------------------------------------

def _img(n: int) -> list[int]:
    """Return *n* image-placeholder token IDs."""
    return [IMAGE_TOKEN_ID] * n


def _txt(n: int, start: int = 1) -> list[int]:
    """Return *n* distinct non-special text token IDs."""
    return list(range(start, start + n))


# ---- _pad_for_block_alignment tests ----------------------------------------

class TestPadForBlockAlignment:
    """Unit tests for the padding function."""

    def test_no_images(self):
        """Without images, token_ids should be returned unchanged."""
        tokens = _txt(10)
        padded, ranges, postpad_ranges = _pad_for_block_alignment(tokens, [], BLOCK)
        assert padded == tokens
        assert ranges == []
        assert postpad_ranges == []

    def test_single_image_aligned_start(self):
        """Image starting on a block boundary needs no pre-padding."""
        text = _txt(BLOCK)  # fills exactly one block
        img = _img(BLOCK)   # fills exactly one block
        tokens = text + img
        raw_ranges = _compute_image_token_ranges(tokens, [0xABC])
        padded, pranges, postpad_ranges = _pad_for_block_alignment(tokens, raw_ranges, BLOCK)

        # No padding needed – already aligned.
        assert padded == tokens
        assert len(pranges) == 1
        start, end, h = pranges[0]
        assert start == BLOCK
        assert end == 2 * BLOCK
        assert h == 0xABC
        # Image is exactly block-aligned, so no post-padding.
        assert postpad_ranges == []

    def test_single_image_unaligned_start(self):
        """Image NOT on a block boundary must be padded before."""
        text = _txt(10)          # 10 tokens, not block-aligned
        img = _img(BLOCK)        # 16 image tokens (fits 1 block)
        trailing = _txt(3, 100)  # some text after
        tokens = text + img + trailing
        raw_ranges = _compute_image_token_ranges(tokens, [0xABC])
        padded, pranges, postpad_ranges = _pad_for_block_alignment(tokens, raw_ranges, BLOCK)

        # Pre-padding: 16 - 10 = 6 PAD tokens before image
        expected_pre_pad = BLOCK - 10
        assert padded[:10] == text
        assert padded[10:BLOCK] == [PAD_TOKEN_ID] * expected_pre_pad

        # Image tokens occupy block 1 entirely
        start, end, h = pranges[0]
        assert start == BLOCK
        assert end == 2 * BLOCK
        assert padded[start:end] == img

        # Trailing text follows immediately (no extra padding after image)
        assert padded[end:end + 3] == trailing
        # Image is exactly block-aligned (BLOCK tokens), so no post-padding.
        assert postpad_ranges == []

    def test_image_not_filling_block(self):
        """Image whose token count is not a multiple of block_size gets post-padding."""
        text = _txt(BLOCK)
        img = _img(20)  # 20 tokens → needs 2 blocks (16 + 4 → pad 12)
        tokens = text + img
        raw_ranges = _compute_image_token_ranges(tokens, [0xDEF])
        padded, pranges, postpad_ranges = _pad_for_block_alignment(tokens, raw_ranges, BLOCK)

        start, end, h = pranges[0]
        assert start == BLOCK            # image starts at block boundary
        assert end == 3 * BLOCK          # padded to fill 2 blocks
        assert (end - start) % BLOCK == 0  # aligned
        # First image block: 16 IMAGE tokens
        assert padded[start:start + BLOCK] == _img(BLOCK)
        # Second image block: 4 IMAGE tokens + 12 PAD tokens
        assert padded[start + BLOCK:start + BLOCK + 4] == _img(4)
        assert padded[start + BLOCK + 4:end] == [PAD_TOKEN_ID] * 12
        # Post-padding covers the 12 PAD tokens at the end of the last image block.
        assert len(postpad_ranges) == 1
        ps, pe = postpad_ranges[0]
        assert ps == start + BLOCK + 4   # real image tokens end here
        assert pe == end                 # post-pad extends to block boundary

    def test_two_images_separated_by_text(self):
        """Two images separated by text should each get their own block range."""
        text1 = _txt(5)
        img1 = _img(BLOCK)
        text2 = _txt(7, 50)
        img2 = _img(BLOCK)
        text3 = _txt(4, 100)
        tokens = text1 + img1 + text2 + img2 + text3
        raw_ranges = _compute_image_token_ranges(tokens, [0xA, 0xB])
        padded, pranges, postpad_ranges = _pad_for_block_alignment(tokens, raw_ranges, BLOCK)

        # Image 1
        s1, e1, h1 = pranges[0]
        assert s1 % BLOCK == 0
        assert e1 % BLOCK == 0
        assert h1 == 0xA

        # Image 2
        s2, e2, h2 = pranges[1]
        assert s2 % BLOCK == 0
        assert e2 % BLOCK == 0
        assert h2 == 0xB

        # Ranges don't overlap
        assert e1 <= s2
        # Both images are exactly BLOCK tokens, so no post-padding expected.
        assert postpad_ranges == []

    def test_adjacent_images(self):
        """Two images with no text in between still get separate block ranges."""
        img1 = _img(BLOCK)
        img2 = _img(BLOCK)
        tokens = img1 + img2
        raw_ranges = _compute_image_token_ranges(tokens, [0xA, 0xB])

        # _compute_image_token_ranges sees one contiguous run of IMAGE_TOKEN_IDs
        # (all the same token value), so it produces only 1 range consuming hash 0xA.
        assert len(raw_ranges) == 1

        padded, pranges, postpad_ranges = _pad_for_block_alignment(tokens, raw_ranges, BLOCK)
        assert len(pranges) == 1
        for s, e, _ in pranges:
            assert s % BLOCK == 0
            assert e % BLOCK == 0
        # Combined image is exactly 2*BLOCK tokens, no post-padding needed.
        assert postpad_ranges == []

    def test_image_at_start(self):
        """Image at position 0 should need no pre-padding."""
        img = _img(BLOCK)
        text = _txt(5)
        tokens = img + text
        raw_ranges = _compute_image_token_ranges(tokens, [0x1])
        padded, pranges, postpad_ranges = _pad_for_block_alignment(tokens, raw_ranges, BLOCK)

        start, end, _ = pranges[0]
        assert start == 0  # no pre-padding needed
        assert end == BLOCK
        # Image is exactly BLOCK tokens, no post-padding.
        assert postpad_ranges == []

    def test_image_at_end(self):
        """Image at the end with post-padding still aligns."""
        text = _txt(10)
        img = _img(20)
        tokens = text + img
        raw_ranges = _compute_image_token_ranges(tokens, [0x2])
        padded, pranges, postpad_ranges = _pad_for_block_alignment(tokens, raw_ranges, BLOCK)

        start, end, _ = pranges[0]
        assert start % BLOCK == 0
        assert end % BLOCK == 0
        assert end == len(padded)  # nothing after image (including padding)
        # 20 image tokens → last block has 4 real + 12 post-pad.
        assert len(postpad_ranges) == 1
        ps, pe = postpad_ranges[0]
        assert ps == start + 20   # real image ends after 20 tokens
        assert pe == end


# ---- Sequence integration tests ---------------------------------------------

class TestSequenceBlockAlignment:
    """Verify that Sequence applies padding and that BlockManager sees pure blocks."""

    @staticmethod
    def _make_seq(token_ids, image_hashes=None, block_size=BLOCK):
        old_bs = Sequence.block_size
        Sequence.block_size = block_size
        seq = Sequence(token_ids, image_hashes=image_hashes)
        Sequence.block_size = old_bs
        # Pin block_size as an instance attribute so it survives class-var restore.
        seq.block_size = block_size
        return seq

    def test_image_blocks_are_pure(self):
        """Every block inside an image range must contain only IMAGE or PAD tokens."""
        text = _txt(10)
        img = _img(20)
        trailing = _txt(3, 100)
        token_ids = text + img + trailing
        seq = self._make_seq(token_ids, image_hashes=[0xCAFE], block_size=BLOCK)

        for start, end, _ in seq.image_token_ranges:
            for bi in range(start // BLOCK, end // BLOCK):
                block_tokens = seq.block(bi)
                for tok in block_tokens:
                    assert tok in (IMAGE_TOKEN_ID, PAD_TOKEN_ID), (
                        f"Block {bi} contains unexpected token {tok}"
                    )

    def test_no_padding_without_images(self):
        """A text-only sequence should have no padding overhead."""
        tokens = _txt(30)
        seq = self._make_seq(tokens, block_size=BLOCK)
        assert len(seq) == 30
        assert seq.token_ids == tokens


# ---- BlockManager cache reuse tests -----------------------------------------

class TestBlockManagerImageReuse:
    """Ensure that the same image in different text contexts produces reusable blocks."""

    def test_same_image_different_text_reuses_image_blocks(self):
        """Image blocks should be reused across requests with different text prefixes."""
        BS = BLOCK
        NUM_BLOCKS = 64
        bm = BlockManager(NUM_BLOCKS, BS)
        old_bs = Sequence.block_size
        Sequence.block_size = BS
        img_hash = 0xBEEF
        # Request 1: short text + image
        tokens1 = _txt(5) + _img(BS)
        seq1 = Sequence(tokens1, image_hashes=[img_hash])
        seq1.block_size = BS
        bm.allocate(seq1)

        # Identify image block IDs from seq1.
        img_start1 = seq1.image_token_ranges[0][0]
        img_end1 = seq1.image_token_ranges[0][1]
        img_block_indices1 = list(range(img_start1 // BS, img_end1 // BS))
        img_block_ids1 = [seq1.block_table[i] for i in img_block_indices1]

        # Request 2: DIFFERENT text prefix + SAME image
        tokens2 = _txt(10, start=500) + _img(BS)
        seq2 = Sequence(tokens2, image_hashes=[img_hash])
        seq2.block_size = BS
        bm.allocate(seq2)

        img_start2 = seq2.image_token_ranges[0][0]
        img_end2 = seq2.image_token_ranges[0][1]
        img_block_indices2 = list(range(img_start2 // BS, img_end2 // BS))
        img_block_ids2 = [seq2.block_table[i] for i in img_block_indices2]

        # Image blocks must be the SAME physical blocks (reused).
        assert img_block_ids1 == img_block_ids2, (
            f"Expected image blocks to be reused: {img_block_ids1} vs {img_block_ids2}"
        )

        # Image blocks should be accounted as cached or reused in seq2.
        # (Either num_cached_tokens covers them or they are in image_reused_blocks.)
        total_image_tokens = img_end2 - img_start2
        image_reused_tokens = len(seq2.image_reused_blocks) * BS
        assert seq2.num_cached_tokens >= total_image_tokens or image_reused_tokens >= total_image_tokens
        Sequence.block_size = old_bs

    def test_text_only_no_regression(self):
        """Text-only allocation should work exactly as before."""
        BS = BLOCK
        NUM_BLOCKS = 32
        bm = BlockManager(NUM_BLOCKS, BS)
        old_bs = Sequence.block_size
        Sequence.block_size = BS
        tokens = _txt(40)
        seq = Sequence(tokens)
        seq.block_size = BS
        bm.allocate(seq)
        assert len(seq.block_table) == seq.num_blocks
        Sequence.block_size = old_bs


# ---- Postpad KV-slot masking tests ------------------------------------------

class TestPostpadSlotMasking:
    """Verify that image post-padding positions are tracked and that the
    KV cache is equivalent before and after image-block reuse."""

    @staticmethod
    def _make_seq(token_ids, image_hashes=None, block_size=BLOCK):
        old_bs = Sequence.block_size
        Sequence.block_size = block_size
        seq = Sequence(token_ids, image_hashes=image_hashes)
        Sequence.block_size = old_bs
        seq.block_size = block_size
        return seq

    def test_no_postpad_when_image_block_aligned(self):
        """If the image length is already a multiple of block_size, there is no
        post-padding and image_postpad_ranges must be empty."""
        tokens = _txt(BLOCK) + _img(BLOCK)
        seq = self._make_seq(tokens, image_hashes=[0xABC])
        assert seq.image_postpad_ranges == []
        assert seq.image_postpad_count == 0
        assert seq.num_effective_tokens == seq.num_tokens

    def test_postpad_tracked_correctly(self):
        """Post-padding tokens at the end of the last image block are tracked."""
        # 20 image tokens → last block has 4 real + 12 postpad (BLOCK=16)
        tokens = _txt(BLOCK) + _img(20)
        seq = self._make_seq(tokens, image_hashes=[0xDEF])
        assert len(seq.image_postpad_ranges) == 1
        ps, pe = seq.image_postpad_ranges[0]
        assert pe - ps == BLOCK - 4   # 12 postpad tokens
        assert seq.image_postpad_count == 12
        assert seq.num_effective_tokens == seq.num_tokens - 12

    def test_postpad_positions_in_last_image_block(self):
        """Post-pad positions must fall at the end of the last image block."""
        tokens = _txt(BLOCK) + _img(20)
        seq = self._make_seq(tokens, image_hashes=[0xDEF])
        # image_token_ranges[0] gives the padded range [16, 48)
        img_start, img_end, _ = seq.image_token_ranges[0]
        ps, pe = seq.image_postpad_ranges[0]
        # Postpad is inside the image range and at its very end.
        assert img_start <= ps < pe == img_end
        # All tokens in postpad range should be PAD_TOKEN_ID.
        for pos in range(ps, pe):
            assert seq.token_ids[pos] == PAD_TOKEN_ID, (
                f"Expected PAD at position {pos}, got {seq.token_ids[pos]}"
            )

    def test_two_images_postpad(self):
        """Each image with non-aligned length produces its own postpad range."""
        # img1: 20 tokens → postpad 12;  img2: 18 tokens → postpad 14
        tokens = _txt(BLOCK) + _img(20) + _txt(5, 100) + _img(18)
        seq = self._make_seq(tokens, image_hashes=[0xA, 0xB])
        assert len(seq.image_postpad_ranges) == 2
        # First image postpad.
        ps1, pe1 = seq.image_postpad_ranges[0]
        assert pe1 - ps1 == BLOCK - 4   # 12
        # Second image postpad.
        ps2, pe2 = seq.image_postpad_ranges[1]
        assert pe2 - ps2 == BLOCK - 2   # 14
        assert seq.image_postpad_count == 12 + 14

    def test_num_effective_tokens_no_images(self):
        """Text-only sequences have no postpad; effective == total tokens."""
        tokens = _txt(30)
        seq = self._make_seq(tokens)
        assert seq.image_postpad_ranges == []
        assert seq.num_effective_tokens == len(seq)

    def test_serialization_preserves_postpad_ranges(self):
        """image_postpad_ranges survives a __getstate__ / __setstate__ round-trip."""
        tokens = _txt(BLOCK) + _img(20)
        seq = self._make_seq(tokens, image_hashes=[0x123])
        serialized = pickle.dumps(seq.__getstate__())
        restored = Sequence.__new__(Sequence)
        restored.__setstate__(pickle.loads(serialized))
        assert restored.image_postpad_ranges == seq.image_postpad_ranges


# ---- run with pytest --------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
