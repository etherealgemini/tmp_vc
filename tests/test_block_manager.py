import unittest
import importlib.util
import hashlib
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
nanovllm_pkg = types.ModuleType("nanovllm")
engine_pkg = types.ModuleType("nanovllm.engine")
sys.modules.setdefault("nanovllm", nanovllm_pkg)
sys.modules.setdefault("nanovllm.engine", engine_pkg)

if "xxhash" not in sys.modules:
    xxhash_module = types.ModuleType("xxhash")

    class _XXH64:
        def __init__(self):
            self._h = hashlib.blake2b(digest_size=8)

        def update(self, data):
            self._h.update(data)

        def intdigest(self):
            return int.from_bytes(self._h.digest(), "little")

    xxhash_module.xxh64 = _XXH64
    sys.modules["xxhash"] = xxhash_module

if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")

    class _Array:
        def __init__(self, values):
            self._values = values

        def tobytes(self):
            return b"".join(int(v).to_bytes(8, "little", signed=True) for v in self._values)

    numpy_module.array = _Array
    sys.modules["numpy"] = numpy_module

sampling_params_spec = importlib.util.spec_from_file_location(
    "nanovllm.sampling_params", ROOT / "nanovllm" / "sampling_params.py"
)
sampling_params_module = importlib.util.module_from_spec(sampling_params_spec)
sampling_params_spec.loader.exec_module(sampling_params_module)
sys.modules["nanovllm.sampling_params"] = sampling_params_module

sequence_spec = importlib.util.spec_from_file_location(
    "nanovllm.engine.sequence", ROOT / "nanovllm" / "engine" / "sequence.py"
)
sequence_module = importlib.util.module_from_spec(sequence_spec)
sequence_spec.loader.exec_module(sequence_module)
sys.modules["nanovllm.engine.sequence"] = sequence_module

block_manager_spec = importlib.util.spec_from_file_location(
    "nanovllm.engine.block_manager", ROOT / "nanovllm" / "engine" / "block_manager.py"
)
block_manager_module = importlib.util.module_from_spec(block_manager_spec)
block_manager_spec.loader.exec_module(block_manager_module)
sys.modules["nanovllm.engine.block_manager"] = block_manager_module

BlockManager = block_manager_module.BlockManager
Sequence = sequence_module.Sequence


class BlockManagerImageAllocationTest(unittest.TestCase):

    def setUp(self):
        self._orig_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self._orig_block_size

    def test_can_allocate_with_image_reuse_from_used_block(self):
        image_token_id = 151655
        manager = BlockManager(num_blocks=5, block_size=4)

        seq_kept = Sequence([1, 2, 3, 4, image_token_id, image_token_id, image_token_id, image_token_id, 9, 9, 9, 9], image_hashes=[12345])
        manager.allocate(seq_kept)

        seq_new = Sequence([5, 6, 7, 8, image_token_id, image_token_id, image_token_id, image_token_id, 0, 0, 0, 0], image_hashes=[12345])
        self.assertTrue(manager.can_allocate(seq_new))

        manager.allocate(seq_new)
        self.assertEqual(seq_new.block_table[1], seq_kept.block_table[1])
        self.assertIn(1, seq_new.image_reused_blocks)

    def test_can_allocate_false_when_required_free_blocks_insufficient(self):
        image_token_id = 151655
        manager = BlockManager(num_blocks=4, block_size=4)

        seq_kept = Sequence([1, 2, 3, 4, image_token_id, image_token_id, image_token_id, image_token_id, 9, 9, 9, 9], image_hashes=[12345])
        manager.allocate(seq_kept)

        seq_new = Sequence([5, 6, 7, 8, image_token_id, image_token_id, image_token_id, image_token_id, 0, 0, 0, 0], image_hashes=[12345])
        self.assertFalse(manager.can_allocate(seq_new))


if __name__ == "__main__":
    unittest.main()
