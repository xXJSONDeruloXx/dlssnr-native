import os
from pathlib import Path
import tempfile
import unittest
from native.model import read_model


class ModelTests(unittest.TestCase):
    def test_unrecognized_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input'
            path.write_bytes(b'not the required model')
            with self.assertRaisesRegex(ValueError, 'hash'):
                read_model(path)

    @unittest.skipUnless(os.environ.get('DLSSNR_TEST_MODEL'), 'optional private model fixture')
    def test_pinned_model_has_exact_tensor_payload(self):
        tensors = read_model(os.environ['DLSSNR_TEST_MODEL'])
        self.assertEqual(len(tensors), 153)
        self.assertEqual(sum(len(t.data) for t in tensors), 147683778)
