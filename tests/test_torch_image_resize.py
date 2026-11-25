import unittest

import torch

from openpi.shared.image_tools import resize_with_pad_torch


class TorchResizeShapeTest(unittest.TestCase):
    def test_singleton_batch_and_unbatched_inputs(self):
        for source, expected in [
            ((1, 16, 24, 3), (1, 8, 8, 3)),
            ((16, 24, 3), (8, 8, 3)),
            ((1, 3, 16, 24), (1, 3, 8, 8)),
            ((3, 16, 24), (3, 8, 8)),
        ]:
            with self.subTest(source=source):
                self.assertEqual(tuple(resize_with_pad_torch(torch.zeros(source), 8, 8).shape), expected)
