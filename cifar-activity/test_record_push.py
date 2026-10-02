"""Check streamed checkpoint loading against a conventional model load."""
import tempfile
import unittest
from pathlib import Path

import timm
import torch
from safetensors.torch import save_file

from record_push import ROOT, intermediate_features, load_timm_streamed, save_json


class StreamedLoaderTest(unittest.TestCase):
    def test_siglip_features_without_class_token(self):
        torch.set_num_threads(4)
        torch.manual_seed(123)
        name = 'vit_base_patch16_siglip_224.v2_webli'
        reference = timm.create_model(name, pretrained=False, num_classes=0,
                                      dynamic_img_size=True).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_file(reference.state_dict(), str(path))
            loaded = load_timm_streamed(name, path, device='cpu')
        reference.to(torch.bfloat16)
        for key, value in loaded.state_dict().items():
            self.assertTrue(torch.equal(reference.state_dict()[key], value), key)
        x = torch.randn(2, 3, 224, 224, dtype=torch.bfloat16)
        with torch.inference_mode():
            actual = intermediate_features(loaded, x, name)
            expected = intermediate_features(reference, x, name)
        self.assertEqual(tuple(actual.shape), (2, 5, 768))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float(), expected.float(), atol=.002, rtol=.002)
        save_json(ROOT / 'siglip-feature-smoke.json', {
            'passed': True, 'model': name, 'shape': list(actual.shape),
            'class_token': False, 'normalization': reference.pretrained_cfg['mean'],
            'max_feature_difference': float((actual.float() - expected.float()).abs().max()),
            'accuracy_claim': False,
        })

    def test_parameters_buffers_and_inference(self):
        torch.set_num_threads(4)
        torch.manual_seed(123)
        name = 'vit_tiny_patch16_dinov3_qkvb'
        reference = timm.create_model(name, pretrained=False, num_classes=0,
                                      dynamic_img_size=True, global_pool='token').eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            save_file(reference.state_dict(), str(path))
            loaded = load_timm_streamed(name, path, device='cpu')
        reference.to(torch.bfloat16)
        for key, value in loaded.state_dict().items():
            self.assertTrue(torch.equal(reference.state_dict()[key], value), key)
        self.assertTrue(torch.equal(reference.rope.periods, loaded.rope.periods))
        reference_buffers = dict(reference.named_buffers())
        for key, value in loaded.named_buffers():
            self.assertTrue(torch.equal(reference_buffers[key], value), key)
        x = torch.randn(2, 3, 224, 224, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected, actual = reference(x), loaded(x)
        # Different parameter layouts can select different BF16 CPU kernels.
        torch.testing.assert_close(actual.float(), expected.float(), atol=.002, rtol=.002)
        ROOT.mkdir(parents=True, exist_ok=True)
        save_json(ROOT / 'streamed-loader-smoke.json', {
            'model': name, 'device': 'cpu', 'dtype': 'bfloat16',
            'max_output_difference': float((actual.float() - expected.float()).abs().max()),
            'parameters_and_rope_periods_identical': True,
            'passed': True, 'accuracy_claim': False,
        })


if __name__ == '__main__':
    unittest.main()
