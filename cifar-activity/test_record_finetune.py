"""Check adapter gradient scope and CIFAR augmentation on a tiny CPU backbone."""
import copy
import unittest

import timm
import torch
import torch.nn.functional as F

from record_finetune import Adapter, checkpoint, preprocess


class AdapterTest(unittest.TestCase):
    def test_trainable_scope_and_checkpoint(self):
        torch.set_num_threads(4)
        torch.manual_seed(2026)
        name = 'vit_tiny_patch16_dinov3_qkvb'
        backbone = timm.create_model(name, pretrained=False, num_classes=0,
                                     global_pool='token', dynamic_img_size=True)
        dim = backbone.embed_dim * 5
        head = {'indices': [0, 1, 2, 3, 4], 'mean': torch.zeros(dim),
                'scale': torch.ones(dim), 'w': torch.randn(dim + 1, 10) * .001}
        model = Adapter(backbone, head)
        x = torch.randn(2, 3, 64, 64)
        logits = model(x, name)
        loss = F.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        for i, block in enumerate(model.backbone.blocks):
            for parameter in block.parameters():
                self.assertEqual(parameter.requires_grad, i >= len(model.backbone.blocks) - 4)
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                else:
                    self.assertIsNone(parameter.grad)
        saved = checkpoint(model)
        self.assertTrue(all(not key.startswith('backbone.blocks.0.') for key in saved))
        restored = copy.deepcopy(model)
        restored.load_state_dict(saved, strict=False)
        model.eval(); restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x, name), restored(x, name))

    def test_augmentation_shape_and_finiteness(self):
        images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        output = preprocess(images, augment=True, device='cpu')
        self.assertEqual(tuple(output.shape), (4, 3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
