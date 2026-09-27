import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from modeling import PromptedQA
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional for core tests")
class TorchContracts(unittest.TestCase):
    def make_base(self):
        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 8)
                self.qa_outputs = torch.nn.Linear(8, 2)
                self.seen = None

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, **kw):
                self.seen = kw
                emb = kw.get("inputs_embeds")
                if emb is None:
                    emb = self.embedding(kw["input_ids"])
                logits = self.qa_outputs(emb)
                loss = None
                if "start_positions" in kw:
                    loss = torch.nn.functional.cross_entropy(logits[:, :, 0], kw["start_positions"])
                return SimpleNamespace(loss=loss, start_logits=logits[:, :, 0], end_logits=logits[:, :, 1])
        return Base()

    def test_prompt_loss_labels_and_output_coordinates(self):
        base = self.make_base()
        model = PromptedQA(base, 5)
        ids = torch.tensor([[1, 2, 3, 4]])
        loss, start, end = model(ids, torch.ones_like(ids), torch.tensor([2]), torch.tensor([2]))
        self.assertEqual(base.seen["start_positions"].item(), 7)
        self.assertEqual(base.seen["attention_mask"].shape, (1, 9))
        self.assertEqual(start.shape, (1, 4))
        loss.backward()
        self.assertIsNotNone(model.prompt.grad)
        self.assertFalse(base.embedding.weight.requires_grad)

    def test_reload_restores_predictions(self):
        model = PromptedQA(self.make_base(), 5)
        ids = torch.tensor([[1, 2, 3, 4]])
        expected = model(ids, torch.ones_like(ids))[1]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = PromptedQA(self.make_base(), 5)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            self.assertTrue(torch.equal(expected, restored(ids, torch.ones_like(ids))[1]))


if __name__ == "__main__":
    unittest.main()
