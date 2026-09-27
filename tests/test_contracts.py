import tempfile
import unittest
from pathlib import Path
from qa_core import *


class Contracts(unittest.TestCase):
    def example(self, ident="a"):
        return {"id":ident,"question":"Which color?","context":"The cube is blue.","answers":{"text":["blue"],"answer_start":[12]}}

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): validate_examples([self.example(), self.example()])

    def test_batch_independent_ids(self):
        examples = [self.example(str(i)) for i in range(35)]
        self.assertEqual(len(validate_examples(examples)), 35)

    def test_character_labels_and_out_of_window(self):
        offsets=[None,(0,3),(4,8),(9,11),(12,16),(16,17)]
        self.assertEqual(span_labels(offsets,12,"blue"),(4,4))
        self.assertEqual(span_labels(offsets[:3],12,"blue"),(0,0))
        self.assertEqual(shift_label(4,50,512),54)
        self.assertEqual(shift_label(0,50,512),50)

    def test_joint_span_uses_context_and_original_text(self):
        f={"example_id":"a","offset_mapping":[None,(0,3),(4,8),(9,11),(12,16),(16,17)]}
        result=best_spans([self.example()],[f],[[999,0,0,0,9,0]],[[999,0,0,0,9,0]])
        self.assertEqual(result,[{"id":"a","prediction_text":"blue"}])

    def test_original_reference_coordinates(self):
        ex=self.example();ex["answers"]["answer_start"]=[4]
        with self.assertRaises(ValueError):validate_examples([ex])

    def test_missing_feature_rejected(self):
        with self.assertRaises(ValueError):best_spans([self.example()],[],[],[])

    def test_checkpoint_integrity_and_prompt_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"state.bin";p.write_bytes(b"synthetic checkpoint")
            write_checkpoint_manifest(p,{"prompt_length":50})
            self.assertEqual(verify_checkpoint(p,50)["prompt_length"],50)
            with self.assertRaises(ValueError):verify_checkpoint(p,0)
            p.write_bytes(b"changed state")
            with self.assertRaises(ValueError):verify_checkpoint(p,50)


if __name__ == "__main__":unittest.main()
