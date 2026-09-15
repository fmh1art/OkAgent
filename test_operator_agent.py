import unittest

from operator_agent import dataset_profile, parse_operator, parse_done, parse_code, validate_generated_code


class OperatorAgentTest(unittest.TestCase):
    def test_code_fence_compatibility(self):
        self.assertEqual(parse_code('<code>```python\nprint(1)\n```</code>'), 'print(1)\n')
        self.assertEqual(parse_code('```python\nprint(1)\n```'), 'print(1)\n')
        self.assertEqual(parse_code('print(1)'), 'print(1)\n')

    def test_demo_profile_unchanged(self):
        profile = dataset_profile("full")
        self.assertEqual(profile["row_count"], 3000)
        self.assertEqual(profile["label_counts"], {"0": 2967, "1": 33})

    def test_one_wrapper_and_aliases(self):
        raw = '<operator>{"name":"Partition","input":["full"],"output":["train","test"],"parameters":{"partition_method":"split"}}</operator>'
        spec = parse_operator(raw, 34761)
        self.assertEqual(spec["input"], ["full"])
        with self.assertRaises(ValueError):
            parse_operator(raw + raw, 34761)

    def test_operator_parameters_must_be_object(self):
        for n in [0, [], True]:
            import json
            raw = '<operator>' + json.dumps({"operator": "MyCustomOperation", "params": n}) + '</operator>'
            with self.assertRaises(ValueError):
                parse_operator(raw, 34761)

    def test_done_and_network_validation(self):
        self.assertEqual(parse_done('<done>done</done>'), 'done')
        with self.assertRaises(ValueError):
            validate_generated_code('import requests')


if __name__ == '__main__':
    unittest.main()
