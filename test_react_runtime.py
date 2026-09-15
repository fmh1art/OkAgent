import json
import tempfile
import unittest
from pathlib import Path

from react_runtime import run_react, parse_action, execute_generated


class LoopTest(unittest.TestCase):
    def test_python_boolean_compatibility(self):
        _, action = parse_action('<operator>{"operator":"Custom","params":{"early_stopping":True}}</operator>')
        self.assertTrue(action['params']['early_stopping'])
        with self.assertRaises((ValueError, SyntaxError)):
            parse_action('<operator>{"operator":"Custom","params":dict()}</operator>')

    def test_custom_repeated_operators_beyond_twelve_and_feedback(self):
        class Planner:
            def next_operator(self, context):
                n = len(context['history'])
                if n:
                    assert context['history'][-1]['observation']['quality'] == n
                if n == 16:
                    return '<done>finished after observing round 16</done>'
                return '<operator>{"operator":"TryDifferentProxy","params":{"trial":%s}}</operator>' % n
        class Coder:
            def write_code(self, operator_xml, context):
                return 'code'
        def executor(code, payload, directory, timeout):
            return {'status': 'ok', 'quality': len(payload['context']['history']) + 1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_react(Planner(), Coder(), {}, root, executor=executor)
            self.assertEqual(result['rounds'], 17)
            traces = [json.loads(l) for l in (root/'trajectory.jsonl').read_text().splitlines()]
            self.assertEqual(len(traces), 17)
            self.assertEqual(sum(r['operator']['operator']=='TryDifferentProxy' for r in traces), 16)
            self.assertTrue((root/'rounds/000016/round.json').exists())

    def test_failure_is_given_to_planner_and_same_operator_runs_again(self):
        class Actor:
            def next_operator(self, context):
                if len(context['history']) == 1:
                    assert context['history'][0]['observation']['status']=='error'
                if len(context['history']) == 2:
                    return '<done>ok</done>'
                return '<operator>{"operator":"Proxy","params":{}}</operator>'
            def write_code(self, text, context):
                if not context['history']:
                    return '<code>print("partial output")\nraise ValueError("fit failed")</code>'
                return '<code>print("{\\"status\\":\\"ok\\"}")</code>'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=run_react(Actor(),Actor(),{},root)
            self.assertEqual(result['rounds'],3)
            self.assertIn('partial output',(root/'rounds/000001/stdout.log').read_text())
            self.assertIn('fit failed',(root/'rounds/000001/stderr.log').read_text())

    def test_list_of_operators_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_action('<operator>[{"operator":"A"},{"operator":"B"}]</operator>')

    def test_interrupted_round_resumes_with_new_number(self):
        class Actor:
            def next_operator(self, context):
                raise KeyboardInterrupt()
        class Resume:
            def next_operator(self, context):
                assert context['history'][0]['status']=='interrupted'
                return '<done>resumed</done>'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                run_react(Actor(),None,{},root)
            result=run_react(Resume(),None,{},root)
            self.assertEqual(result['rounds'],2)
            self.assertTrue((root/'rounds/000001/round.json').exists())


if __name__ == '__main__':
    unittest.main()
