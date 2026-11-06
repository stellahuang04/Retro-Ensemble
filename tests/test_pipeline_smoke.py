"""Offline integration: real prompts/JSON parsing/output; mocked LLM and tokenizer only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import batch_run_moa_testset as moa
from tools.merge_batch_predictions import clean_merged_candidates, add_candidate_ids
from tools.canonicalize_batch import canonicalize_precursors


class FakeClient:
    def complete_content_only(self, **kwargs):
        # Include an out-of-pool ID and duplicate ID to verify rank preservation.
        return json.dumps({'selected_candidate_ids': ['C0001', 'OUTSIDE', 'C0001', 'C0002'],
                           'decision_trace': ['Synthetic test reasoning'], 'reason': 'Fixture'})


class PipelineSmoke(unittest.TestCase):
    def test_moa_modes_preserve_raw_slots(self):
        for reasoning in (True, False):
            with self.subTest(reasoning=reasoning), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = []
                for model in ('augmented_transformer', 'graph2smiles', 'template_relevance', 'retrosim'):
                    for rank, candidate in enumerate(('CC.O', 'CCN'), 1):
                        rows.append(dict(sample_index=0, id='fixture', target='CCO', model=model,
                                         rank=rank, score=0.5, raw_candidate=candidate,
                                         canonical_candidate=canonicalize_precursors(candidate)))
                merged = add_candidate_ids(clean_merged_candidates(pd.DataFrame(rows)))
                merged.to_csv(root / 'merged.csv', index=False)
                with patch('tools.llm_batch_selector.get_llm_client', return_value=FakeClient()), \
                     patch.object(moa, 'count_chat_tokens', return_value=100), \
                     patch.object(moa, 'count_chat_characters', return_value=1000):
                    moa.run_moa_batch(root/'merged.csv', root/'results.csv', root,
                                      agent_providers=['cerebras', 'cerebras'],
                                      agent_models=['fixture-a', 'fixture-b'],
                                      aggregator_provider='nchc', aggregator_model='fixture-c',
                                      include_reasoning=reasoning)
                result = pd.read_csv(root/'results.csv', keep_default_na=False).iloc[0]
                self.assertEqual(result.status, 'OK', result.error_message)
                self.assertEqual(result.top_1, 'CC.O')
                self.assertTrue(result.top_2.startswith('__INVALID_CANDIDATE_ID__:'))
                self.assertEqual(result.top_3, 'CC.O')
                self.assertEqual(result.top_4, 'CCN')
                payload = json.loads((root/'samples/0_fixture/aggregator_input.json').read_text())
                self.assertEqual(len(payload['agent_responses']), 2)
                for agent in payload['agent_responses']:
                    response = json.loads(agent['response'])
                    self.assertEqual(response['selected_candidate_ids'][1], 'OUTSIDE')
                    self.assertEqual('reason' in response, reasoning)
                self.assertEqual(bool(result.reason), reasoning)


if __name__ == '__main__':
    unittest.main()
