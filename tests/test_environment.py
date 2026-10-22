"""Regression checks use synthetic patients, never private course datasets."""
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from env import environment
from env_diag import environment_diag
from evaluation import evaluate_agent


class StubAgent:
    def choose_diagnosis(self, state):
        return np.zeros(len(state), dtype=int), np.tile([0.6, 0.4], (len(state), 1))

    def choose_action_s(self, state):
        return np.ones(len(state), dtype=int)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.old_directory = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        Path('dataset').mkdir()
        Path('dataset/disease.txt').write_text('A\nB\n', encoding='utf-8')
        Path('dataset/symptom.txt').write_text('fever\ncough\n', encoding='utf-8')
        self.patients = [
            {'explicit_inform_slots': {'fever': 1}, 'implicit_inform_slots': {'cough': 1}, 'disease_tag': d}
            for d in ['A', 'A', 'B']
        ]
        Path('dataset/dev.pk').write_bytes(pickle.dumps(self.patients))
        self.args = SimpleNamespace(mu=1, nu=2.5)

    def tearDown(self):
        os.chdir(self.old_directory)
        self.temp.cleanup()

    def test_diagnosis_batches_advance_and_keep_partial_batch(self):
        env = environment_diag('dataset/dev.pk', self.args)
        first, labels = env.initialize_state(2)
        self.assertEqual(labels.tolist(), [0, 0])
        last, labels = env.initialize_state(2)
        self.assertEqual(last.shape, (1, 2))
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(env.batch_index.tolist(), [2])
        self.assertEqual(first[:, 1].tolist(), [0, 0])
        self.assertEqual(env.all_state.tolist(), [[1, 1]])

    def test_state_encoding_and_terminal_patients_are_frozen(self):
        env = environment('dataset/dev.pk', self.args)
        state, _ = env.initialize_state(2)
        done = np.array([True, False])
        output = env.step(state, np.ones(2, dtype=int), done, np.zeros(2, dtype=bool),
                          StubAgent(), np.zeros(2), np.zeros(2), np.zeros(2))
        self.assertEqual(output[0].tolist(), [[1, 0], [1, 1]])
        self.assertEqual(output[1][0], 0)
        self.assertTrue(np.isfinite(output[1]).all())

    def test_sample_weighted_metrics_include_final_sample(self):
        env = environment('dataset/dev.pk', self.args)
        accuracy, steps, positive, _ = evaluate_agent(env, StubAgent(), 2, 1, np.zeros(2))
        self.assertAlmostEqual(accuracy, 2 / 3)
        self.assertEqual(steps, 1)
        self.assertEqual(positive, 1)

    def test_correct_diagnosis_terminates_and_rewards(self):
        env = environment_diag('dataset/dev.pk', self.args)
        state, _ = env.initialize_state(2)
        output = env.step(state, np.array([2, 3]), np.zeros(2, dtype=bool),
                          np.zeros(2, dtype=bool), StubAgent())
        self.assertEqual(output[1].tolist(), [10, -10])
        self.assertEqual(output[2].tolist(), [True, True])
        self.assertEqual(output[3].tolist(), [True, False])

    def test_test_split_can_omit_labels(self):
        for patient in self.patients:
            patient.pop('disease_tag')
        Path('dataset/test.pk').write_bytes(pickle.dumps(self.patients))
        _, labels = environment('dataset/test.pk', self.args).initialize_state(3)
        self.assertEqual(labels.tolist(), [-1, -1, -1])


if __name__ == '__main__':
    unittest.main()
