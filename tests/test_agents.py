"""Optional PyTorch checks use tiny networks instead of the large policy."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is optional for environment-only checks')
class AgentShapeTests(unittest.TestCase):
    def test_reinforce_singleton_inference_keeps_batch_axis(self):
        from agent import Policy_Gradient_pair_model
        agent = Policy_Gradient_pair_model.__new__(Policy_Gradient_pair_model)
        agent.device = torch.device('cpu')
        agent.policy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Softmax(dim=1))
        agent.classifier = torch.nn.Linear(2, 2)
        state = np.zeros((1, 2))
        self.assertEqual(agent.choose_action_s(state).shape, (1,))
        labels, probabilities = agent.choose_diagnosis(state)
        self.assertEqual(labels.shape, (1,))
        self.assertEqual(probabilities.shape, (1, 2))

    def test_ppo_singleton_update_stays_finite(self):
        from agent_ppo import PPO_agent_model
        agent = PPO_agent_model.__new__(PPO_agent_model)
        agent.device = torch.device('cpu')
        agent.K_epochs = 1
        agent.eps_clip = 0.2
        agent.policy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Softmax(dim=1))
        agent.critic = torch.nn.Linear(2, 1)
        agent.optimizer_p = torch.optim.Adam(agent.policy.parameters())
        agent.optimizer_v = torch.optim.Adam(agent.critic.parameters())
        agent.MseLoss = torch.nn.MSELoss()
        state = np.zeros((1, 2), dtype=np.float32)
        action, log_prob = agent.choose_action_s(state)
        agent.update_param_ppo(state, action, log_prob, np.array([1.0]))
        for network in [agent.policy, agent.critic]:
            self.assertTrue(all(torch.isfinite(p).all().item() for p in network.parameters()))


if __name__ == '__main__':
    unittest.main()
