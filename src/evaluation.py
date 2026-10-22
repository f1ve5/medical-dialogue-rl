"""Sample-weighted evaluation shared by the training entry points."""
import numpy as np
from scipy.stats import entropy


def evaluate_agent(env, agent, batch_size, max_steps, threshold=None, diagnosis_actions=False):
    correct = steps = positives = triggered = count = 0
    while env.idx < env.sample_size:
        state, labels = env.initialize_state(batch_size)
        size = len(state)
        initial = state.copy()
        prediction, probabilities = agent.choose_diagnosis(state)
        initial_entropy = entropy(probabilities, axis=1)
        current_entropy = initial_entropy.copy()
        done = np.zeros(size, dtype=bool) if diagnosis_actions else initial_entropy < threshold[prediction]
        final_prediction = prediction.copy()
        for _ in range(max_steps):
            active = ~done
            if not active.any():
                break
            steps += int(active.sum())
            action = agent.choose_action_s(state)
            if isinstance(action, tuple):
                action = action[0]
            state, _, done, _, stopped, current_entropy, diagnosis = env.step(
                state, action, done, np.zeros(size, dtype=bool), agent,
                initial_entropy, threshold, current_entropy,
            )
            if diagnosis_actions:
                final_prediction[stopped] = diagnosis[stopped]
                triggered += int(stopped.sum())
        fallback, _ = agent.choose_diagnosis(state)
        if diagnosis_actions:
            final_prediction[~done] = fallback[~done]
        else:
            final_prediction = fallback
        correct += int((final_prediction == labels).sum())
        positives += int((state == 1).sum() - (initial == 1).sum())
        count += size
    if count == 0:
        raise ValueError('Cannot evaluate an empty dataset.')
    return correct / count, steps / count, positives / count, triggered / count
