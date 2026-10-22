"""Launch experiments from any working directory using consistent arguments."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['reinforce', 'ppo', 'diagnosis'])
    parser.add_argument('mode', choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=15)
    parser.add_argument('--seed', type=int, default=514)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_steps) < 1:
        parser.error('epochs, batch-size and max-steps must be positive')
    missing = [name for name in ['train.pk', 'dev.pk', 'test.pk', 'disease.txt', 'symptom.txt']
               if not (ROOT / 'dataset' / name).is_file()]
    if missing:
        parser.error('Missing dataset files: ' + ', '.join(missing) + '. See dataset/README.md.')
    if args.mode != 'train' and not list((ROOT / 'model_saved').glob('*.pth')):
        parser.error('No model checkpoints found. Train this model first; weights are not bundled.')
    os.chdir(ROOT)
    (ROOT / 'log').mkdir(exist_ok=True)
    (ROOT / 'model_saved').mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT / 'src'))
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_scripts = {'reinforce': 'main.py', 'ppo': 'main_ppo.py', 'diagnosis': 'main_diag.py'}
    eval_scripts = {'reinforce': 'eval_dev.py', 'ppo': 'eval_dev_ppo.py', 'diagnosis': 'eval_dev_diag.py'}
    script = eval_scripts[args.model] if args.mode == 'evaluate' else train_scripts[args.model]
    trial = {'reinforce': 1, 'ppo': 2, 'diagnosis': 3}[args.model]
    sys.argv = [script, '-device', args.device, '-BATCHSIZE', str(args.batch_size),
                '-MAXSTEP', str(args.max_steps), '-trail', str(trial),
                '-threshold', '1.0', '-mu', '1.0', '-nu', '2.5']
    if args.model != 'ppo':
        sys.argv += ['-seed', str(args.seed)]
    if args.mode == 'train':
        sys.argv += ['-train', '-EPOCHS', str(args.epochs)]
    runpy.run_path(str(ROOT / 'src' / script), run_name='__main__')


if __name__ == '__main__':
    main()
