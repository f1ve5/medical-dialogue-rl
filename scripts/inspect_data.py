"""Validate and summarize local data without importing project dependencies."""
import io
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Only built-in data containers are supported.')


def inspect(directory):
    directory = Path(directory)
    diseases = directory.joinpath('disease.txt').read_text(encoding='utf-8').splitlines()
    symptoms = directory.joinpath('symptom.txt').read_text(encoding='utf-8').splitlines()
    summary = {'diseases': len(diseases), 'symptoms': len(symptoms), 'splits': {}}
    for split in ['train', 'dev', 'test']:
        patients = DataOnlyUnpickler(io.BytesIO(directory.joinpath(split + '.pk').read_bytes())).load()
        assert isinstance(patients, list) and patients, 'Expected a nonempty list of patients'
        for patient in patients:
            assert isinstance(patient, dict)
            if split != 'test':
                assert patient['disease_tag'] in diseases
            for field in ['explicit_inform_slots', 'implicit_inform_slots']:
                assert isinstance(patient[field], dict)
                assert all(s in symptoms and v in (0, 1, False, True) for s, v in patient[field].items())
        summary['splits'][split] = {'samples': len(patients),
                                   'labeled': sum('disease_tag' in p for p in patients)}
    return summary


if __name__ == '__main__':
    print(json.dumps(inspect(ROOT / 'dataset'), indent=2))
