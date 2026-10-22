# Local dataset

Place these files here before training:

| File | Expected contents | Local count |
| --- | --- | ---: |
| `train.pk` | List of labeled patient dictionaries | 1,912 |
| `dev.pk` | List of labeled patient dictionaries | 239 |
| `test.pk` | List of patient dictionaries without disease labels | 239 |
| `disease.txt` | One disease label per line, in classifier-index order | 12 |
| `symptom.txt` | One symptom per line, in state-index order | 118 |

Each patient contains `explicit_inform_slots` and `implicit_inform_slots`: dictionaries mapping symptom names to Boolean/0/1 values. Labeled splits also contain `disease_tag`. State values are `1` for positive, `-1` for observed negative, and `0` for unknown. The simulator treats unlisted symptoms as negative when queried; it does not model uncertain or noisy responses.

The historical report calls this dataset “MDD”; the supplied files do not establish a precise publication, download URL or license. Do not assume it is the upstream MedlinePlus data simply because that knowledge base was also present in the original folder.

Dataset contents are intentionally ignored by Git pending provenance and redistribution confirmation. The original local files remain available in this working copy. No automatic downloader is supplied because the exact public source is unverified. Never substitute a different split while presenting the historical results as a reproduction.

Run `python scripts/inspect_data.py` to check local structure and counts. That tool rejects pickle class loading. The legacy environment uses ordinary `pickle.load`, so only load trusted local files.
