# Dataset

mesh = (5023, 3)

- `subj_seq_to_idx.pkl`: dict('subject name', dict('sequence name', dict('local index', 'global index')))
- `raw_audio_fixed.pkl`: dict('subject name', dict('sequence name', {"sample_rate": 22000, "audio": 一維 ndarray dtype=int16}))
- `templates.pkl`: dict('subject name', mesh)
