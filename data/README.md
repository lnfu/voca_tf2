# Dataset

mesh = (5023, 3)

- `audio/raw_audio_fixed.pkl`: dict('subject name', dict('sequence name', {"sample_rate": 22000, "audio": 一維 ndarray dtype=int16}))
- `subj_seq_to_idx.pkl`: dict('subject name', dict('sequence name', dict('local index', 'global index')))
- `templates.pkl`: dict('subject name', mesh)
- `data_verts.npy`
- `flame2023.pkl`: https://flame.is.tue.mpg.de/download.php
- `output_graph.pb`: https://github.com/mozilla/DeepSpeech/releases/tag/v0.1.0
