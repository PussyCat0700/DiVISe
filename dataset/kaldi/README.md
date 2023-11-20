Only needed when you have installed kaldi and wanna extract kaldi pitch as model input.

Note that torchaudio.functional supported compute_kaldi_pitch only for a short period of time in the past and this function has been removed in recent torch versions. I recommend setting up a new enviornment for torchauio==0.9.0 separately for preprocessing kaldi pitch.