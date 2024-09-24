GRIFFINLIM = "no_vocoder"
HIFIGAN_NO_GRAD = "vocoder_nogradient"
HIFIGAN_WITH_GRAD = "vocoder_withgradient"
BIGVGAN_NO_GRAD = "vocoder_bigvgan_nogradient"
PWG_NO_GRAD = "vocoder_pwg_nogradient"
# Do not modify values of constants below as they are likely to be correlated to terms in json files.
# This is configured for HiFi-GAN itself. Will only be used in ReVISE.
UNIT_HIFIGAN_NO_GRAD = 'HuBERT Label as input (ReVISE)'
UNIT_SPEECH_TOKENIZER_NO_GRAD = 'SpeechTokenizer as vocoder'
HYBRID_MEL_UNIT_NO_GRAD = 'Hybrid Mel And Unit as input'
UNIT_METHODS = set([UNIT_HIFIGAN_NO_GRAD, UNIT_SPEECH_TOKENIZER_NO_GRAD, HYBRID_MEL_UNIT_NO_GRAD])
MEL_VOCODER_METHODS = set([HIFIGAN_NO_GRAD, HIFIGAN_WITH_GRAD, BIGVGAN_NO_GRAD, PWG_NO_GRAD, HYBRID_MEL_UNIT_NO_GRAD])
GENERATOR_METHODS = MEL_VOCODER_METHODS | UNIT_METHODS
# Contrastive Methods
SIMCLR = "mini_batch"
MOCO = "momentum_contrast"
CONTRAST_METHODS = [SIMCLR, MOCO,]
# Evaluation Modes
TEST_MODE = "test"
VALID_MODE = "validation"
# Emotion Tasks
CLASSIFICATION_TASK_EMOTION = 'emotion'
CLASSIFICATION_TASK_GENDER = 'gender'
