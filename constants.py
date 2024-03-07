GRIFFINLIM = "no_vocoder"
HIFIGAN_NO_GRAD = "vocoder_nogradient"
HIFIGAN_WITH_GRAD = "vocoder_withgradient"
BIGVGAN_NO_GRAD = "vocoder_bigvgan_nogradient"
# Do not modify values of constants below as they are likely to be correlated to terms in json files.
# This is configured for HiFi-GAN itself. Will only be used in ReVISE.
UNIT_HIFIGAN_NO_GRAD = 'HuBERT Label as input (ReVISE)'
UNIT_SPEECH_TOKENIZER_NO_GRAD = 'SpeechTokenizer as vocoder'
# Soft or hard prediction. Has no effect on ReVISE.
UNIT_SOFT = "soft"
UNIT_METHODS = [UNIT_HIFIGAN_NO_GRAD, UNIT_SPEECH_TOKENIZER_NO_GRAD]
GENERATOR_METHODS = [HIFIGAN_NO_GRAD, HIFIGAN_WITH_GRAD, BIGVGAN_NO_GRAD] + UNIT_METHODS