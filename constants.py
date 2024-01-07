GRIFFINLIM = "no_vocoder"
HIFIGAN_NO_GRAD = "vocoder_nogradient"
HIFIGAN_WITH_GRAD = "vocoder_withgradient"
# Do not modify values of constants below as they are likely to be correlated to terms in json files.
# This is configured for HiFi-GAN itself. Will only be used in ReVISE.
UNIT_HIFIGAN_NO_GRAD = 'HuBERT Label as input (ReVISE)'
# Soft or hard prediction. Has no effect on ReVISE.
UNIT_SOFT = "soft"
UNIT_HARD = "hard"
UNIT_METHODS = [UNIT_SOFT, UNIT_HARD, UNIT_HIFIGAN_NO_GRAD,]
GENERATOR_MODES = [GRIFFINLIM, HIFIGAN_NO_GRAD, HIFIGAN_WITH_GRAD, UNIT_HIFIGAN_NO_GRAD,]