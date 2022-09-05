# Investigation into Target Speaking Rate Adaptation for Voice Conversion

**Perform any-2-any speaking rate conversion**

Demo: [go.upb.de/interspeech2022](https://groups.uni-paderborn.de/nt/interspeech2022_vc/index.html)

## Installation
```bash
git clone https://github.com/michael-kuhlmann/phoneme_rate_conversion.git
cd phoneme_rate_conversion
pip install -e .
```

## Training
We provide a training script to train a local phoneme rate estimator from forced alignments.

### Data preparation
1. Prepare the following environment variables:
   - `$DB_ROOT`: Points to the path where the LibriSpeech corpus and alignments will be stored
    - `$STORAGE_ROOT`: Points to the path where the trained models will be stored
2. Download [LibriSpeech](https://www.openslr.org/12/) to `$DB_ROOT`
3. Download [librispeech.json](https://uni-paderborn.sciebo.de/s/f6xCGx1R4lXO24c) and put it under `jsons`
4. Download [librispeech_phone_ali](https://uni-paderborn.sciebo.de/s/f6xCGx1R4lXO24c) to `$DB_ROOT`
    - We used the [Montreal Forced Aligner (MFA)](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to 
      get the alignments. To create the alignments yourself, see [here](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-align-pretrained)

### Running the training script
We use [sacred](https://github.com/IDSIA/sacred) for easily configurable training runs. To start the training with the 
default values, use
```bash
python -m phoneme_rate_conversion.train
```
To customize the configuration, use
```bash
python -m phoneme_rate_conversion.train with shift_ms=10 window_size_ms=20
```
This will change the shift and window size of the STFT from 12.5ms and 50ms to 10ms and 20ms, respectively. You can get
the full config list with customizable options from
```bash
python -m phoneme_rate_conversion.train print_config
```

### Pre-trained models
We provide a [pretrained model](https://uni-paderborn.sciebo.de/s/f6xCGx1R4lXO24c) that was trained on LibriSpeech and 
Timit and showed good generalization.

## Inference
During inference, we can perform a speaking rate conversion between two audios without requiring any text labels.

```python
from phoneme_rate_conversion.inference import SpeakingRateConverter
from scipy.io import wavfile
converter = SpeakingRateConverter.from_config(SpeakingRateConverter.get_config(dict(model_dir='pretrained/')))
c_sample_rate, content_wav = wavfile.read('/path/to/content/wav')
s_sample_rate, style_wav = wavfile.read('/path/to/style/wav')
assert c_sample_rate == s_sample_rate
content_wav_time_scaled = converter(content_wav, style_wav, in_rate=c_sample_rate)
```
This will imprint the speaking rate of `style_wav` onto `content_wav`. The quality of the conversion depends on the 
choice of the utterances, the quality of the speaking rate estimator and the voice activity detection (VAD) algorithm. 
`SpeakingRateConverter` supports different time scaling and VAD algorithms which can be customized by overwriting the
`time_scale_fn` and `vad` arguments:
```python
import phoneme_rate_conversion as prc
converter = prc.inference.SpeakingRateConverter(
   model_dir='pretrained/',
   time_scale_fn=prc.modules.time_scaling.WSOLA(sample_rate=c_sample_rate),
   vad=prc.utils.vad.WebRTCVAD(sample_rate=c_sample_rate),
)
```
To deactivate the VAD, pass `vad=None`.

## Unsupervised Phoneme Segmentation (Scenario 2)
In our paper, we also proposed a completely unsupervised approach based on [unsupervised phoneme segmentation](https://github.com/felixkreuk/UnsupSeg).
We slightly modified the code to work with this repository:
```bash
git clone https://github.com/michael-kuhlmann/UnsupSeg.git
cd UnsupSeg
pip install -e .
```
The inference works similarly:
```python
from phoneme_rate_conversion.inference import UnsupSegSpeakingRateConverter
converter = UnsupSegSpeakingRateConverter.from_config(
   UnsupSegSpeakingRateConverter.get_config(dict(model_dir='pretrained/')))
content_wav_time_scaled = converter(content_wav, style_wav, in_rate=c_sample_rate)
```
You can find a pretrained model in the same [pretrained](https://uni-paderborn.sciebo.de/s/f6xCGx1R4lXO24c) directory 
or use one from the [original repository](https://github.com/felixkreuk/UnsupSeg).
