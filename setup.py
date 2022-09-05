from distutils.core import setup
from setuptools import find_packages


setup(
    name='phoneme_rate_conversion',
    version='0.0.1',
    description=(
        'Supplementary code for the publication: '
        '"Investigation into Target Speaking Rate Adaptation for Voice '
        'Conversion" (INTERSPEECH 2022).'
    ),
    author='Michael Kuhlmann',
    author_email='kuhlmann@nt.upb.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'cached_property',
        'torch',
        'numpy',
        'scipy',
        'matplotlib',
        'ipywidgets',
        'seaborn',
        'tensorboardX',
        'einops',
        'sacred==0.8.2',
        'praat-textgrids',
        'tgt',
        'psutil',
        'webrtcvad',
        'samplerate',
        'pyyaml',
        'padertorch @ git+http://github.com/fgnt/padertorch',
        'paderbox @ git+http://github.com/fgnt/paderbox',
        'lazy_dataset @ git+http://github.com/fgnt/lazy_dataset',
        # 'unsup_seg @ git+https://github.com/michael-kuhlmann/UnsupSeg'
    ]
)
