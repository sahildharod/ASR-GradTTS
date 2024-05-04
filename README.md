## GRAD-TTS : Diffusion Probabilistic Model for Text-to-Speech

Team Members : Sahil Dharod (210070026), Azeem Motiwala (210070018), Jay Chaudhary (210070022), Shlesh Gholap (210070080)

The paper GRAD-TTS presented the first acoustic feature generator utilizing the concept of diffusion probabilistic modelling. The main generative engine of Grad-TTS is the diffusion-based decoder that transforms Gaussian noise parameterized with the encoder output into mel-spectrogram while alignment is performed with Monotonic Alignment Search. The model we propose allows to vary the number of decoder steps at inference, thus providing a tool to control the trade-off between inference speed and synthesized speech quality.
In this hack role, we made the following changes:
1) Drawing inspiration from

2) The authors did not try any other variance schedule apart from linear
   We have implemented cosine noise scheduling for the diffusion process which given by :
   $min (x,y)$

### Installation
Firstly install the Python package requirements acc to the original implementation
```bash
pip install -r requirements.txt
```
To resolve other version related errors, install the following packages:
```bash
pip install torchaudio==0.9.0
pip install setuptools==59.5.0
pip install numba
```
Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```
Download and extract the LJSpeech dataset using the following command
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
```

