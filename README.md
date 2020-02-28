# StarGAN-Voice-Conversion

This is a pytorch implementation of the paper: StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks  https://arxiv.org/abs/1806.02169 .

# Dependencies
* Python 3.6 (or 3.5)
* Pytorch (torch 1.3.1, torchvision 0.4.2)
* pyworld
* tqdm
* librosa
* tensorboardX and tensorboard

# Installation
**Tested on a Python verison 3.6.2 in linux VM environment**

## Python packages

**NB:**
* Most recent `Pillow` version couldn't load `PILLOW_VERSION`, so reinstalled to a working version
* For mac users who cannot install `pyworld` see: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder 

```bash
pip install torch torchvision
pip uninstall pillow
pip install pillow==5.4.1
conda install -c conda-forge librosa
pip install pyworld==0.2.8
pip install tqdm==4.41.1
```

## Libraries
* Download and install SoX version 14.4.2. Binary: https://sourceforge.net/projects/sox/files/sox/14.4.2/
* You may also need to download libsndfile verison 1.0.28. Binary: http://www.linuxfromscratch.org/blfs/view/svn/multimedia/libsndfile.html
* For mac users, these are available as `brew` commands

# Usage
## Download Dataset

Download and unzip [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) corpus to designated directories.

```bash
mkdir ../data/VCTK-Data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ../data/VCTK-Data
```
If the downloaded VCTK is in tar.gz, run this:

```bash
tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
```

## Preprocess Data

We will use Mel-Cepstral coefficients(MCEPs) here.

This example script is for the VCTK data which needs resampling to 16kHz, the script allows you to preprocess the data without resampling either.

```bash
# VCTK-Data
python preprocess.py --resample_rate 16000 \
                     --origin_wavpath ../data/VCTK-Data/VCTK-Corpus/wav48 \
                     --target_wavpath ../data/VCTK-Data/VCTK-Corpus/wav16 \
                     --mc_dir_train ../data/VCTK-Data/mc/train \
                     --mc_dir_test ../data/VCTK-Data/mc/test \
                     --speaker_dirs p262 p272 p229 p232 p292 p293 p360 p361 p248 p251
```

## Train Model

```bash
python main.py --train_data_dir ../data/VCTK-Data/mc/train \
               --test_data_dir ../data/VCTK-Data/mc/test \
               --use_tensorboard False \
               --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav16 \
               --model_save_dir ../data/aca16sjb/VCTK-Data/models \
               --sample_dir ../data/VCTK-Data/samples \
               --num_iters 200000 \
               --batch_size 8 \
               --speakers p262 p272 p229 p232 \
               --num_speakers 4
```

If you encounter an error such as:
```bash
ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
```
You may need to export `export LD_LIBRARY_PATH`: (see [Stack Overflow](https://stackoverflow.com/questions/49875588/importerror-lib64-libstdc-so-6-version-cxxabi-1-3-9-not-found))
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<PATH>/<TO>/<YOUR>/.conda/envs/<ENV>/lib/
```

## Convert

For example: restore model at step 125000 and specify the speakers.

```
python convert.py --resume_iters 125000 \
                  --num_speakers 4 \
                  --speakers p262 p272 p229 p232 \
                  --train_data_dir ../data/VCTK-Data/mc/train/ \
                  --test_data_dir ../data/VCTK-Data/mc/test/ \
                  --wav_dir ../data/aca16sjb/VCTK-Data/VCTK-Corpus/wav16 \
                  --model_save_dir ../data/VCTK-Data/models \
                  --convert_dir ../data/VCTK-Data/converted
```
This saves your converted flies to `../data/VCTK-Data/converted/`

## Calculate Mel Cepstral Distortion

Calculate the Mel Cepstral Distortion of the reference speaker vs the synthesized speaker. Use `--spk_to_spk` tag to define multiple speaker to speaker folders generated with the convert script.

```
python mel_cep_distance.py --convert_dir ../data/VCTK-Data/converted/125000 \
                           --spk_to_spk p262_to_p272 \
                           --output_csv p262_to_p272.csv
```

# TODO:
- [ ] Fine tune the model
- [ ] Include sample set of results
- [ ] Include VCC dataset download instructions/script
