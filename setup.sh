#!/bin/bash

git clone https://github.com/malllabiisc/EmbedKGQA.git
cd EmbedKGQA
mkdir checkpoints
mkdir checkpoints/MetaQA

MINICONDA_INSTALLER_SCRIPT=Miniconda3-py37_4.9.2-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT

chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

conda install --channel defaults conda python=3.7.0 --yes
conda update --channel defaults --all --yes

conda --version
python --version

export PYTHONPATH="/usr/local/bin/python"

conda create --name EmbedKGQA --file requirements.txt
source activate EmbedKGQA
pip install pyyaml
pip install numba
pip install path.py
pip install tb-nightly

pip install gdown
gdown https://drive.google.com/uc?id=1Ly_3RR1CsYDafdvdfTG35NPIG-FLH-tz
gdown https://drive.google.com/uc?id=1uWaavrpKKllVSQ73TTuLWPc4aqVvrkpx

unzip data.zip
unzip pretrained_models.zip

rm data.zip
rm pretrained_models.zip

mv EmbedKGQA/data .
mv EmbedKGQA/pretrained_models .

cd data/MetaQA

gdown https://drive.google.com/uc?id=1nBg92x1j-CwOF6w18VFoUblomUzQkdbj
gdown https://drive.google.com/uc?id=1Mf5Mr5FqfOO_OSSPUactqG3cxjES8vaO
gdown https://drive.google.com/uc?id=1l6BIXdizaGQ5toruixMUdm7hioFKxpsf

cd ../..
