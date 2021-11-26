# STFTune
Pitch correction via STFT analysis/synthesis. Final project for MUSIC320A: Introduction to Audio Signal Processing I.

## Usage

1. [Install Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html) if you don't already have it.

2. Setup the conda virtual environment
```
conda env create -f environment.yml
conda activate stftune
```

3. Run `jupyter notebook` and open `stftune/demo.ipynb`. After running the cells, you should a dropdown to select an audio file and some interactive sliders to tune parameters. This main function will do the following:
    - Display the audio read from the file so you can listen to it.
    - Perform STFT analysis and draw a spectrogram of the signal.
    - Detect the peaks in the STFT spectrum and draw them. The parameters for this can be tuned with the interactive sliders. 
    - List the notes detected in the signal and draw a chart of them. 