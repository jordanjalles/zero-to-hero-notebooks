# Zero to Hero, typed along

My notebooks from Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) course: a scalar autograd engine, the makemore character models (including backpropagation written out by hand through every layer), and a small GPT.

Part of a series of repos from my 2026 machine-learning sabbatical: a designer learning ML by building models from scratch and running experiments.

## What's here

- `micrograd/`: autograd engine from scratch
- `build_makemore/`: bigrams → MLP → batchnorm → manual backprop → WaveNet-style model
- `chatgpt_from_scratch/`: nanoGPT on Shakespeare, and the tokenizer lecture

## Running it

Follow along with the videos. Get `names.txt` and `tinyshakespeare.txt` from [karpathy/makemore](https://github.com/karpathy/makemore) and [char-rnn](https://github.com/karpathy/char-rnn).

**Code:** written by hand in Jupyter notebooks, with a chatbot as a tutor for questions and plotting helpers. Course material and structure are Karpathy's (MIT); these are my own typed versions.

**Requirements:** Python 3.11+, PyTorch 2.x, torchvision, matplotlib, a CUDA GPU for anything beyond MNIST.
