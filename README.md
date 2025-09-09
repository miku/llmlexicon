# LLM Lexicon

> Short notes on LLM and related technologies.

## Benchmarks

Used to evaluate the performance of a model in a specific task. Examples include:

* LMsys chatbot arena


## BOND

See also:

* [BOND: Aligning LLMs with Best-of-N Distillation](https://arxiv.org/pdf/2407.14622)

## Byte-Pair Encoding (BPE)

A compression algorithm, originally developed in 1994. A 2016 paper introducing
subword tokens used BPE.

> We adapt byte pair encoding (BPE) (Gage, 1994), a compression algorithm, to
> the task of word segmentation. BPE allows for the representation of an open
> vocabulary through a fixed-size vocabulary of variable-length character
> sequences, making it a very suit- able word segmentation strategy for neural
> network models.

See also:

* [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162.pdf) (2016)


## Distilliation


## Elo Rating System

Relative skill assessment. Used for example in the LMsys chatbot arena scoring.

## Encoder-Decoder Network

Encoder transforms data into lower dimensional space ("bottleneck") and the
decoder uses this lower dimensional space to construct the a desired output. It
expand the compressed representation back into the target format.

Used for sequence-to-sequence tasks, like translation, or image captioning
(encode and image and decode a description).

A special case are autoencoders, where input and output are the same; used for
dimensionality reduction, denoising, or learning compact representations.

The encoder and decoder can use different architectures.

## Finetuning

## Instruction-Tuning

Part of post-training.

## KV-cache

## Logit

Unnormalized neural network output, before it is converted to probabilities.
Term comes from logistic regression.

## Posttraining

## Pretraining

Unsupervised training of a language model, on large amounts of text. Token
budgets:

| Model       | Token Budget | Reference                        |
|-------------|--------------|----------------------------------|
| Gemma 3 27B | 14T          | https://arxiv.org/abs/2503.19786 |
| Gemma 3 12B | 12T          | https://arxiv.org/abs/2503.19786 |
| Gemma 3 4B  | 4T           | https://arxiv.org/abs/2503.19786 |
| Gemma 3 1B  | 2T           | https://arxiv.org/abs/2503.19786 |

## Proximal policy optimization (PPO)


## Reward model (RM)

A reward model is a neural network trained to predict human preferences by
outputting scalar reward scores for model responses. It essentially serves as a
proxy for human judgment during the reinforcement learning phase.

## RLHF

> Reinforcement learning from human feedback (RLHF) is a key driver of quality
> and safety in state-of-the-art large language models.

> For example, large language models can generate outputs that are untruthful,
> toxic, or simply not helpful to the user. In other words, these models are
> not aligned with their users.

The pipeline:

* Step 1: Train a base language model using standard supervised learning
* Step 2: Collect human preference data by having humans compare pairs of model outputs
* Step 3: Train a reward model to predict which outputs humans prefer
* Step 4: Use the reward model to fine-tune the original model with reinforcement learning

See also:

* [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)

## Quantization Aware Training (QAT)

Models are trained with FP32, usually. Introduce quantized weights. Model is
exposed to quantization at training time.


## Supervised Fine-tuning (SFT)


## Tokenizer

Examples:

* [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

## Training an LLM

Typically three stages:

* Pretraining on large corpora with next token prediction
* Finetuning with to follow instructions via supervised fine-tuning (SFT)
* RLHF is used to further improve the quality of the generations

See also:

* [BOND: Aligning LLMs with Best-of-N Distillation](https://arxiv.org/pdf/2407.14622)


## Variational Autoencoder (VAE)

An encoder-decoder architecture, where the input and output are the same.
Mapping the input to parameters of a probability distribution. Decoder uses a
sample from the distribution to reconstruct.

The randomness allows to sample from anywhere in the latent space to generate
similar or new data.
