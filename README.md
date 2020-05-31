# Fake News Detector as a New Evaluation Metric for Text GANs

In this work we explore two problems: the process of text generation and neural fake news detection. LSTM (with differenc embeddings) and Bert models are proposed to deal with neural fake news detection task. Also the impact of sampling type on classifier accuracy is checked. The stadu is based on Cornell Newsroom Summarization Dataset and the final dataset consists of 1.2 mln texts, half of them is real, the other half was generated using GPT-2 model. Also, we explore the process of training language model as GAN.


All **data** is available at: https://drive.google.com/drive/u/0/folders/1ldPR1TKsBJHrVnFyIprw1aBDQ8yaNJao

Some of generated samples can be seen at `generated_samples` folder

# How it works

You need to clone the repository to your local machine and do the following steps:
1. install all the requirements `pip install -r requirements.txt`
2. specify the configuration options in `configuration.py`

After that you can run .ipynb notebook from /notebooks folder with postfix _run to run any of *instructor* models (For example, '*LSTM_discriminator_run.ipynb*')

All metric updates will be logged to wandb project page (you need to specify it jupyter notebook file).


# Wandb project page

[General page](https://app.wandb.ai/2ispany3/dpl)

**Report on LSTM model as fake news detector** https://app.wandb.ai/2ispany3/dpl/reports/LSTM-model--Vmlldzo5NjQxMw

**Report on Bert models as fake news detector** - https://app.wandb.ai/2ispany3/dpl/reports/Bert-models--VmlldzoxMjA5MjY

**Report on Samplings experiments** - https://app.wandb.ai/2ispany3/dpl/reports/Samplings-on-final-dataset--Vmlldzo5NjUyMQ

**Report on RelGAN model with different discriminator models** - https://app.wandb.ai/2ispany3/dpl/reports/RelGAN-with-different-discriminators--VmlldzoxMjA5NDI
