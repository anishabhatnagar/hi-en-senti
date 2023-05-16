# Bilingual or Byelingual? Code mixing and romanization in Hindi Sentiment Analysis

## Problem Definition
**Code-Mixing**: is a communication phenomenon where the speakers embed words, phrases or morphemes from one language into another. 

Hindi-English Code-Mixing looks like: 

Example: 	ye class bauhat fun hai 

Translation: 	this class is a lot of fun 

**Romanization**: the representation of a language text using the Roman alphabet. 

Do romanization and code-mixing trigger substantial changes in model performance? 

## Dataset

The following datasets have been used - 

1. [UMSAB](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) - It contains 14.7K tweets in 8 different languages. This dataset was used for finetuning our baseline models.
2. [Sentimix-2020](https://ritual-uh.github.io/sentimix2020/) - The fintuned models are evaluated on this Hindi-English-Romanized code switched twitter data. 
3. A synthetic data consisting of pure English translations, pure Hindi translations (in Devanagari script) and transliterated Hindi (in Latin script) was generated from the [Sentimix-2020](https://ritual-uh.github.io/sentimix2020/) data . GPT-3.5 Turbo was used for the translations and [IndicXLIT](https://ai4bharat.org/indic-xlit) for transliterations. 

### Classes and labels -
The following label mapping was used consistently through out the three datasets - 
```
{
  'negative' : 0,
  'neutral' : 1,
  'positive' : 2
}
```

## Models
The following three baselines were examined -
1. [XLM-Tw model](https://arxiv.org/abs/2104.12250)
2. [mBERT](https://arxiv.org/abs/1810.04805)
3. [TwHIN-BERT](https://arxiv.org/abs/2209.07562)

## Environment setup 

to create a conda environment witht he provided ``` environment.yml``` file, run the follwoing commands - 

```
conda env create -f environment.yml
source activate my_env
```
## Code

the main.py file accepts the following command line arguments to control the experiments - 

```
1. model : can be either of XLM-T, mBERT, TWHIN-Bert. it loads the tokenizer and model from the hugginface hub. (defaults to XLM-T)
2. model_on_disk : use this if the model is stored on disk. provide the complete path of the model on disk 
3. dataset : can be either of Sentimix or UMSAB. Sentimix will load the full set including the original code switched tweets, translations into Hindi (devanagri) and English and the transliteration into Hindi (Romanised) (defaults to UMSAB).
4. data_dir : path to the dataset for SentiMix 
5. task : can be either inference or finetuning (for mbert/TWHINBert , to be used with UMSAB dataset only). (defaults to inference)
6. cpt_dir : The directory to store check points. (defaults to a folder named checkpoint_logs.)
7. op_dir : The directory to store predictions and other outputs suchs as validation history through out the job. (defaults to a folder named output_logs.)
8. BATCH_SIZE : defaults to 200
9. seed : defaults to 1. 
10. lr : defaults to 0.0001 
11. max_epochs : defaults to 20. 
```

### Finetuning
Finetuning is performed for **mBERT** and **TwHIN-BERT** on the **UMSAB** dataset only.

For Finetuning mBERT, execute the follwoing command - 
```
python3 main.py --model mBERT --dataset UMSAB --task Finetuning --cpt_dir checkpoint_logs_mbert --op_dir output_logs_mbert --BATCH_SIZE 8 --lr 2e-5
```

For Finetuning TwHIN-BERT, execute the follwoing command - 
```
python3 main.py --model TWHIN-Bert --dataset UMSAB --task Finetuning --cpt_dir checkpoint_logs --op_dir output_logs --BATCH_SIZE 8 --lr 2e-5
```

### Inference
to perform inference with the chosen model, execute the following command -
```
python3 main.py --model XLM-T --dataset UMSAB --task inference --cpt_dir checkpoint_logs --op_dir output_logs
```
(This will evaluate XLM-T on the UMSAB dataset.)

Additionally, a sample shell file (run_job.sh) is also provided to excute these jobs over High Performance Computing clusters.

## Collaborators
1. Anisha Bhatnagar (ab10945@nyu.edu)
2. Gauri Gupta (gg2751@nyu.edu)
3. Benjamin Feuer (bf996@nyu.edu)
3. Daiheng Zhang (dz2266@nyu.edu)
