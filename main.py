import argparse
import numpy as np
import pandas as pd
import pickle

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Custom_dataset import create_sentimix_dataset, preprocess_dataset
from train import finetune
from inference import infer

import torch

def save(val_history, file_path):
    with open(file_path,'wb') as f:
        pickle.dump(val_history, f)

if __name__ == '__main__':
    print("parsing arguments")
    parser = argparse.ArgumentParser(description="Sentiment Analysis on Code-Switched data")
    parser.add_argument('--model', default="XLM-T", help="can be either of XLM-T, mBERT, TWHIN-Bert")
    parser.add_argument('--model_on_disk', default="", help="complete path of model on disk")
    parser.add_argument('--dataset', default="SentiMix", type=str, help="can be either of Sentimix or UMSAB")
    parser.add_argument('--data_dir', default="dataset", type=str, help="path to the dataset for SentiMix")
    parser.add_argument('--task', default="inference", type=str, help="can be either inference or fintuning (for mbert/TWHINBer with UMSAB dataset only).")
    parser.add_argument('--cpt_dir', default="checkpoint_logs", type=str, help="The directory to store check points.")
    parser.add_argument('--op_dir', default="output_logs", type=str, help="The directory to store predictions and other outputs suchs as loss metrics through out the job.")
    # #parser.add_argument("--NUM_GPUS", default="4", type=int, help="NUM_GPUs."
    parser.add_argument("--BATCH_SIZE", default="200", type=int, help="BATCH_SIZE.")
    parser.add_argument('--seed', default=1, type=int, help='...')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--max_epochs', default=20, type=int, help='Number of training epochs (will train all of them then select the best one)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("device: ",torch.cuda.get_device_name(0))

    print("generating model and tokenizer-")
    if args.model == "XLM-T":
        model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    elif args.model == "mBERT":
        model_path = "bert-base-multilingual-uncased"
    elif args.model == "TWHIN-Bert":
        model_path = "Twitter/twhin-bert-base"
    else:
        model_path = args.model # incase we wanna load models other than those.
    
    num_labels = 3
    
    print("loading tokenizer from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if(args.model_on_disk != ""):
        model_path = args.model_on_disk

    print("loading model from ",model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    if args.dataset == "UMSAB":
        languages = ["english","hindi"]

    elif args.dataset == "SentiMix":
        languages = ["English","Hindi","transliterated","Code-Switched"]
    
    if args.task == "Finetuning": # called only with UMSAB 
        dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "all")
        print('\nlabels are ',set(dataset['train']['label']))
        dataset['train'] = preprocess_dataset(dataset['train'], tokenizer)
        dataset['validation'] = preprocess_dataset(dataset['validation'], tokenizer)
        print('size of train set = ',len(dataset['train']))
        print('size of validation set = ', len(dataset['validation']))
        model, best_step_english, val_history_english = finetune(model_name=model_path, model=model, train_data=dataset['train'], val_data=dataset['validation'] ,LANGUAGE="all",
                MAX_EPOCHS=args.max_epochs, LR=args.lr, SEED=args.seed, BS=args.BATCH_SIZE,checkpoint_dir=args.cpt_dir)
        save(val_history_english, f'{args.op_dir}/{args.model}_all.p')
        model.save_pretrained(f'{args.cpt_dir}/FINAL_{args.model}_{args.max_epochs}_{args.lr}', from_pt=True) # Save final model
        print(f'model saved at - {args.cpt_dir}/FINAL_{args.model}_{args.max_epochs}_{args.lr}')

    elif args.task == "inference":
        if args.dataset == "UMSAB":
            for lang in languages:
                print(f'\nloading {args.dataset} - {lang} dataset for {args.task}.')
                dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", lang)
                dataset = preprocess_dataset(dataset['test'], tokenizer)
                print(f'\nperformance of model {args.model} on {args.dataset} for {lang} language.\n')
                infer(model, dataset, lang, args.op_dir, args.seed)
        
        if args.dataset == "SentiMix":
            for lang in languages:
                print(f'\nloading {args.dataset} - {lang} dataset for {args.task}.')
                dataset = create_sentimix_dataset(args.data_dir, lang, tokenizer)
                print(f'\nperformance of model {args.model} on {args.dataset} for {lang} language.\n')
                infer(model, dataset, lang, args.op_dir, args.seed)
