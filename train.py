from pathlib import Path
import datetime
from datasets import Dataset

import numpy as np
from transformers import TrainingArguments, Trainer, EvalPrediction

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


def finetune(model_name, model, train_data: Dataset, val_data: Dataset, LANGUAGE,
                MAX_EPOCHS = 20, LR = 0.001, SEED = 1, BS = 200, checkpoint_dir = "."): 

    # Fixed params
    EVAL_STEPS = 20
    BATCH_SIZE = BS
    NUM_LABELS = 3
    print("batchsize = ",BATCH_SIZE)
    #for logging
    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d_%H%M%S_%f')
    UNIQUE_NAME = f"{LANGUAGE}_{model_name.replace('//','-')}_{LR}_{SEED}_{now}"
    UNIQUE_NAME = UNIQUE_NAME.replace('.','-')
    DIR = f"{checkpoint_dir}/{UNIQUE_NAME}/"
    Path(DIR).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
    learning_rate=LR,
    num_train_epochs=MAX_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=10,
    output_dir=DIR,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    seed=SEED,
    load_best_model_at_end=True,
    do_eval=True,
    eval_steps=EVAL_STEPS,
    evaluation_strategy="steps"
    )
    val_history = []
    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='macro')
        acc = (preds == p.label_ids).mean()
        val_history.append(f1)
        return {"macro_f1":f1, "acc": acc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = train_data,
        eval_dataset = val_data,
        compute_metrics = compute_accuracy,
    )
    trainer.train()
    trainer.evaluate()

    best_step = val_history.index(val_history.pop())+1
    n_steps = len(val_history)

    print(f'Validation history for {LANGUAGE} : {val_history}')
    print(f'total number of steps for {LANGUAGE} = {n_steps}')
    print(f'Best Step for {LANGUAGE} = {best_step}')

    return model, best_step, val_history
