import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, Trainer, EvalPrediction

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def infer(model, dataset : Dataset, language, DIR, SEED):
    EVAL_STEPS = 20
    BATCH_SIZE = 32
    
    val_history = []
    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        f1 = f1_score(p.label_ids, preds, average='macro')
        acc = (preds == p.label_ids).mean()
        val_history.append(f1)
        return {"macro_f1":f1, "acc": acc}
    print('initalizing args')
    test_args = TrainingArguments(
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=10,
        output_dir=DIR,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        seed=SEED,
        do_predict = True,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps"
    )
    print('initializing trainer object')
    tester = Trainer(
        model=model,
        args=test_args,
        compute_metrics = compute_accuracy,
    )
    
    print('generating predictions')
    test_preds_raw, test_labels , out = tester.predict(dataset)
    test_preds = np.argmax(test_preds_raw, axis=-1)

    print(out)
    test_preds_raw_path = f"{DIR}/test_preds_raw_{language}.txt"
    test_preds_path = f"{DIR}/test_preds_{language}.txt"
    np.savetxt(test_preds_raw_path, test_preds_raw)
    np.savetxt(test_preds_path, test_preds)

    print(classification_report(test_labels, test_preds, digits=3))
