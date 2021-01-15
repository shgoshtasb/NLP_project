This repository contains code of the project of Natural Language Processing course. We adapted the calibration method combining Platt scaling and histogram bining introduced in [Kumar et al (2019)](https://arxiv.org/abs/1909.10155) to estimate the density of correct label distribution during training in [Jung et al (2020) paper titled "Posterior Calibrated Training on Sentence Classification Tasks"](https://arxiv.org/abs/2004.14500). This repository is based on the codes from both of these papers which can be found [here](https://github.com/p-lambda/verified_calibration) and [here](https://github.com/THEEJUNG/PosCal). We train Bert classifier on tasks defined in [xSLUE datasets](https://arxiv.org/abs/1911.03663).

## Requirements
We use python 3.7. Please run ```pip install -r requirements.txt``` to install python dependencies. 

## Running the BERT classifier with regularized training 
Download the xSLUE dataset. The following command trains the classifier on the ShortRomance task located in xslue_data/ShortRomance with MLE loss. In order to train Bert with PosCal/PlattBinMarginal/PlattBinTop/Platt regularization use the flags --poscal_train, --plattbin_train, --plattbintop_train and --platt_train respectively.Training is terminated with early stopping when the validation loss averaged over previous 50 steps exceeds the current validation loss. We use one warmup epoch where we update the Q matrix but don't stop training for more stability in the results. 

```
python code/classify_bert.py --model_type bert --model_name_or_path bert_base_uncased --task_name ShortRomance --do_train --data_dir xslue_data/ShortRomance --output_dir shortromance_MLE
```

Alternatively, running do.py will generate all the results in the report with the following hyperparameters:

| Hyperparameter         | Value           | Flag                        |
| ---------------------  |:---------------:|----------------------------:|
| Learning rate          | 2e-5            | learning_rate               |
| Regularization lambda  | 0.6 or 1.0      | calloss_lambda              |
| # Updates of Q / epoch | 5 (2 for VUA)   | num_updates                 |
| # of bins              | 10              | bin_size                    |
| Batch size             | 32              | per_gpu_train_batch_size    |

The quanittative, qualitative results and calibration plots are generated in Untitled.ipynb.
