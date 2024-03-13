# BERT-trn-extraction
Automatic extraction of transcriptional regulatory interactions of bacteria from biomedical literature using a fine-tuned BERT model

Computational Genomics, Center for Genome Sciences, UNAM 
- Dr. Carlos Francisco Méndez Cruz
- Dr. Ali Berenice Posada Reyes
- LCG. Alfredo Varela Vega 

## datasets

Datasets used for TRN reconstructions (e.coli and salmonella)

1. Master dataset used for fine tunning bert models (training, validation and evaluation):  
```ECO_dataset_master_curated_dataset_v2_subset.tsv```

2. The 264 PMIDS downloaded for salmonella TRN reconstruction using LUKE fine-tunned model:
```264_salmo_264_salmo_pmids_for_bert_trn_reconstruction.tsv```

3. Sentences preprocessed in a dataframe structure for TRN inference of Salmonella's 264 pmids:
```STM_data_set_dl_articles_264_preprocessed_for_luke.pkl```
   
4. Salmonella regulated genes list:

   ```STM_regulated.tsv```

5. Salmonella transcription factor lists:
   
   ```STM_tfs.tsv```


## 01_preprocessing 

Preprocessing for fine-tunning 6 BERT architectures

1. Getting the entity_spans and necessary data structure for fine-tunning LUKE: 
- Using our E.coli dataset to prepare for fine-tunning.
```/bin/preprocessing_for_luke_of_ecoli.ipynb```

- Using raw articles, gene lists and factor lists (salmonella) for inference.
```/bin/preprocessing_articles_for_LUKE_from_entity_lists.py``` 

2. Output will be .pkl for data structure preservation: 
```../results/ECO_dataset_master_curated_whole_tagging_info_with_span_data_serialized_4_v2.pkl```


## 02_modelling

Hyperparameter search and Fine-tunning of 6 BERT architectures 

1. Fine-tunning for hyperparameter selection of 6 different BERT architectures:
>`02_modelling/bin`

- [BERT](https://huggingface.co/bert-base-uncased): `bert_v1_hyperparameter_search.py`
- [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1): `biobert_v2_hyperparameter_search.py`
- [BioLinkBERT](https://huggingface.co/michiyasunaga/BioLinkBERT-base): `biolinkbert_v2_hyperparameter_search.py`
- [BioMegatron](https://huggingface.co/EMBO/BioMegatron345mUncased): `biomegatron_v1_hyperparameter_search.py`
- [BioRoBERTa](https://huggingface.co/allenai/biomed_roberta_base): `bioroberta_v1_hyperparameter_search.py`
- [LUKE](https://huggingface.co/studio-ousia/luke-base): `luke_v2_hyperparameter_search.py`

> All the above models were trained using `01_preprocessing/results/ECO_dataset_master_curated_whole_tagging_info_with_span_data_serialized_4_v2.pkl`


2. Best fine-tunned model LUKE:
>`02_modelling/bin`
- Retraining the best model using best hyperparameters found with `02_modelling/bin/luke_v2_hyperparameter_search.py`: `luke_best_model_v1.py`
- Best model with output showing each sentence to generate error analysis: `luke_best_model_error_analysis.ipynb`
- Only inference steps loading the fine-tunned model: `luke_best_model_for_inference.py`


3. Best model predictions during test and inference: 

>`02_modelling/results`

- E.coli predictions:`luke_best_model_all_predictions_for_error_analysis.tsv`
- Salmonella predictions during inference: 
  - Whole Dataset predictions including the 4 classes (activator, regulator, repressor, no_relation): `luke_stm_predictions_264_all_dataset.tsv`
  - Whole Dataset predictions with corresponding sentence including the 4 classes (activator, regulator, repressor, no_relation: `luke_stm_predictions_264_with_sentence_all_dataset.tsv`
  - Dataset predictions for TRN generation using classes that reflect a regulation(activator, repressor, regulator: `luke_stm_predictions_264_dataset_without_no_relation.tsv`

## 03_visualization 

Dataset visualization and statistics 

- Hyperparameter sweep visualization (batch_size, learning rate, epoch, validation_loss, validation_f1): `bin/hyperparameter_sweeps_viz.ipynb`
- Sentence level stats(length, distribution of categories, pmids distribution): `stats_and_preprocessing_viz.Rmd`
