# Transcriptome Transformer (TxT)
Transcriptome Transformer: prediction of clinical features from transcriptome using transformer via multi-task learning with modeling gene interactions

## Installation

### Dependency

The code has been tested in the following environment:


| Package           | Version   |
|-------------------|-----------|
| Python            | 3.9.7     |
| Pytorch           | 1.12.1    |
| CUDA              | 11.3      |
| numpy             | 1.21.2    |
| pandas            | 1.4.0     |
| scikit-learn      | 1.1.2     |
| scikit-survival   | 0.18.0    |
| scipy             | 1.7.3     |

You can chage the package version according to your need.

### Install via Mamba

You can set up the environment using [Mamba](https://github.com/conda-forge/miniforge).
```
mamba env create -f environment.yaml
```

## Running on your dataset

## 1. Pre-training gene embedding on a biological network

The embedding for each gene, used as input for training, is derived from a pre-trained node2vec model on a biological network.

You can use other methods according to your need.

### 1) Input data

(1) *Network*

* File type: CSV
* Format overview:

    ```
    protein1,protein2
    CHEK2,TP53
    FOXA1,GATA3
    MSH2,TP53
    CHEK2,MSH2
    ...
    ```

### 2) Training

```
python src/embed_gene.py --network_file <network.csv> --result_dir <result_dir>
    --embedding_dim <embedding_dim> --sparse --device <device>
```

*Options*

- `--network_file`: (csv) A network file representing relationships between genes.
- `--result_dir`: (dir) A directory to save output files.
- `--embedding_dim`: (int) The size of each embedding vector.
- `--sparse`: An option to control the memory efficiency of storing random walks.
- `--device`: (int) Device number.

Check the [script](https://github.com/BonilKoo/TxT/blob/main/src/embed_gene.py#L19) for other options.

*Output*

- `embedding.csv`: (csv) A csv file representing gene embedding. The gene names are listed in the first column, and the subsequent columns contain the embedding values for each gene in different dimensions.

## 2. Training and Evaluation

### 1) Input data

(1) *Omics profile*

* File type: CSV
* Format overview:

    ```
    Sample,CHEK2,FOXA1,GATA3,...
    sample1,2.142685,0.655440,0.103477,...
    sample2,0.657207,0.413347,3.192018,...
    sample3,3.651882,0.091432,0.103477,...
    ...
    ```

(2) *Clinical features*

* File type: TSV
* Format overview:
    - Regression

        ```
        Sample	Age
        sample1 85
        sample2 70
        sample3 30
        ...
        ```

    - Classification

        ```
        Sample  NHG
        sample1 2
        sample2 3
        sample3 3
        sample4 1
        ...
        ```

    - Survival

        ```
        Sample	OS_duration	OS_event
        sample1 3313    1
        sample2 4094    0
        sample3 4096    0
        ...
        ```

    - Multi-task

        ```
        Sample  Age Size    NHG PAM50   OS_duration OS_event
        sample1 85  13  2   LumA    3313    1
        sample2 70  16  3   Basal   4094    0
        sample3 30  20  3   LumA    4096    0
        ...
        ```

(3) *gene embedding*

* File type: CSV
* Format overview:

    ```
    Gene,0,1,2,...
    CHEK2,-0.776163,-0.510058,-0.566633,...
    FOXA1,-0.133213,-0.649964,-0.390464,...
    GATA3,-0.459504,-0.668453,-0.255762,...
    ...
    ```

(4) *Task list*

Only for multi-task learning

* File type: TSV
* Format overview:
    - Multi-task

        ```
        name    task
        Age regression
        Size    regression
        NHG classification
        PAM50   classification
        OS_duration survival_time
        OS_event    survival_event
        ```

### 2) Training

* Regression

    ```
    python src/run_regression.py --input_file <omics_profile.csv> --output_file <clinical_feature.tsv>
    --embed_file <gene_embedding.csv> --result_dir <resuir_dir> --scaler [MinMax/Standard/None]
    --device <device> --xavier_uniform --norm_first
    ```

* Classification

    ```
    python src/run_classification.py --input_file <omics_profile.csv> --output_file <clinical_feature.tsv>
    --embed_file <gene_embedding.csv> --result_dir <result_dir> --scaler [MinMax/Standard/None]
    --device <device> --xavier_uniform --norm_first
    ```

* Survival

    ```
    python src/run_survival.py --input_file <omics_profile.csv> --output_file <clinical_feature.tsv>
    --embed_file <gene_embedding.csv> --result_dir <result_dir> --scaler [MinMax/Standard/None]
    --device <device> --xavier_uniform --norm_first
    ```

* Multi-task

    ```
    python src/run_multi_task.py --input_file <omics_profile.csv> --output_file <clinical_feature.tsv>
    --embed_file <gene_embedding.csv> --result_dir <result_dir> --scaler [MinMax/Standard/None]
    --device <device> --xavier_uniform --norm_first
    --task_file <task_list.tsv>
    ```

*Options*

- `--input_file`: (csv) A omics profile file representing a gene expression dataset where each row corresponds to a sample, and each column, labeled with gene names, represents the expression level of the corresponding gene in the respective sample. The numerical values in the matrix indicate the expression levels of each gene in the corresponding samples.
- `--output_file`: (tsv) A file containing clinical feature data. The format is organized with a header line indicating the type of data and subsequent rows containing sample-specific information.
- `--embed_file`: (csv) A csv file representing gene embedding. The gene names are listed in the first column, and the subsequent columns contain the embedding values for each gene in different dimensions.
- `--result_dir`: (dir) A directory to save output files.
- `--scaler`: (str) A data normalization method. You can choose among [MinMax/Standard/None]. The gene expression levels were normalized by using the expression values of the traning set.
- `--device`: (int) Device number.
- `--xavier_uniform`: An option to use Xavier Uniform initialization for model weights to prevent issues like vanishing or exploding gradients during the training process.
- `--norm_first`: An option to perform LayerNorms before other attention and feedforward operations, otherwise after.
- `--task_file`: (tsv) Only for multi-task learning. A tsv file that outlines a set of tasks. Each row in the file represents a specific task, and the information is organized into two columns: "task name" and "prediction type".

Chek the [script]() for other options.

*Outputs*

- `log.txt`: A file encompassing record of training loss, validation loss, and performance metrics for each training, validation, and test set.
- `model.pt`: A file for the trained model's learned parameters.
