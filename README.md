# Transcriptome Transformer (TxT)
Transcriptome Transformer: prediction of clinical features and survival from transcriptome by modeling gene interactions with transformer in a multi-task framework

## Installation

### Dependency

The code has been tested in the following environment:


| Package           | Version |
|-------------------|---------|
| Python            | 3.10.13 |
| PyTorch           | 1.12.1  |
| PyTorch Geometric | 2.4.0   |
| PyTorch Cluster   | 1.6.0   |
| CUDA              | 11.6    |
| networkx          | 2.8.6   |
| numpy             | 1.21.2  |
| pandas            | 1.4.2   |
| scikit-learn      | 1.2.2   |
| scikit-survival   | 0.20.0  |
| scipy             | 1.10.1  |

You can chage the package version according to your need.

### Install via Mamba

You can set up the environment using [Mamba](https://github.com/conda-forge/miniforge).
```
mamba env create -f environment.yaml
mamba activate TxT
```

## Running on your dataset

## 1. Pre-training gene embedding on a biological network

The embedding for each gene, used as input for training, is derived from a pre-trained node2vec model on an undirected biological network.

You can use other methods according to your need.

### 1) Input Data

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

### 2) Embedding Gene

```
python embed_gene.py --network_file <network.csv> --result_dir <result_dir> \
                    --embedding_dim <embedding_dim> --sparse --device <device>
```

*Options*

- `--network_file`: (csv) A network file representing relationships between genes.
- `--result_dir`: (dir) A directory to save output files.
- `--embedding_dim`: (int) The size of each embedding vector (default: 64).
- `--sparse`: An option to control the memory efficiency of storing random walks.
- `--device`: (int) Device number (default: 0).

Check the [script](https://github.com/BonilKoo/TxT/blob/main/embed_gene.py#L14) for other options.

*Output*

- `embedding.csv`: (csv) A csv file representing gene embedding. The gene names are listed in the first column, and the subsequent columns contain the embedding values for each gene in different dimensions.
- `performance.csv`: (csv) A file containing a summary of the model's performance on training, validation and test datasets, including accuracy and AUROC.
- `loss.csv`: (csv) A file containing a record of training and validation metrics for each epoch.
- `arguments.csv`: (csv) A csv file including the argument name and its corresponding value.
- `node2vec_model.pt`: A file for the trained node2vec model's learned parameters.
- `link_prediction.joblib`: A file for the trained logistic regression model for link prediction.

## 2. Running TxT (Training and Evaluation)

### 1) Input Data

(1) *Omics Profile*

* File type: CSV
* Format overview:

    ```
    Sample,CHEK2,FOXA1,GATA3,...
    sample1,2.142685,0.655440,0.103477,...
    sample2,0.657207,0.413347,3.192018,...
    sample3,3.736490,0.171904,1.942691,...
    sample4,4.191718,0.091432,1.700311,...
    ...
    ```

(2) *Clinical Feature*

* File type: CSV
* Format overview:
    - Regression

        ```
        Sample,Age
        sample1,85
        sample2,70
        sample3,50
        sample4,65
        ...
        ```

    - Classification

        ```
        Sample,PAM50
        sample1,LumA
        sample2,Basal
        sample3,LumB
        sample4,Her2
        ...
        ```

    - Survival Prediction

        ```
        Sample,OS_duration,OS_event
        sample1,3313,1
        sample2,4094,0
        sample3,4096,0
        sample4,4079,0
        ...
        ```

    - Multi-task Learning

        ```
        Sample,Age,Size,NHG,PAM50,OS_duration,OS_event
        sample1,85,13,2,LumA,3313,1
        sample2,70,16,3,Basal,4094,0
        sample3,50,15,2,LumB,4096,0
        sample4,65,15,3,Her2,4079,0
        ...
        ```

(3) *Gene Embedding*

* File type: CSV
* Format overview:

    ```
    Gene,0,1,2,...
    CHEK2,-0.776163,-0.510058,-0.566633,...
    FOXA1,-0.133213,-0.649964,-0.390464,...
    GATA3,-0.459504,-0.668453,-0.255762,...
    ...
    ```

(4) *Task List*

Only required for multi-task learning

* File type: CSV
* Format overview:
    - Multi-task Learning

        ```
        name,task
        Age,regression
        Size,regression
        NHG,classification
        PAM50,classification
        OS_duration,survival_time
        OS_event,survival_event
        ```

### 2) Running

* Regression

    ```
    python run_TxT.py --task regression \
    --input_file <omics_profile.csv> --output_file <clinical_feature.csv> \
    --embed_file <gene_embedding.csv> --result_dir <resuir_dir> \
    --scaler [MinMax/Standard/None] --device <device> \
    --xavier_uniform --norm_first
    ```

* Classification

    ```
    python run_TxT.py --task classification \
    --input_file <omics_profile.csv> --output_file <clinical_feature.csv> \
    --embed_file <gene_embedding.csv> --result_dir <result_dir> \
    --scaler [MinMax/Standard/None] --device <device> \
    --xavier_uniform --norm_first
    ```

* Survival Prediction

    ```
    python run_TxT.py --task survival \
    --input_file <omics_profile.csv> --output_file <clinical_feature.csv> \
    --embed_file <gene_embedding.csv> --result_dir <result_dir> \
    --scaler [MinMax/Standard/None] --device <device> \
    --xavier_uniform --norm_first
    ```

* Multi-task Learning

    ```
    python run_TxT.py --task multitask --task_file <task_list.csv> \
    --input_file <omics_profile.csv> --output_file <clinical_feature.csv> \
    --embed_file <gene_embedding.csv> --result_dir <result_dir> \
    --scaler [MinMax/Standard/None] --device <device> \
    --xavier_uniform --norm_first
    ```

*Options*

- `--task`: (str) The type of task to perform. Choose among [regression/classification/survival/multitask].
- `--task_file`: (csv) Only required for multi-task learning. A tsv file that outlines a set of tasks. Each row in the file represents a specific task, and the information is organized into two columns: "task name" and "prediction type".
- `--input_file`: (csv) A omics profile file representing a gene expression dataset where each row corresponds to a sample, and each column, labeled with gene names, represents the expression level of the corresponding gene in the respective sample. The numerical values in the matrix indicate the expression levels of each gene in the corresponding samples.
- `--output_file`: (csv) A file containing clinical feature data. The format is organized with a header line indicating the type of data and subsequent rows containing sample-specific information.
- `--embed_file`: (csv) A csv file representing gene embedding. The gene names are listed in the first column, and the subsequent columns contain the embedding values for each gene in different dimensions.
- `--result_dir`: (dir) A directory to save output files.
- `--scaler`: (str) A data normalization method. You can choose among [MinMax/Standard/None]. The gene expression levels were normalized by using the expression values of the traning set.
- `--device`: (int) Device number.
- `--xavier_uniform`: An option to use Xavier Uniform initialization for model weights to prevent issues like vanishing or exploding gradients during the training process.
- `--norm_first`: An option to perform LayerNorms before other attention and feedforward operations, otherwise after.

Check the [script](https://github.com/BonilKoo/TxT/blob/main/run_TxT.py#L14) for other options.

*Outputs*

- `performance.csv`: (csv) A file containing a summary of the model's performance on training, validation and test datasets.
- `loss.csv`: (csv) A file containing a record of training and validation loss for each epoch.
- `arguments.csv`: (csv) A csv file including the argument name and its corresponding value.
- `TxT.pt`: A file for the trained model's learned parameters.
