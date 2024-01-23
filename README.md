# Ditto
Ditto: deep multi-task learning with modeling gene interaction using transformer for predicting clinical features and survival

<p align="center">
	<img src="./img/Ditto.svg" />
</p>

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

### 1. Pre-training gene embedding on a biological network

The embedding for each gene, used as input for training, is derived from a pre-trained node2vec model on a biological network.

You can use other methods according to your need.

#### Input data

*network*

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

#### Training

```
python embed_gene.py --network_file <network.csv> --result_dir <result_dir>
    --embedding_dim <embedding_dim> --sparse
    --device <device>
```

*Options*

- `--network_File`: (csv) A network file representing relationships between genes.
- `--result_dir`: (dir) A directory to save output files.
- `--embedding_dim`: (int) The size of each embedding vector.
- `--sparse`: An option to control the memory efficiency of storing random walks.
- `--device`: (int) Device number.

Check the [script](https://github.com/BonilKoo/Ditto/blob/main/src/embed_gene.py#L19) for other options.

*Output*

- `embedding.csv`: (csv) A csv file representing gene embedding. The gene names are listed in the first column, and the subsequent columns contain the embedding values for each gene in different dimensions.

### 2. Training and Evaluation

#### Input data

*Omics profile*

* File type: CSV
* Format overview:

    ```
    Sample,CHEK2,FOXA1,GATA3,...
    sample1,2.142685,0.655440,0.103477,...
    sample2,0.657207,0.413347,3.192018,...
    sample3,3.651882,0.091432,0.103477,...
    ...
    ```

*Clinical features*

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

*Task list*

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

*gene embedding*

* File type: CSV
* Format overview:

    ```
    Gene,0,1,2,...
    CHEK2,-0.776163,-0.510058,-0.566633,...
    FOXA1,-0.133213,-0.649964,-0.390464,...
    GATA3,-0.459504,-0.668453,-0.255762,...
    ...
    ```

#### Training

