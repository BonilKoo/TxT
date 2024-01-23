# Ditto
Ditto: deep multi-task learning with modeling gene interaction using transformer for predicting clinical features and survival

<p align="center">
	<img src="./img/Ditto.svg" />
</p>

## Installation

### Dependency

The code has been test in the following environment:


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

### 1 Pre-training gene embedding on a biological network

The embedding for each gene, used as input for training, is derived from a pre-trained node2vec model on a biological network.

You can use other embedding methods according to your need.

#### Data

*network file*

* File type: CSV
* Format Overview:
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
    --device <device> --seed <seed>
```

*Options*

- `--network_File`: (csv) A network file representing relationships between genes.
- `--result_dir`: (dir) A directory to save output files.
- `--embedding_dim`: (int) The size of each embedding vector.
- `--sparse`: An option to control the memory efficiency of storing random walks.
- `--device`: (int) Device number.

*Output*

- `embedding.csv`: (csv) A csv file representing gene embedding. The gene names are listed in the first column, and the subsequent columns contain the embedding values for each gene in different dimensions.

### 2 Training and Evaluation

#### Data

#### Training

