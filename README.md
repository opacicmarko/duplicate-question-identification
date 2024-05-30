# duplicate-question-identification

## Dataset
The [Quora Question Pairs](https://www.kaggle.com/competitions/quora-question-pairs) dataset is used in this paper. Download the [data](https://www.kaggle.com/competitions/quora-question-pairs/data) and extract the downloaded file into the `src/data` directory.

As part of the main `duplicate_questions.ipynb` notebook, the dataset from `train.csv` is be split into training, validation and test sets in the Dataset section of the notebook. This new split is saved into `data/mojave` by default.

## Jupyter Notebook
The Python dependencies are listed in the `requirements.txt` file. You can install them by running the following command from the `src` directory:
```
pip install -r requirements.txt
```
