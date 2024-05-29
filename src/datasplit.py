import pandas as pd
from sklearn.model_selection import train_test_split

def make_dataset_split(data_path: str, train_path: str, validation_path: str, test_path: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(data_path)
    # Use 20% of data for the test set
    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['is_duplicate'], random_state=random_state)
    # Use 1/4 of the remaining for a 60/20/20 train/validation/test split
    train_df, validation_df = train_test_split(train_df, test_size=0.25, stratify=train_df['is_duplicate'], random_state=random_state)

    train_df.to_csv(train_path)
    validation_df.to_csv(validation_path)
    test_df.to_csv(test_path)

    return train_df, validation_df, test_df