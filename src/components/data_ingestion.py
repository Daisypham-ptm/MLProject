import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    movies_data_path = os.path.join(
        "artifacts",
        "movies.csv"
    )

    ratings_data_path = os.path.join(
        "artifacts",
        "ratings.csv"
    )

    tags_data_path = os.path.join(
        "artifacts",
        "tags.csv"
    )


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(
            "Entered data ingestion component"
        )

        try:
            movies = pd.read_csv(
                "notebook/data/movies.csv"
            )

            ratings = pd.read_csv(
                "notebook/data/ratings.csv"
            )

            tags = pd.read_csv(
                "notebook/data/tags.csv"
            )

            logging.info(
                "MovieLens datasets loaded successfully"
            )

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            movies.to_csv(
                self.ingestion_config.movies_data_path,
                index=False
            )

            ratings.to_csv(
                self.ingestion_config.ratings_data_path,
                index=False
            )

            tags.to_csv(
                self.ingestion_config.tags_data_path,
                index=False
            )

            logging.info(
                "Raw datasets saved to artifacts folder"
            )

            return (
                movies,
                ratings,
                tags
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()

    movies, ratings, tags = (
        obj.initiate_data_ingestion()
    )

    print(
        f"Movies: {movies.shape}"
    )

    print(
        f"Ratings: {ratings.shape}"
    )

    print(
        f"Tags: {tags.shape}"
    )