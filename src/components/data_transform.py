import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    movie_features_path = os.path.join(
        "artifacts",
        "movie_features.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = (
            DataTransformationConfig()
        )

    def initiate_data_transformation(
        self,
        movies,
        ratings,
        tags
    ):

        try:

            logging.info(
                "Starting data transformation"
            )

            rating_stats = (
                ratings
                .groupby("movieId")
                .agg(
                    avg_rating=("rating", "mean"),
                    rating_count=("rating", "count")
                )
                .reset_index()
            )

            tag_stats = (
                tags
                .groupby("movieId")
                .agg(
                    tag_count=("tag", "count")
                )
                .reset_index()
            )

            movie_features = (
                movies
                .merge(
                    rating_stats,
                    on="movieId",
                    how="left"
                )
                .merge(
                    tag_stats,
                    on="movieId",
                    how="left"
                )
            )

            movie_features["avg_rating"] = (
                movie_features["avg_rating"]
                .fillna(0)
            )

            movie_features["rating_count"] = (
                movie_features["rating_count"]
                .fillna(0)
            )

            movie_features["tag_count"] = (
                movie_features["tag_count"]
                .fillna(0)
            )

            tag_text = (
                tags
                .groupby("movieId")["tag"]
                .apply(
                    lambda x: " ".join(
                        x.astype(str)
                    )
                )
                .reset_index()
            )

            movie_features = movie_features.merge(
                tag_text,
                on="movieId",
                how="left"
            )

            movie_features["tag"] = (
                movie_features["tag"]
                .fillna("")
            )

            movie_features["content"] = (
                movie_features["genres"]
                + " "
                + movie_features["tag"]
            )

            logging.info(
                "Saving movie features"
            )

            save_object(
                file_path=self.data_transformation_config.movie_features_path,
                obj=movie_features
            )

            logging.info(
                "Data transformation completed"
            )

            return movie_features

        except Exception as e:
            raise CustomException(e, sys)