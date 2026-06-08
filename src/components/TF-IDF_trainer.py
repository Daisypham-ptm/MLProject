import os
import sys
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    tfidf_file_path = os.path.join(
        "artifacts",
        "tfidf.pkl"
    )

    cosine_sim_file_path = os.path.join(
        "artifacts",
        "cosine_sim.pkl"
    )

    movie_features_file_path = os.path.join(
    "artifacts",
    "movie_features.pkl"
    )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, movie_features):
        try:
            logging.info(
                "Starting TF-IDF vectorization"
            )

            tfidf = TfidfVectorizer(
                stop_words= "english"
            )

            tfidf_matrix = tfidf.fit_transform(
                movie_features['content']
            )

            logging.info(
                "Computing cosine similarity matrix"
            )

            cosine_sim = cosine_similarity(
                tfidf_matrix,
                tfidf_matrix
            )

            logging.info(
                "Saving TF-IDF model"
            )
            
            save_object(
                file_path=self.model_trainer_config.tfidf_file_path,
                obj=tfidf
            )

            logging.info(
                "Saving cosine similarity matrix"
            )

            save_object(
                file_path=self.model_trainer_config.cosine_sim_file_path,
                obj=cosine_sim
            )
            logging.info(
                "Saving movie features"
            )

            save_object(
                file_path=self.model_trainer_config.movie_features_file_path,
                obj=movie_features
            )

            logging.info(
                "Model training completed successfully"
            )

            return {
                "num_movies": len(movie_features),
                "num_features": tfidf_matrix.shape[1]
            }

        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    movies, ratings, tags = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()

    movie_features = data_transformation.initiate_data_transformation(
        movies,
        ratings,
        tags
    )
    modeL_trainer = ModelTrainer()

    result = modeL_trainer.initiate_model_trainer(
        movie_features
    )

    print(result)

