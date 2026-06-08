import os
import sys
from dataclasses import dataclass

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

class Recommender:
    def __init__(self):
        try:
            logging.info(
                "Loading recommendation artifacts"
            )

            self.cosine_sim = load_object(
                os.path.join(
                    "artifacts",
                    "cosine_sim.pkl"
                )
            )

            self.movie_features = load_object(
                os.path.join(
                    "artifacts",
                    "movie_features.pkl"
                )
            )

            self.indices = {
                title: idx
                for idx, title in enumerate(
                    self.movie_features['title']
                )
            }

            logging.info(
                "Artifacts loaded successfully"
            )

        except Exception as e:
            raise CustomException(e, sys)
        
    def recommend_movies(
            self,
            title, 
            n=10,
            min_rating=3.5,
            min_votes=50
            ):
        
        try:
            logging.info(
                f"Generating recommendations for: {title}"
                )
            
            if title not in self.indices:
                raise ValueError(
                    f"Movie '{title}' not found"
                    )
                
            idx = self.indices[title]
            sim_scores = list(
                enumerate(
                self.cosine_sim[idx]
                    )
                    )
            sim_scores = sorted(
                sim_scores,
                key=lambda x: x[1],
                reverse=True
                )

            sim_scores = sim_scores[1:100]
            movie_indices = [i[0] for i in sim_scores]

            recommendations = (
                self.movie_features[['title', 'genres', 'avg_rating', 'rating_count']].iloc[movie_indices].copy()
            )
            recommendations[
                "similarity_score"
                ] = [
                    score
                    for _, score in sim_scores
                    ]
            recommendations = recommendations[
                (recommendations["avg_rating"] >= min_rating)
                &
                (recommendations["rating_count"] >= min_votes)
                ].sort_values(
                    by="similarity_score",
                    ascending=False
                ).head(n)
            logging.info(
                "Recommendations generated successfully"
                )
            return recommendations
        
        except Exception as e:
            raise CustomException(e, sys)