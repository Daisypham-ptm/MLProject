from src.pipeline.recommender import Recommender

if __name__ == "__main__":

    recommender = Recommender()

    # recommendations = recommender.recommend_movies(
    #     "Toy Story (1995)",
    #     n=10
    # )
    recommendations = recommender.recommend_movies(
    "Jumanji (1995)",
    n=10
    )

    print("\nRecommended Movies:\n")
    print(
        recommendations.to_string(
            index=False
        )
    )