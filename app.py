import streamlit as st
from src.pipeline.recommender import Recommender


# ==========================
# Load Recommender
# ==========================
@st.cache_resource
def load_recommender():
    return Recommender()


recommender = load_recommender()


# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #FF4B4B;
}

.sub-title {
    text-align: center;
    font-size: 1.2rem;
    color: #BBBBBB;
    margin-bottom: 30px;
}

.movie-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e1e1e;
    margin-bottom: 15px;
    border: 1px solid #333333;
}

.movie-title {
    font-size: 1.3rem;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# Header
# ==========================
st.markdown(
    '<div class="main-title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Content-Based Filtering using TF-IDF and Cosine Similarity</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================
# Movie List
# ==========================
movie_list = sorted(
    recommender.movie_features["title"].tolist()
)

default_index = 0

if "Toy Story (1995)" in movie_list:
    default_index = movie_list.index(
        "Toy Story (1995)"
    )


# ==========================
# Sidebar
# ==========================
st.sidebar.title("⚙️ Settings")

selected_movie = st.sidebar.selectbox(
    "Select Movie",
    movie_list,
    index=default_index
)

n_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    1,
    20,
    10
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0,
    5.0,
    3.5,
    0.1
)

min_votes = st.sidebar.slider(
    "Minimum Votes",
    0,
    200,
    50
)

recommend_btn = st.sidebar.button(
    "🎥 Recommend Movies",
    use_container_width=True
)


# ==========================
# Recommendation
# ==========================
if recommend_btn:

    with st.spinner("Finding similar movies..."):

        recommendations = (
            recommender.recommend_movies(
                title=selected_movie,
                n=n_recommendations,
                min_rating=min_rating,
                min_votes=min_votes
            )
        )

    if recommendations.empty:

        st.warning(
            "No recommendations found. Try lowering rating or vote thresholds."
        )

    else:

        st.success(
            f"Found {len(recommendations)} recommendations."
        )

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Movies Found",
                len(recommendations)
            )

        with col2:
            st.metric(
                "Min Rating",
                min_rating
            )

        with col3:
            st.metric(
                "Min Votes",
                min_votes
            )

        st.divider()

        st.subheader("🎥 Recommended Movies")

        for _, movie in recommendations.iterrows():

            st.markdown(
                f"""
                <div class="movie-card">

                <div class="movie-title">
                {movie['title']}
                </div>

                <br>

                🎭 <b>Genres:</b>
                {movie['genres']}

                <br><br>

                ⭐ <b>Rating:</b>
                {movie['avg_rating']:.2f}

                &nbsp;&nbsp;&nbsp;

                👥 <b>Votes:</b>
                {int(movie['rating_count'])}

                &nbsp;&nbsp;&nbsp;

                🎯 <b>Similarity:</b>
                {movie['similarity_score']:.3f}

                </div>
                """,
                unsafe_allow_html=True
            )