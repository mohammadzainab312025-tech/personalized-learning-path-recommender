import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

# Load datasets
ratings = pd.read_csv("data/ratings.csv")
courses = pd.read_csv("data/courses.csv")
interns = pd.read_csv("data/interns.csv")

# Create user-item matrix
user_item_matrix = ratings.pivot(
    index='intern_id',
    columns='course_id',
    values='rating'
).fillna(0)

# Train Matrix Factorization Model
model = NMF(
    n_components=3,
    init='random',
    random_state=42,
    max_iter=500
)

# Learn latent features
user_features = model.fit_transform(user_item_matrix)
item_features = model.components_

# Predict ratings
predicted_ratings = np.dot(user_features, item_features)

# Convert to DataFrame
predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

def recommend_courses(intern_id, top_n=5):

    if intern_id not in predicted_df.index:
        return []

    # Get predicted scores
    scores = predicted_df.loc[intern_id]

    # Courses already taken
    enrolled_courses = ratings[
        ratings['intern_id'] == intern_id
    ]['course_id'].tolist()

    # Remove completed courses
    recommendations = scores.drop(enrolled_courses)

    # Top recommendations
    top_courses = recommendations.sort_values(
        ascending=False
    ).head(top_n)

    recommended_list = []

    for course_id in top_courses.index:

        course_data = courses[
            courses['course_id'] == course_id
        ]

        recommended_list.append({
            "Course ID": int(course_id),
            "Course Name": course_data['course_name'].values[0],
            "Category": course_data['category'].values[0],
            "Difficulty": course_data['difficulty'].values[0]
        })

    return recommended_list