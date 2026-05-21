import streamlit as st
import pandas as pd
from model.recommender import recommend_courses

# Page Config
st.set_page_config(
    page_title="Intern Learning Path Recommender",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Personalized Learning Path Recommender")

st.write("""
This system recommends personalized learning courses
for interns using Collaborative Filtering and
Matrix Factorization.
""")

# Load interns data
interns = pd.read_csv("data/interns.csv")

# Create dropdown
intern_names = interns['name'].tolist()

selected_name = st.selectbox(
    "Select Intern",
    intern_names
)

# Get intern ID
selected_intern = interns[
    interns['name'] == selected_name
]

intern_id = int(selected_intern['intern_id'].values[0])

department = selected_intern['department'].values[0]
skill_level = selected_intern['skill_level'].values[0]

# Display profile
st.subheader("👤 Intern Profile")

st.write(f"**Name:** {selected_name}")
st.write(f"**Department:** {department}")
st.write(f"**Skill Level:** {skill_level}")

# Recommendation Button
if st.button("Get Learning Recommendations"):

    recommendations = recommend_courses(intern_id)

    st.subheader("📚 Recommended Learning Path")

    if recommendations:

        for i, course in enumerate(recommendations, start=1):

            st.markdown(f"""
### {i}. {course['Course Name']}

- **Course ID:** {course['Course ID']}
- **Category:** {course['Category']}
- **Difficulty:** {course['Difficulty']}
            """)

    else:
        st.warning("No recommendations available.")