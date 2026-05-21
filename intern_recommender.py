import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# ----------------------
# DATA
# ----------------------
data = {
    "Python Basics":     [5, 3, 0, 1],
    "Machine Learning":  [4, 0, 0, 1],
    "Data Analysis":     [5, 4, 0, 0],
    "Deep Learning":     [0, 0, 5, 4],
    "SQL Basics":        [3, 5, 4, 0],
    "Web Development":   [0, 3, 5, 4],
}

interns = ["Ali", "Sara", "Ahmed", "Zain"]

df = pd.DataFrame(data, index=interns)

print("\n📊 Original Matrix:\n")
print(df)

# ----------------------
# MODEL
# ----------------------
model = NMF(n_components=2, random_state=42, max_iter=500)

W = model.fit_transform(df)
H = model.components_

pred = np.dot(W, H)

pred_df = pd.DataFrame(pred, index=interns, columns=df.columns)

print("\n🤖 Predicted Ratings:\n")
print(pred_df.round(2))

# ----------------------
# RECOMMEND FUNCTION
# ----------------------
def recommend(user):
    print(f"\n🎯 Recommendations for {user}:\n")

    known = df.loc[user]
    scores = pred_df.loc[user]

    scores = scores[known == 0]
    top = scores.sort_values(ascending=False)

    print(top.head(3))


# ----------------------
# RUN
# ----------------------
recommend("Ali")
recommend("Sara")
recommend("Ahmed")
recommend("Zain")