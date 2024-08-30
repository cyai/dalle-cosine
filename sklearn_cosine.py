from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from prompts import prompts


def get_similarity(prompt: str, image_id: int):
    original_prompt = prompts[image_id]
    documents = [prompt, original_prompt]
    count_vectorizer = CountVectorizer(stop_words="english")
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)

    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(
        doc_term_matrix,
        columns=count_vectorizer.get_feature_names_out(),
        index=["prompt", "original_prompt"],
    )
    print(df)
    print(cosine_similarity(df, df))


get_similarity(
    "A big green monster with a small red hat and a blue umbrella in the snow with a snowman",
    1,
)
