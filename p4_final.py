from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import time

# Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MPNet-base-v2")

# Convert DataFrame rows into LangChain Documents
def create_documents_from_dataframe(df: pd.DataFrame):
    return [
        Document(page_content=row["Scenario"], metadata={"resolution": row["Resolution"]})
        for _, row in df.iterrows()
    ]

# Initialize FAISS Vector Store
def initialize_faiss_store(documents):
    return FAISS.from_documents(documents, embedding_model)

# Function to sanitize input and scenarios
def sanitize_text(text):
    text = text.strip().lower()  # Remove extra spaces and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Query Resolution with Semantic Matching for Multiple Resolutions
def query_resolution(faiss_store, user_input, df):
    start_time = time.time()  # Start timing
    # Sanitize user input
    user_input_sanitized = sanitize_text(user_input)

    # Embed user input
    user_input_embedding = embedding_model.embed_query(user_input_sanitized)

    # Check for semantic similarity in the DataFrame using Cosine Similarity
    df["Scenario_Embedding"] = df["Scenario"].apply(embedding_model.embed_query)
    df["Similarity"] = df["Scenario_Embedding"].apply(
        lambda x: cosine_similarity([x], [user_input_embedding])[0][0]
    )

    # Filter rows where similarity is greater than or equal to the threshold (0.85)
    matching_rows = df[df["Similarity"] >= 0.65]

    if not matching_rows.empty:
        # Find the highest similarity score
        max_similarity_score = matching_rows["Similarity"].max()

        # Filter the rows where the similarity score equals the highest similarity score
        highest_similarity_rows = matching_rows[matching_rows["Similarity"] == max_similarity_score]

        # Print the rows with the highest similarity score
        print(f"Matching Rows with Highest Similarity Score ({max_similarity_score:.4f}):")
        print(highest_similarity_rows)

        # Get resolutions corresponding to these rows
        resolutions = highest_similarity_rows["Resolution"].tolist()
        return f"Suggested Resolutions with Similarity Score {max_similarity_score:.4f}:\n" + "\n".join(
            f"- {res}" for res in resolutions)
    else:
        return "No resolution found with high similarity score. Please send mail to XYZ."

# Main Execution with While Loop
if __name__ == "__main__":
    df = pd.read_csv("CPU_Issues_Dataset_Large.csv")
    df = df.drop(df.columns[0], axis=1)
    df = df.drop_duplicates()

    documents = create_documents_from_dataframe(df)

    # Initialize FAISS Vector Store
    faiss_store = initialize_faiss_store(documents)

    print("Enter 'exit' to quit the program.\n")

    while True:
        user_input = input("Enter a scenario: ").strip()
        if user_input.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        result = query_resolution(faiss_store, user_input, df)
        print("\nSuggested Resolution(s):")
        print(result)
        print("\n" + "-" * 50 + "\n")