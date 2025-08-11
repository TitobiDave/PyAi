from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
class vectordb:
 def __init__(self):
    self.pc = Pinecone(api_key="pcsk_5SF2tR_6fVHudP8xgnHzGNM2nB5BYAwQ54brNwyqnixa5Gc1bLWak5N2oizz2SuLGEyjeZ")
    self.index = self.pc.Index("langchainvector")
    self.model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone



# Recommendation function
 def recommend_products(self,query, top_k=5):
    if not isinstance(query, str) or len(query.strip()) < 3:
        return {"error": "Invalid query. Please enter a valid product request."}
    
    try:
        # Vectorize the query
        query_vector = self.model.encode([query])[0].tolist()
        results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        print(f"results is {results}")
        matches = [
            {
                "score": round(match['score'], 4),
                "product_id": match['id'],
                "description": match['metadata']['description']
            }
            for match in results.get('matches', [])
        ]
        if not matches:
            return {
                "matches": [],
                "response": "Sorry, we couldn't find any similar products."
            }

        return {
            "matches": matches
        }

    except Exception as e:
        return {"error": f"An error occurred while processing the query: {str(e)}"}  
