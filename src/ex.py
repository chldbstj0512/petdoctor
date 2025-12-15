from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# 🔥 전체 삭제
index.delete(delete_all=True)

print("✅ Pinecone index cleared.")
