from qdrant_client import QdrantClient

class VectorDatabase:
    def __init__(self):
        self.conn = QdrantClient(':memory:')
        
    