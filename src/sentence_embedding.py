from sentence_transformers import SentenceTransformer

class SentenceEmbedding:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')

    def getSentenceEncoderModel(self):
        return self.sentence_model

    def getSentenceEmbedding(self, sentence: str, convert_to_tensor: bool):
        sentence_embedding = self.sentence_model.encode(sentence, show_progress_bar=True, convert_to_tensor=convert_to_tensor)
        return sentence_embedding

    def getListOfSentenceEmbedding(self, sentence_list: list[str], convert_to_tensor: bool):
        sentence_embeddings = self.sentence_model.encode(sentence_list, show_progress_bar=True, convert_to_tensor=convert_to_tensor)
        return sentence_embeddings
