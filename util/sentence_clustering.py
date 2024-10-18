from typing import Type, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from transformers import AutoTokenizer, AutoModel
import torch


class SentenceClustering(object):
    def __init__(
            self,
            model_path: str = 'distilbert-base-uncased',
            minimum_clusters_num: int = 5,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            clean_up_tokenization_spaces=False,
        )
        self._model = AutoModel.from_pretrained(model_path)
        self._mini_num = minimum_clusters_num

    def __call__(
            self,
            sentences: List[str],
    ) -> List[str]:
        return self.extract_representative_sentences(sentences)

    @staticmethod
    def find_optimal_clusters(
            data: np.ndarray,
            max_k: int,
    ) -> int:
        silhouette_scores = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(data)
            score = silhouette_score(data, km.labels_)
            silhouette_scores.append(score)
        return silhouette_scores.index(max(silhouette_scores)) + 2

    def get_sentence_embedding(
            self,
            sentence: str,
            return_tensors: str = "pt",
            padding: bool = True,
            truncation: bool = True,
            max_length: int = 512,
    ) -> np.ndarray:
        inputs = self._tokenizer(
            sentence,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def extract_representative_sentences(
            self,
            sentences: List[str],
    ) -> List[str]:
        embeddings = np.array([self.get_sentence_embedding(s) for s in sentences])

        max_k = len(sentences) - 1
        optimal_k = self.find_optimal_clusters(embeddings, max_k)

        # n_clusters = max(optimal_k, int(len(sentences) * self._mini_rate))
        n_clusters = max(optimal_k, self._mini_num)

        km = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = km.fit_predict(embeddings)

        representative_sentences = []
        for cluster_id in range(n_clusters):
            cluster_center = km.cluster_centers_[cluster_id]
            cluster_sentences = [sent for sent, label in zip(sentences, cluster_labels) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_labels == cluster_id]
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            representative_index = np.argmin(distances)
            representative_sentences.append(cluster_sentences[representative_index])

        return representative_sentences


if __name__ == "__main__":
    """
    python -m util.sentence_clustering
    """

    lists = [
        ['Lack of complexity increase in subsequent stages.', 'Lack of clear definition of variables.'],
        [
            'Insufficient specification of parameters', 'Circular reasoning in the derivation of the formula',
            'Lack of consideration for external factors', 'Incorrect application of mathematical principles',
            'Overly complex solution for a simple problem', 'Failure to account for unit conversions',
            'Inability to derive a numerical answer', 'Lack of clarity in presenting the solution',
            "Ignoring the problem statement's requirements", 'Unnecessary repetition of steps'],
        [
            'Lack of clear definition of variables.', 'Insufficient simplification of equations.',
            'Failure to isolate the variable.', 'Inability to solve for the variable.',
            'Inadequate application of algebraic concepts.', 'Inability to simplify complex equations.',
            'Inadequate solution to the equation.', 'Failure to provide a clear and concise solution.',
            'Inability to apply mathematical concepts to solve the problem.',
            'Inadequate explanation of the solution process.'],
    ]

    flattened_list = [
        item
        for sublist in lists
        for item in sublist
    ]
    print(len(flattened_list))

    unique_list = list(set(flattened_list))
    print(len(unique_list))

    sc = SentenceClustering(
        'distilbert-base-uncased',
        minimum_rate_after_reduction=0.5,
    )

    representative_sentences = sc(unique_list)

    for s in representative_sentences:
        print(s)
