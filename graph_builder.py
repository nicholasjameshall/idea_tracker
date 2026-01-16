import json
import random
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any


class NodeType(Enum):
    PERSON = "PERSON"
    SOURCE = "SOURCE"
    IDEA = "IDEA"


class Relationship(Enum):
    HOSTED = "HOSTED"
    APPEARED_ON = "APPEARED_ON"
    DISCUSSED = "DISCUSSED_IN"
    SAID = "SAID"


class ModelClient:
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        # TODO: get actual embeddings
        for text in texts:
            random.seed(text)
            vector = [random.uniform(-1, 1) for _ in range(3)]
            embeddings.append(vector)
        return embeddings


class Node:
    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.properties = properties or {}

    @property
    def embedding_text(self) -> str:
        return (
            self.properties.get('name') or
            self.properties.get('text') or
            self.node_id
        )

    def __repr__(self) -> str:
        return f"<Node {self.node_id}: {self.embedding_text}>"


class Edge:
    def __init__(
        self,
        origin_id: str,
        target_id: str,
        relationship: Relationship
    ):
        self.origin_id = origin_id
        self.target_id = target_id
        self.relationship = relationship

    def __repr__(self) -> str:
        return (
            f"({self.origin_id}) "
            f"-[{self.relationship.name}]-> "
            f"({self.target_id})"
        )


class NodeVectorDatabase:
    def __init__(self, model_client: ModelClient):
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.model_client = model_client

    def add_nodes(self, nodes: List[Node]) -> None:
        if not nodes:
            return

        embeddings = self.model_client.get_batch_embeddings(
            [node.embedding_text for node in nodes]
        )

        for node, vector in zip(nodes, embeddings):
            self.storage[node.node_id] = {
                'node': node,
                'embedding': vector
            }

    def find_similar(
        self,
        new_nodes: List[Node],
        threshold: float = 0.99
    ) -> List[Tuple[str, str]]:
        """
        Calculates similarity between new nodes and existing DB nodes.
        Returns list of tuples: (new_node_id, existing_node_id)
        """
        potential_duplicates = []

        new_texts = [n.embedding_text for n in new_nodes]
        new_embeddings = self.model_client.get_batch_embeddings(new_texts)

        for new_id, new_vec in zip(
            [n.node_id for n in new_nodes],
            new_embeddings
        ):
            for existing_id, data in self.storage.items():
                existing_vec = data['embedding']

                similarity = self._cosine_similarity(new_vec, existing_vec)

                if similarity >= threshold:
                    potential_duplicates.append((new_id, existing_id))

        return potential_duplicates

    def _cosine_similarity(
        self,
        vec_a: List[float],
        vec_b: List[float]
    ) -> float:
        # TODO: replace with numpy
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class Graph:
    def __init__(
        self,
        nodes: Optional[List[Node]] = None,
        edges: Optional[List[Edge]] = None
    ):
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []

    def add_graph(self, other: 'Graph') -> None:
        if not isinstance(other, Graph):
            raise TypeError("Operand must be a Graph")

        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)

    @classmethod
    def from_json(cls, file_name: str) -> 'Graph':
        with open(file_name, 'r') as file:
            graph_data = json.load(file)

        return cls(
            nodes=[
                Node(
                    node_id=node['id'],
                    node_type=NodeType(node['type']),
                    properties=node.get('properties'),
                )
                for node in graph_data.get('nodes', [])
            ],
            edges=[
                Edge(
                    origin_id=edge['origin'],
                    target_id=edge['target'],
                    relationship=Relationship(edge['relationship']),
                )
                for edge in graph_data.get('edges', [])
            ]
        )

    def replace_duplicate(self, old_id: str, new_id: str) -> None:
        print(f"🔄 Merging {old_id} -> {new_id}...")

        # Remove from nodes
        self.nodes = [n for n in self.nodes if n.node_id != old_id]

        # Update edges
        # TODO: update with more efficient way to change edges.
        # Conversion to adjacency map worth it? Will this happen
        # more than once?
        for edge in self.edges:
            if edge.target_id == old_id:
                edge.target_id = new_id
            if edge.origin_id == old_id:
                edge.origin_id = new_id


def is_valid_merge(old_id: str, new_id: str) -> bool:
    response = input(f"Is {old_id} the same as {new_id}? Yes or No. ")
    return response.lower() == "yes"


if __name__ == "__main__":
    model_client = ModelClient()
    vector_db = NodeVectorDatabase(model_client=model_client)
    knowledge_graph = Graph()

    files = ["example_source.json", "example_source.json"]

    for file_name in files:
        try:
            new_graph = Graph.from_json(file_name)
        except FileNotFoundError:
            print(f"File {file_name} not found, skipping.")
            continue

        # Dict containing node IDs mapped to their respective embeddings
        stored_nodes = vector_db.storage

        # List of tuples containing original and duplicate IDs
        duplicates = vector_db.find_similar(new_graph.nodes)

        # Update new graph to remove duplicates
        if duplicates:
            for duplicate in duplicates:
                old_id, new_id = duplicate
                if is_valid_merge(old_id, new_id):
                    new_graph.replace_duplicate(old_id, new_id)

        # Update vector DB to include new node types
        unique_nodes_to_add = []
        for node in new_graph.nodes:
            if node.node_id not in stored_nodes:
                unique_nodes_to_add.append(node)
        vector_db.add_nodes(unique_nodes_to_add)

        # Incorporate new graph into KG
        knowledge_graph.add_graph(new_graph)

    print([node.node_id for node in knowledge_graph.nodes])
    print([edge for edge in knowledge_graph.edges])
