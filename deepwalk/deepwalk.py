from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
import random
import fasttext
import tempfile

class DeepWalk:
    def __init__(self, graph, walk_length=80, walks_per_vertex=10):
        self.graph = graph
        self.walk_length = walk_length
        self.walks_per_vertex = walks_per_vertex

    def simulate_random_walks(self):
        random_walks = []
        for _ in range(self.walks_per_vertex):
            O = list(self.graph.nodes()).copy()
            random.shuffle(O)
            for vi in O:
                Wvi = self._random_walk(vi)
                random_walks.append(Wvi)
        return random_walks


    def _random_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            neighbors = list(self.graph.neighbors(walk[-1]))
            if len(neighbors) == 0:
                break
            next_node = random.choice(neighbors)
            walk.append(next_node)
        return walk
    
    def train(self):
        sentences = self.simulate_random_walks()
        lines = [" ".join(map(str, walk)) for walk in sentences]
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:
            tmp.write("\n".join(lines))
            tmp.flush()

            self.model = fasttext.train_unsupervised(
                input=tmp.name,
                model='skipgram',
                dim=128,
                ws=5,
                epoch=10,
                minCount=0,
                neg=0,
                loss='hs',
                thread=4,
                minn=0, maxn=0
            )

    def get_embedding(self, node):
        return self.model.get_word_vector(str(node))
    
    def get_embeddings(self):
        embeddings = []
        for node in range(self.graph.number_of_nodes()):
            embeddings.append(self.get_embedding(node))
        return torch.tensor(embeddings)
    
