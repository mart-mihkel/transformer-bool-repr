import json
import logging
from pathlib import Path
from typing import Annotated, TypedDict
from itertools import combinations

import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import copy

logger = logging.getLogger("boolrepr")

class Batch(TypedDict):
    x: Annotated[Tensor, "batch input"]
    y: Annotated[Tensor, "input"]

class Clustering():
    def __init__(self,
      model: torch.nn.Module,
      model_path: str,
      epochs: list[str],
      eval_data: DataLoader,
      fourier_coefs: list[tuple]
    ):
        self.epochs = epochs
        self.layers = {}
        self.x = [item["x"] for i, item in enumerate(eval_data.dataset.dataset.data) if i in eval_data.dataset.indices]
        self.y = [item["y"] for i, item in enumerate(eval_data.dataset.dataset.data) if i in eval_data.dataset.indices]
        self.fourier_coefs = fourier_coefs
        for epoch in epochs:
            temp_model = copy.deepcopy(model).to("cpu")
            temp_model.load_state_dict(torch.load(model_path + f"/chkpt-{epoch}.pth"))
            
            temp_model.eval()
            hidden_layers = []
            for i, batch in enumerate(iter(eval_data)):
                batch: Batch

                x = batch["x"].to("cpu")
                y = batch["y"].to("cpu")

                with torch.no_grad():
                    y_hat, hidden_layer = temp_model(x, return_layer=True)
                hidden_layers.append(hidden_layer) # (batch_size, embed_dim) 
            epoch_representations = torch.cat(hidden_layers) #  (eval_dataset_size, embed_dim) 
            self.layers[epoch] = epoch_representations

    def cluster_over_epochs(self):
        logger.info("Clustering model representations...")
        number_of_clusters = {}
        for epoch, representations in self.layers.items():
            clusters = self.cluster_layer(representations)
            number_of_clusters[epoch] = clusters
        return number_of_clusters
    
    def cluster_layer(self, representations):
        if len(self.x[0].shape) > 1:
            input_dim = self.x[0].shape[1]
        else:
            input_dim = self.x[0].shape[0]
        best_silhouette, best_k = 0, -1
        for num_clusters in range(2, representations.shape[1]):
            k_means = KMeans(num_clusters)
            cluster_labels = k_means.fit_predict(representations)
            silhouette_val = silhouette_score(representations, cluster_labels)
            #logger.info(f"Silhouette Score (k={num_clusters}): {silhouette_val}")
            if silhouette_val > best_silhouette:
                best_silhouette, best_k = silhouette_val, num_clusters
        return best_k

    def get_fourier_series(self):
        if len(self.x[0].shape) > 1:
            input_dim = self.x[0].shape[1]
        else:
            input_dim = self.x[0].shape[0]
        
        expanded_terms = []
        for term in self.fourier_coefs:
            term_values = []
            for row in self.x:
                temp_product = term[1]
                for index in term[0]:
                    if len(row) == 1:
                        temp_product *= (-1)**row[0][index].item()
                    else:
                        temp_product *= (-1)**row[index].item()
                term_values.append(temp_product)
            expanded_terms.append(torch.tensor(term_values))
        return torch.stack(expanded_terms)
    
    def correlate(self):
        epoch = max(self.layers.keys())
        representation = self.layers[epoch].T
        expanded_terms = self.get_fourier_series()
        logger.info(f"Comparing {len(representation)} neurons with {len(expanded_terms)} Fourier terms")
        for i in range(len(expanded_terms[0])):
            print(expanded_terms[:, i].sum())
        u = torch.cov(torch.cat([representation, expanded_terms]))
        
        # Extract cross-correlations between representation vars and expanded terms
        num_repr = representation.shape[0]
        num_terms = expanded_terms.shape[0]
        
        # Top-right block of covariance matrix
        cross_corr = u[:num_repr, num_repr:num_repr+num_terms]
        for i in range(cross_corr.shape[0]):
            for j in range(cross_corr.shape[1]):
                if abs(cross_corr[i][j]) > 0.1: # Decently strong correlation
                    print(f"Representation var {i} correlates with Fourier term {self.fourier_coefs[j][0]} with correlation {cross_corr[i][j]}")
        return None

                

