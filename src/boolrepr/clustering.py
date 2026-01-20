import json
import logging
from pathlib import Path
from typing import Annotated, TypedDict
from itertools import combinations

import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import copy
import random

logger = logging.getLogger("boolrepr")

class Batch(TypedDict):
    x: Annotated[Tensor, "batch input"]
    y: Annotated[Tensor, "input"]

class Clustering():
    def __init__(self,
      model: torch.nn.Module,
      model_path: str,
      epochs: list[int],
      eval_data: DataLoader,
      fourier_coefs: list[tuple],
      relevant_vars: list
    ):
        self.epochs = epochs
        self.layers = {}
        self.x = [item["x"] for i, item in enumerate(eval_data.dataset.dataset.data) if i in eval_data.dataset.indices]
        self.y = [item["y"] for i, item in enumerate(eval_data.dataset.dataset.data) if i in eval_data.dataset.indices]
        self.fourier_coefs = fourier_coefs
        self.relevant_vars = []

        for var in relevant_vars:
            if type(var) == tuple:
                self.relevant_vars.append(var[0])
            else:
                self.relevant_vars.append(var)

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
        # HDBSCAN approach
        hdbscan = HDBSCAN(store_centers="centroid", copy=False)
        hdbscan.fit_predict(representations)
        #logger.info(set(labels))
        best_k = len(hdbscan.centroids_)

        # K-Means approach
        """
        best_silhouette, best_k = 0, -1
        for num_clusters in range(2, representations.shape[1]):
            #k_means = KMeans(num_clusters)
            #cluster_labels = k_means.fit_predict(representations)
            
            silhouette_val = silhouette_score(representations, cluster_labels)
            #logger.info(f"Silhouette Score (k={num_clusters}): {silhouette_val}")
            if silhouette_val > best_silhouette:
                best_silhouette, best_k = silhouette_val, num_clusters
        """
        return best_k

    def visualize(self, cluster_map, eval_accs, train_accs, out_name):


        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Number of clusters', color=color)
        ax1.plot(cluster_map.keys(), cluster_map.values(), color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.annotate(xy=(max(self.epochs),cluster_map[max(self.epochs)]), xytext=(5,0), textcoords='offset points', text=str(cluster_map[max(self.epochs)]), va='center')
        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(cluster_map.keys(), eval_accs, color='darkblue', label="Eval")
        ax2.plot(cluster_map.keys(), train_accs, color='lightblue', label="Train")
        ax2.tick_params(axis='y', labelcolor=color)
            
        fig.tight_layout()
        plt.legend()
        plt.savefig(out_name)

    def get_fourier_series(self):
        if len(self.x[0].shape) > 1:
            input_dim = self.x[0].shape[1]
        else:
            input_dim = self.x[0].shape[0]
        print(f"input_dim: {input_dim}")
        selected_terms = []
        expanded_terms = []
        for i, term in enumerate(self.fourier_coefs):
            term_values = []
            if term[1] == 0:
                continue
            for row in self.x:
                temp_product = term[1]
                for index in term[0]:
                    if len(row) == 1:
                        temp_product *= row[0][index].item()
                    else:
                        temp_product *= row[index].item()
                selected_terms.append(i)
                term_values.append(temp_product)
            expanded_terms.append(torch.tensor(term_values))
        return selected_terms, torch.stack(expanded_terms)
    
    def correlate(self):
        epoch = max(self.layers.keys())
        representation = self.layers[epoch].T
        terms, expanded_terms = self.get_fourier_series()
        logger.info(f"Comparing {len(representation)} neurons with {len(expanded_terms)} Fourier terms")
        
        u = torch.cov(torch.cat([representation, expanded_terms]))
        
        # Extract cross-correlations between representation vars and expanded terms
        num_repr = representation.shape[0]
        num_terms = expanded_terms.shape[0]
        
        # Top-right block of covariance matrix
        cross_corr = u[:num_repr, num_repr:num_repr+num_terms]
        for i in range(cross_corr.shape[0]):
            for j in range(cross_corr.shape[1]):
                if abs(cross_corr[i][j]) > 0.1: # Decently strong correlation
                    print(f"Representation var {i} correlates with Fourier term {self.fourier_coefs[terms[j]][0]} with correlation {cross_corr[i][j]}")
        return None

    def test_ood(self, model : torch.nn.Module):
        num_terms = len(self.x[0])
        transformer = False
        if num_terms == 1:
            transformer = True
            num_terms = len(self.x[0][0])
        for delta in [0, 2, 10, 100, 10000]:
            new_x = []

            for row in self.x:
                row = row.tolist()
                for i in range(num_terms):
                    if i not in self.relevant_vars:
                        if transformer:
                            if random.random() < 0.5:
                                row[0][i] -= delta
                            else:
                                row[0][i] += delta
                        else:
                            if random.random() < 0.5:
                                row[i] -= delta
                            else:
                                row[i] += delta
                new_x.append(row)
            dataset = TensorDataset(torch.tensor(new_x), torch.tensor(self.y))
            data_loader = DataLoader(dataset)
            model.eval()
            eval_acc = 0
            for i, batch in enumerate(iter(data_loader)):
                batch: Batch

                x = batch[0].to("cpu")
                y = batch[1].to("cpu")

                with torch.no_grad():
                    y_hat = model(x, return_layer=False)
                #y_hat = ((-1) * y_hat + 1)/2
                y = ((-1) * y + 1)/2
                eval_acc += ((y_hat > 0.5) == y.bool()).float().mean().item()
            logger.info(f"Average eval accuracy with delta={delta}: {eval_acc/len(data_loader)}")

            


