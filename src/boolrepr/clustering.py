import copy
import logging
import random
from typing import Annotated, TypedDict

import torch
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics.cluster import silhouette_score
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("boolrepr")


class Batch(TypedDict):
    x: Annotated[Tensor, "batch input"]
    y: Annotated[Tensor, "input"]


class Clustering:
    def __init__(
        self,
        model: Module,
        model_path: str,
        epochs: list[int],
        eval_data: DataLoader,
        fourier_coefs: list[tuple] | None,
        relevant_vars: list[tuple[int, int]],
    ):
        self.epochs = epochs
        self.layers: dict[int, Tensor] = {}

        self.x: list[Tensor] = [
            item["x"]
            for i, item in enumerate(eval_data.dataset.dataset.data)  # type: ignore
            if i in eval_data.dataset.indices  # type: ignore
        ]

        self.y: list[Tensor] = [
            item["y"]
            for i, item in enumerate(eval_data.dataset.dataset.data)  # type: ignore
            if i in eval_data.dataset.indices  # type: ignore
        ]

        self.fourier_coefs = fourier_coefs
        self.relevant_vars: list[tuple[int, int] | int] = []

        for var in relevant_vars:
            if isinstance(var, tuple):
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

                with torch.no_grad():
                    y_hat, hidden_layer = temp_model(x, return_layer=True)

                # (batch_size, embed_dim)
                hidden_layers.append(hidden_layer)

            #  (eval_dataset_size, embed_dim)
            epoch_representations = torch.cat(hidden_layers)

            self.layers[epoch] = epoch_representations

    def cluster_over_epochs(self) -> dict[int, int]:
        logger.info("Clustering model representations...")
        number_of_clusters = {}
        for epoch, representations in self.layers.items():
            clusters = self.cluster_hdbscan(representations)
            number_of_clusters[epoch] = clusters

        return number_of_clusters

    def cluster_hdbscan(self, representations: Tensor) -> int:
        hdbscan = HDBSCAN(store_centers="centroid", copy=False)
        hdbscan.fit_predict(representations)
        best_k = len(hdbscan.centroids_)
        return best_k

    def cluster_kmeans(self, representations: Tensor) -> int:
        best_silhouette, best_k = 0, -1
        for num_clusters in range(2, representations.shape[1]):
            k_means = KMeans(num_clusters)
            cluster_labels = k_means.fit_predict(representations)
            silhouette_val = silhouette_score(representations, cluster_labels)
            if silhouette_val > best_silhouette:
                best_silhouette, best_k = silhouette_val, num_clusters

        return best_k

    def get_fourier_series(self) -> tuple[list[int], Tensor]:
        if self.fourier_coefs is None:
            raise ValueError("No fourier coefs")

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
        if self.fourier_coefs is None:
            raise ValueError("No fourier coefs")

        epoch = max(self.layers.keys())
        representation = self.layers[epoch].T
        terms, expanded_terms = self.get_fourier_series()
        logger.info(
            f"Comparing {len(representation)} neurons with {len(expanded_terms)} Fourier terms"
        )

        u = torch.cov(torch.cat([representation, expanded_terms]))

        # Extract cross-correlations between representation vars and expanded terms
        num_repr = representation.shape[0]
        num_terms = expanded_terms.shape[0]

        # Top-right block of covariance matrix
        cross_corr = u[:num_repr, num_repr : num_repr + num_terms]
        for i in range(cross_corr.shape[0]):
            for j in range(cross_corr.shape[1]):
                if abs(cross_corr[i][j]) > 0.1:  # Decently strong correlation
                    logger.info(
                        "Representation var %d correlates with Fourier term %d with correlation %d",
                        i,
                        self.fourier_coefs[terms[j]][0],
                        cross_corr[i][j],
                    )

    def test_ood(self, model: Module):
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
                    y_hat, _ = model(x, return_layer=False)

                y = ((-1) * y + 1) / 2
                eval_acc += ((y_hat > 0.5) == y.bool()).float().mean().item()

            logger.info(
                f"Average eval accuracy with delta={delta}: {eval_acc / len(data_loader)}"
            )
