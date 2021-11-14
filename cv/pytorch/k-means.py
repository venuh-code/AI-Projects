import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

n_cluster = 4
data = make_blobs(n_samples=1000, n_features=2,centers=n_cluster)
matrix = torch.from_numpy(data[0]).to(device).float()

print(matrix.type())

class KMEANS():
        def __init__(self, n_clusters=n_cluster, max_iter=None, verbose=False, show=True):

            self.n_clusters = n_clusters
            self.labels = None
            self.dists = None
            self.centers = None
            self.variation = torch.Tensor([float("Inf")]).to(device)
            self.verbose = verbose
            self.started = False
            self.max_iter = max_iter
            self.count = 0
            self.show = show

        def fit(self,x):
            self.plus(x)
            while True:
                self.nearest_center(x)
                self.update_center(x)
                if self.verbose:
                    print(self.variation, torch.argmin(self.dists, (0)))
                if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                    break
                elif self.max_iter is not None and self.count == self.max_iter:
                    break
                self.count += 1
                if self.show:
                    self.show_result(x)

        def nearest_center(self, x):
            labels = torch.empty((x.shape[0],)).long().to(device)
            dists = torch.empty((0, self.n_clusters)).to(device)
            for i, sample in enumerate(x):
                dist = torch.sum(torch.mul(sample-self.centers, sample-self.centers), (1))
                labels[i] = torch.argmin(dist)

                dists = torch.cat([dists, dist.unsqueeze(0)], (0))
                #print("dists", dists)

            self.labels = labels
            if self.started:
                self.variation = torch.sum(self.dists-dists)
            self.dists = dists
            self.started = True

        def update_center(self, x):
            centers = torch.empty((0, x.shape[1])).to(device)
            for i in range(self.n_clusters):
                mask = self.labels == i
                cluster_samples = x[mask]
                centers = torch.cat(
                    [centers, torch.mean(cluster_samples, 0).unsqueeze(0)], 0
                )
            self.centers = centers

        def show_result(self, x):
            markers = ["o", "s", "v", "p"]
            self.labels  = self.labels.to('cpu')

            if x.shape[1] != 2 or len(set(self.labels.numpy())) > 4:
                raise Exception("only two demension data")
            print("len", len(set(list(self.labels))))
            for i, label in enumerate(set(list(self.labels.numpy()))):
                samples = x[self.labels == label]
                plt.scatter(
                    [s[0].item() for s in samples],
                    [s[1].item() for s in samples],
                    marker = markers[i],
                )
            self.labels  = self.labels.to('cuda')
            plt.show()

        def plus(self, x):
            num_samples = x.shape[0]
            init_row = torch.randint(0, x.shape[0], (1,)).to(device)
            init_points = x[init_row]
            self.centers = init_points
            for i in range(self.n_clusters - 1):
                distances = []
                for row in x:
                    distances.append(
                        torch.min(torch.norm(row-self.centers, dim=1))
                    )

                distances = torch.Tensor(distances)
                temp = torch.sum((distances)) * torch.rand(1)
                for j in range(num_samples):
                    temp -= distances[j]
                    if temp < 0:
                        self.centers = torch.cat(
                            [self.centers, x[j].unsqueeze(0)], dim=0
                        )
                        break

if __name__ == "__main__":

    import time
    a = time.time()
    clf = KMEANS()
    clf.fit(matrix)
    b = time.time()
    print("total time:{}s, spped:{}iter/s".format(b-a, (b-a)/clf.count))

