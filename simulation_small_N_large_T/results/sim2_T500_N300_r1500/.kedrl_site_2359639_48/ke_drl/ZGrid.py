import torch

class ZGrid:
    @staticmethod
    @torch.no_grad()
    def _kmeans_torch(r: torch.Tensor, n_clusters: int, max_iter: int = 100, tol: float = 1e-4) -> torch.Tensor:
        device, dtype = r.device, r.dtype
        N, d = r.shape
        idx = torch.randperm(N, device=device)[:n_clusters]
        centers = r[idx].clone()
        for _ in range(max_iter):
            dists  = torch.cdist(r, centers)          # (N,K)
            labels = torch.argmin(dists, dim=1)       # (N,)
            new_centers = torch.zeros_like(centers)
            counts = torch.bincount(labels, minlength=n_clusters).clamp_min(1).to(dtype)
            new_centers.index_add_(0, labels, r)
            new_centers = new_centers / counts.unsqueeze(1)
            shift = (new_centers - centers).norm(dim=1).max()
            centers = new_centers
            if shift <= tol: break
        return centers

    @staticmethod
    @torch.no_grad()
    def _boundary_vertices_torch(points_t: torch.Tensor, n_directions: int = 512) -> torch.Tensor:
        """Approximate hull vertices with support directions, entirely on device."""
        device, dtype = points_t.device, points_t.dtype
        _, d = points_t.shape
        eye = torch.eye(d, device=device, dtype=dtype)
        directions = [eye, -eye]
        extra = max(0, int(n_directions) - 2 * d)
        if extra:
            gen = torch.Generator(device=device)
            gen.manual_seed(20260512)
            random_dirs = torch.randn((extra, d), generator=gen, device=device, dtype=dtype)
            random_dirs = random_dirs / random_dirs.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
            directions.append(random_dirs)
        dirs = torch.cat(directions, dim=0)
        vertices = torch.argmax(points_t @ dirs.transpose(0, 1), dim=0)
        return torch.unique(vertices)

    @staticmethod
    @torch.no_grad()
    def Z_kmeans(r: torch.Tensor, n_clusters: int, constant_factor: float) -> torch.Tensor:
        """
        Cluster reward samples and expand hull vertices radially.

        Parameters:
            - r: (N, D) torch tensor with observed (or finite discounted) r.
            - n_clusters: number of k-means clusters (atoms).
            - constant_factor: expansion factor > 0 for hull points.

        Returns:
            - expanded_centers: (n_clusters, D) torch tensor.
        """
        if r.ndim != 2:
            raise ValueError("r must be a 2D tensor of shape (N, D).")
        if constant_factor <= 0:
            raise ValueError("constant_factor must be > 0.")

        device, dtype = r.device, r.dtype

        # 1) K-means (Torch)
        centers = ZGrid._kmeans_torch(r, n_clusters=n_clusters).to(device=device, dtype=dtype)  # (K,D)

        # 2) Global centroid
        mu = centers.mean(dim=0)  # (D,)

        # 3) Boundary-like vertices without leaving Torch/GPU.
        vertices = ZGrid._boundary_vertices_torch(centers)  # (H,)

        # 4) Radial expansion
        expanded = centers.clone()
        expanded[vertices] = mu + constant_factor * (centers[vertices] - mu)
        return expanded


##=========================
##### usage #####
# Z_grids = ZGrid.Z_kmeans(r, n_clusters=num_grid_points, constant_factor=1.8)
