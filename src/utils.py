import torch

def magnitude_loss(x,h):
    assert x.shape[0] == h.shape[0]
    batch_magnitude = calculate_magnitude(x)
    latent_magnitude = calculate_magnitude(h)
    return torch.abs(batch_magnitude-latent_magnitude)

def decorated_magnitude_loss(x,h):
    assert x.shape[0] == h.shape[0]
    batch_magnitude = calculate_magnitude(x)
    latent_magnitude = calculate_magnitude(h)
    return batch_magnitude,latent_magnitude,torch.abs(batch_magnitude-latent_magnitude)

def calculate_magnitude(x):
    similarity_matrix = torch.exp(-torch.cdist(x,x,p=2))
    similarity_matrix_inv = torch.inverse(similarity_matrix)
    magnitude = torch.sum(torch.flatten(similarity_matrix_inv))
    return magnitude

def min_dist(x):
    d = torch.cdist(x,x,p=2)
    # Zero the diagonal entries which may be non-zero due to numerical fluctuations.
    d = d-torch.diag(torch.diag(d))
    min_dist = torch.min(d[d.nonzero(as_tuple=True)])
    return min_dist

def max_dist(x):
    d = torch.cdist(x,x,p=2)
    max_dist = torch.max(d)
    return max_dist

def is_scattered(x):
    card_A = torch.tensor(x.shape[0],dtype=torch.float32,requires_grad=False)
    return (min_dist(x) > torch.log10(card_A-1))
