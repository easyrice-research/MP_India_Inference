from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, x_data, transform=None):
        self.x_data = x_data  # NumPy array of images embeddings
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        embeddings = self.x_data[idx]        

        return embeddings