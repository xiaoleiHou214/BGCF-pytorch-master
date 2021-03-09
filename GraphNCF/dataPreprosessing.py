from torch.utils.data import Dataset

# movielens 1k

class Wsdream(Dataset):

    def __init__(self,rt):
        super(Dataset,self).__init__()
        self.uId = list(rt['[User ID]'])
        self.iId = list(rt['[Service ID]'])
        self.rt = list(rt['[RT]'])
        self.uCountry = list(rt['[User Country]'])
        self.uAs = list(rt['[User AS]'])
        self.iCountry = list(rt['[Service Country]'])
        self.iAs = list(rt['[Service AS]'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item],self.uCountry[item],self.uAs[item],self.iCountry[item],self.iAs[item])