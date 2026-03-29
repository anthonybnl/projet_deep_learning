import medmnist


def load_test_dataset():
    dataset = medmnist.ChestMNIST(split="test", download=True, size=64)

    return [(img, label) for img, label in dataset]


def get_label_names():
    return list(medmnist.INFO["chestmnist"]["label"].values())
