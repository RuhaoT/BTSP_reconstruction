"""Calculate the sparsity distribution of a batch of binary patterns."""
import numpy as np
import matplotlib.pyplot as plt
import s3dataset


def calculate_sparsity_distribution(batch: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Calculate the sparsity distribution of a batch of binary patterns.

    Args:
        batch (np.ndarray): The batch of binary patterns.
        axis (int): The axis along which to calculate the sparsity distribution.
            For example, if the each column of the batch represents a pattern,
            set axis=1 to calculate the sparsity of each pattern.

    Returns:
        np.ndarray: The sparsity distribution.
    """
    pattern_sparsity = np.sum(batch, axis=axis) / batch.shape[axis]
    return pattern_sparsity

def plot_masked_sparsity_distribution(dataset: s3dataset.MinimalBTSPDataset, mask_type: str, title:str):
    """
    Plot the sparsity distribution of a dataset with a mask applied.

    Args:
        dataset (s3dataset.MinimalBTSPDataset): The dataset to plot.
        mask_type (str): The type of mask to apply.
        title (str): The title of the plot.
    """
    # Apply the mask
    dataset.set_mask(mask_type, 0.5)

    # Calculate the sparsity distribution
    pattern_sparsity = calculate_sparsity_distribution(dataset.to_binary_tensors(True), axis=1)

    # Plot the sparsity distribution
    fig = plt.figure()
    plt.hist(pattern_sparsity, bins=50)
    plt.xlabel("Proportion of Active Neurons")
    plt.ylabel("Frequency(Pattern Number)")
    plt.title(title)

    # Add numbers to each histogram category
    counts, bins, patches = plt.hist(pattern_sparsity, bins=50)
    for count, patch in zip(counts, patches):
        plt.text(patch.get_x() + patch.get_width() / 2, count, int(count), 
                 ha='center', va='bottom')

    plt.savefig(f"{mask_type}_sparsity_distribution.png")
    plt.close(fig)

if __name__ == "__main__":
    # Load example binary patterns
    example_dataset = s3dataset.MinimalBTSPDataset(32,8)
    example_dataset.from_file("./dataset.csv")
    batch = example_dataset.to_binary_tensors()

    # plot the sparsity distribution of the dataset with different masks
    MASK_TYPES = ["full", "colors", "coordinates", "none"]
    for mask_type in MASK_TYPES:
        if mask_type == "none":
            title = "Sparsity Distribution of Original Dataset"
        else:
            title = f"Sparsity Distribution of Dataset with {mask_type.capitalize()} Mask"
        plot_masked_sparsity_distribution(example_dataset, mask_type, title)
