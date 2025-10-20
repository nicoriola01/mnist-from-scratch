import matplotlib.pyplot as plt
from src.data_loader import data_test, labels_test
from src.neural_network import get_errors

errors = get_errors(data_test, labels_test)
print(f"\nFound {len(errors)} errors out of 10.000 test images.")

num_to_show = min(len(errors), 25)  # Show up to 25 errors
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("Misclassified Digits", fontsize=16)

for i, error in enumerate(errors[:num_to_show]):
    ax = axes[i // 5, i % 5]

    image = data_test[:, error['index']].reshape(28, 28)

    ax.imshow(image, cmap='gray')
    ax.set_title(f"Pred: {error['predicted']}, Actual: {error['actual']}")
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()