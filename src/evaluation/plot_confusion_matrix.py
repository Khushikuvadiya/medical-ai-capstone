import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = np.array([
    [140, 94],
    [13, 377]
])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

display.plot()

plt.title("Baseline ResNet18 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix_baseline.png",
    dpi=300
)

plt.show()

print("Saved: outputs/confusion_matrix_baseline.png")