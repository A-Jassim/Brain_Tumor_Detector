import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def plot_metrics(history):
    """Plot accuracy and loss curves."""
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.show()


def evaluate_model(model, dataset, class_names, title='Test'):
    """Evaluate model and show report."""
    # Get predictions
    y_true, y_pred = [], []
    for imgs, labs in dataset:
        # Predict (threshold = 0.5)
        # Output > 0.5 means tumor, else no tumor
        preds = (model.predict(imgs) > 0.5).astype(int).flatten()
        y_true.extend(labs.numpy())
        y_pred.extend(preds)

    # Print report
    # Shows precision, recall, F1-score for each class
    print(f"--- {title} Set Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    acc = accuracy_score(y_true, y_pred)
    print(f"{title} accuracy: {acc:.2f}")

    # Show confusion matrix
    # Rows = true labels, Columns = predicted labels
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} Confusion Matrix')
    plt.show()
