import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def plot_metrics(history):
    """Affiche les courbes d’accuracy et de loss."""
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.show()


def evaluate_model(model, dataset, class_names, title='Test'):
    """Évalue le modèle et affiche rapport + matrice de confusion."""
    y_true, y_pred = [], []
    for imgs, labs in dataset:
        preds = (model.predict(imgs) > 0.5).astype(int).flatten()
        y_true.extend(labs.numpy())
        y_pred.extend(preds)

    print(f"--- {title} Set Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    acc = accuracy_score(y_true, y_pred)
    print(f"{title} accuracy: {acc:.2f}")

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
