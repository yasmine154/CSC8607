import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dataset import CardioDataset


# =============================================================
# 1. Configuration générale
# =============================================================

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device utilisé : {device}")


# =============================================================
# 2. Chargement du dataset
# =============================================================

dataset = CardioDataset(
    "data/cardio_train.csv"
)

generator = torch.Generator().manual_seed(SEED)

train_set, val_set, test_set = random_split(
    dataset,
    [0.8, 0.1, 0.1],
    generator=generator
)


# =============================================================
# 3. DataLoaders
# =============================================================

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)


# Récupération d'un batch pour connaître automatiquement
# le nombre de features.
example_batch = next(iter(train_loader))

input_size = example_batch["features"].shape[1]

print(f"Nombre de features : {input_size}")


# =============================================================
# 4. Architecture du MLP
# =============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=128
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.net(x)


# =============================================================
# 5. Fonction de calcul de l'accuracy
# =============================================================

def compute_accuracy(outputs, targets):
    """
    Transforme les probabilités en classes avec
    un seuil de 0.5 et calcule l'accuracy.
    """

    predictions = (
        outputs >= 0.5
    ).float()

    correct = (
        predictions == targets
    ).float().sum()

    accuracy = correct / targets.numel()

    return accuracy.item()


# =============================================================
# 6. Expérience L1 / L2
# =============================================================

def train_with_regularization(
    l1_lambda=1e-4,
    l2_lambda=1e-3,
    epochs=10
):

    print("\n======================================")
    print("EXPÉRIENCE L1 / L2")
    print("======================================")

    print(f"L1 lambda : {l1_lambda}")
    print(f"L2 lambda : {l2_lambda}")

    model = MLP(
        input_size=input_size,
        hidden_size=128
    ).to(device)

    criterion = nn.BCELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        running_accuracy = 0.0

        for batch in train_loader:

            inputs = (
                batch["features"]
                .to(device)
            )

            targets = (
                batch["labels"]
                .to(device)
            )

            # ---------------------------------------------
            # Étape 1 : remettre les gradients à zéro
            # ---------------------------------------------

            optimizer.zero_grad()

            # ---------------------------------------------
            # Étape 2 : forward pass
            # ---------------------------------------------

            outputs = model(inputs)

            # ---------------------------------------------
            # Étape 3 : loss principale
            # ---------------------------------------------

            base_loss = criterion(
                outputs,
                targets
            )

            # ---------------------------------------------
            # Étape 4 : pénalité L1
            # somme des valeurs absolues des paramètres
            # ---------------------------------------------

            l1_penalty = sum(
                p.abs().sum()
                for p in model.parameters()
            )

            # ---------------------------------------------
            # Étape 5 : pénalité L2
            # somme des carrés des paramètres
            # ---------------------------------------------

            l2_penalty = sum(
                (p ** 2).sum()
                for p in model.parameters()
            )

            # ---------------------------------------------
            # Étape 6 : loss totale
            # ---------------------------------------------

            loss = (
                base_loss
                + l1_lambda * l1_penalty
                + l2_lambda * l2_penalty
            )

            # ---------------------------------------------
            # Étape 7 : backpropagation
            # ---------------------------------------------

            loss.backward()

            # ---------------------------------------------
            # Étape 8 : mise à jour des poids
            # ---------------------------------------------

            optimizer.step()

            running_loss += loss.item()

            running_accuracy += compute_accuracy(
                outputs,
                targets
            )

        mean_loss = (
            running_loss
            / len(train_loader)
        )

        mean_accuracy = (
            running_accuracy
            / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss = {mean_loss:.4f} "
            f"| Accuracy = {mean_accuracy:.4f}"
        )

    return model


# =============================================================
# 7. Évaluation sur le validation set
# =============================================================

def validation_loss(model):
    """
    Calcule la BCE moyenne sur le validation set.
    """

    model.eval()

    criterion = nn.BCELoss()

    total_loss = 0.0

    with torch.no_grad():

        for batch in val_loader:

            inputs = (
                batch["features"]
                .to(device)
            )

            targets = (
                batch["labels"]
                .to(device)
            )

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            total_loss += loss.item()

    return total_loss / len(val_loader)


# =============================================================
# 8. Entraînement pour comparaison des optimiseurs
# =============================================================

def train_model(
    opt_name,
    learning_rate=0.001,
    epochs=30
):

    print("\n======================================")
    print(f"OPTIMISEUR : {opt_name}")
    print("======================================")

    model = MLP(
        input_size=input_size,
        hidden_size=128
    ).to(device)

    criterion = nn.BCELoss()

    # ---------------------------------------------------------
    # Choix de l'optimiseur
    # ---------------------------------------------------------

    if opt_name == "SGD":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Momentum":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    elif opt_name == "RMSprop":

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:

        raise ValueError(
            f"Optimiseur inconnu : {opt_name}"
        )

    # ---------------------------------------------------------
    # TensorBoard
    # ---------------------------------------------------------

    writer = SummaryWriter(
        f"runs/cardio_{opt_name}_lr{learning_rate}"
    )

    # ---------------------------------------------------------
    # Entraînement
    # ---------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        running_accuracy = 0.0

        for batch in train_loader:

            inputs = (
                batch["features"]
                .to(device)
            )

            targets = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            running_accuracy += compute_accuracy(
                outputs,
                targets
            )

        mean_loss = (
            running_loss
            / len(train_loader)
        )

        mean_accuracy = (
            running_accuracy
            / len(train_loader)
        )

        # Enregistrement TensorBoard
        writer.add_scalar(
            "Training Loss",
            mean_loss,
            epoch
        )

        writer.add_scalar(
            "Training Accuracy",
            mean_accuracy,
            epoch
        )

        print(
            f"{opt_name:8s} "
            f"| Epoch {epoch + 1:02d}/{epochs} "
            f"| Loss = {mean_loss:.4f} "
            f"| Accuracy = {mean_accuracy:.4f}"
        )

    writer.close()

    val_loss = validation_loss(model)

    print(
        f"{opt_name} - Validation loss : "
        f"{val_loss:.4f}"
    )

    return model, val_loss


# =============================================================
# 9. Évaluation finale
# =============================================================

def evaluate_model(
    model,
    test_loader
):

    model.eval()

    all_targets = []
    all_preds_probs = []

    # Pas de calcul des gradients pendant le test
    with torch.no_grad():

        for batch in test_loader:

            inputs = (
                batch["features"]
                .to(device)
            )

            targets = (
                batch["labels"]
                .to(device)
            )

            outputs = model(inputs)

            all_targets.extend(
                targets.cpu().numpy()
            )

            all_preds_probs.extend(
                outputs.cpu().numpy()
            )

    # ---------------------------------------------------------
    # Conversion en tableaux numpy
    # ---------------------------------------------------------

    all_targets = np.array(
        all_targets
    ).ravel()

    all_preds_probs = np.array(
        all_preds_probs
    ).ravel()

    # ---------------------------------------------------------
    # Classes binaires avec seuil 0.5
    # ---------------------------------------------------------

    all_preds_classes = (
        all_preds_probs > 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # Métriques
    # ---------------------------------------------------------

    precision = precision_score(
        all_targets,
        all_preds_classes
    )

    recall = recall_score(
        all_targets,
        all_preds_classes
    )

    f1 = f1_score(
        all_targets,
        all_preds_classes
    )

    auc = roc_auc_score(
        all_targets,
        all_preds_probs
    )

    print("\n======================================")
    print("RÉSULTATS SUR LE TEST SET")
    print("======================================")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# =============================================================
# 10. Programme principal
# =============================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # PARTIE A
    # Régularisation normale
    # ---------------------------------------------------------

    print("\n\n######################################")
    print("# PARTIE A : L1 / L2 normales")
    print("######################################")

    train_with_regularization(
        l1_lambda=1e-4,
        l2_lambda=1e-3,
        epochs=10
    )

    # ---------------------------------------------------------
    # PARTIE B
    # Régularisation L1 volontairement trop forte
    #
    # C'est l'expérience explicitement demandée dans le TP :
    #
    # l1_lambda = 0.1
    # l2_lambda = 0
    # ---------------------------------------------------------

    print("\n\n######################################")
    print("# PARTIE B : L1 très forte")
    print("######################################")

    train_with_regularization(
        l1_lambda=0.1,
        l2_lambda=0.0,
        epochs=10
    )

    # ---------------------------------------------------------
    # PARTIE C
    # Comparaison des optimiseurs
    # ---------------------------------------------------------

    print("\n\n######################################")
    print("# PARTIE C : Optimiseurs")
    print("######################################")

    optimizers = [
        "SGD",
        "Momentum",
        "RMSprop",
        "Adam"
    ]

    trained_models = {}
    validation_losses = {}

    for opt_name in optimizers:

        model, val_loss = train_model(
            opt_name,
            learning_rate=0.001,
            epochs=30
        )

        trained_models[opt_name] = model

        validation_losses[opt_name] = val_loss

    # ---------------------------------------------------------
    # Choix du meilleur modèle selon validation loss
    # ---------------------------------------------------------

    best_optimizer = min(
        validation_losses,
        key=validation_losses.get
    )

    best_model = trained_models[
        best_optimizer
    ]

    print("\n======================================")
    print("VALIDATION LOSSES")
    print("======================================")

    for opt_name, loss in validation_losses.items():

        print(
            f"{opt_name:8s} : {loss:.4f}"
        )

    print(
        f"\nMeilleur modèle selon la "
        f"validation loss : {best_optimizer}"
    )

    # ---------------------------------------------------------
    # PARTIE D
    # Évaluation finale sur le test
    # ---------------------------------------------------------

    metrics = evaluate_model(
        best_model,
        test_loader
    )
