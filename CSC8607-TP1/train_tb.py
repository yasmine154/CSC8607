import os
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 1. HYPERPARAMÈTRES
# ============================================================

hparams = {
    "model": "MLP",
    "batch_size": 128,
    "lr": 1e-1,
    "seed": 0,
    "weight_decay": 0.0
}

EPOCHS = 10


# ============================================================
# 2. REPRODUCTIBILITÉ
# ============================================================

# Utiliser toujours les mêmes graines permet d'obtenir
# des expériences plus facilement comparables.

torch.manual_seed(hparams["seed"])
random.seed(hparams["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(hparams["seed"])


# ============================================================
# 3. DEVICE : GPU OU CPU
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 4. DOSSIER TENSORBOARD
# ============================================================

# Chaque entraînement aura son propre dossier.
# Exemple :
# runs/MLP/bs32_lr0.01_20260920-120000

run_name = (
    f"{hparams['model']}/"
    f"bs{hparams['batch_size']}_"
    f"lr{hparams['lr']}_"
    f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

logdir = os.path.join("runs", run_name)

print("Logdir:", logdir)

writer = SummaryWriter(log_dir=logdir)


# ============================================================
# 5. PRÉPARATION DE CIFAR-10
# ============================================================

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR10_STD = (
    0.2023,
    0.1994,
    0.2010
)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# Ensemble d'entraînement original de CIFAR-10
trainset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# Ensemble de test.
# On le charge mais on ne l'utilise PAS pour choisir
# les hyperparamètres.
testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# 6. SPLIT TRAIN / VALIDATION
# ============================================================

# CIFAR-10 contient 50 000 exemples d'entraînement.
#
# On prend :
# 90 % pour train
# 10 % pour validation

N = len(trainset)

val_size = int(0.1 * N)

train_size = N - val_size


train_subset, val_subset = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(
        hparams["seed"]
    )
)


print("Training samples:", len(train_subset))
print("Validation samples:", len(val_subset))
print("Test samples:", len(testset))


# ============================================================
# 7. NOMBRE DE WORKERS
# ============================================================

# Sur Slurm, on ne veut pas utiliser trop de CPUs.

def get_num_workers(default=1, cap=4):

    try:
        n = int(
            os.getenv(
                "SLURM_CPUS_PER_TASK",
                default
            )
        )

    except Exception:
        n = default

    return max(
        0,
        min(cap, n)
    )


num_workers = get_num_workers()

print("num_workers:", num_workers)


# ============================================================
# 8. DATALOADERS
# ============================================================

trainloader = DataLoader(
    train_subset,
    batch_size=hparams["batch_size"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)


valloader = DataLoader(
    val_subset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


testloader = DataLoader(
    testset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# ============================================================
# 9. MODÈLE MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        # Une image CIFAR-10 :
        #
        # 32 x 32 pixels
        # 3 canaux RGB
        #
        # donc :
        # 32 * 32 * 3 = 3072 entrées

        self.fc1 = nn.Linear(
            32 * 32 * 3,
            128
        )

        # CIFAR-10 contient 10 classes

        self.fc2 = nn.Linear(
            128,
            10
        )


    def forward(self, x):

        # (batch, 3, 32, 32)
        #
        # devient :
        #
        # (batch, 3072)

        x = torch.flatten(
            x,
            1
        )

        # Première couche + ReLU

        x = F.relu(
            self.fc1(x)
        )

        # Couche de sortie
        #
        # PAS de Softmax car CrossEntropyLoss
        # attend directement les logits.

        x = self.fc2(x)

        return x


# ============================================================
# 10. INITIALISATION
# ============================================================

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=hparams["lr"],
    momentum=0.9,
    weight_decay=hparams["weight_decay"]
)


print(model)


# ============================================================
# 11. FONCTION POUR CALCULER LES MÉTRIQUES
# ============================================================

@torch.no_grad()
def epoch_metrics(
    loader,
    model,
    criterion,
    device
):

    # Mode évaluation
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0


    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )


        # Forward

        logits = model(x)


        # Calcul de la loss

        loss = criterion(
            logits,
            y
        )


        # loss.item() est la moyenne du batch.
        # On multiplie donc par la taille du batch
        # pour obtenir la somme.

        loss_sum += (
            loss.item()
            * y.size(0)
        )


        # Classe prédite

        pred = logits.argmax(
            dim=1
        )


        correct += (
            pred == y
        ).sum().item()

        total += y.size(0)


    # Moyenne réelle sur tout le dataset

    loss_avg = (
        loss_sum / total
    )

    acc = (
        correct / total
    )


    return loss_avg, acc


# ============================================================
# 12. ENTRAÎNEMENT
# ============================================================

global_step = 0


for epoch in range(
    1,
    EPOCHS + 1
):

    # Mode entraînement

    model.train()


    running_loss_sum = 0.0
    running_total = 0


    # --------------------------------------------------------
    # Boucle sur les mini-batchs
    # --------------------------------------------------------

    for b, (x, y) in enumerate(
        trainloader
    ):


        # Copier les données vers le GPU

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # 1. Réinitialisation des gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # 2. Forward pass
        # ----------------------------------------------------

        logits = model(x)


        # ----------------------------------------------------
        # 3. Calcul de la loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            y
        )


        # ----------------------------------------------------
        # 4. Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # 5. TensorBoard : loss d'un mini-batch
        # ----------------------------------------------------

        # On ne loggue qu'une fois toutes les
        # 10 itérations pour éviter trop de données.

        if b % 10 == 0:

            writer.add_scalar(
                "Loss/train_step",
                loss.item(),
                global_step
            )


        # ----------------------------------------------------
        # 6. Mise à jour des paramètres
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistiques
        # ----------------------------------------------------

        running_loss_sum += (
            loss.item()
            * y.size(0)
        )

        running_total += (
            y.size(0)
        )


        global_step += 1


    # ========================================================
    # FIN DE L'ÉPOQUE
    # ========================================================


    # Loss moyenne du train

    train_loss = (
        running_loss_sum
        / running_total
    )


    # Calcul sur validation

    val_loss, val_acc = epoch_metrics(
        valloader,
        model,
        criterion,
        device
    )


    # ========================================================
    # LOGGING TENSORBOARD
    # ========================================================

    writer.add_scalar(
        "Loss/train",
        train_loss,
        epoch
    )


    writer.add_scalar(
        "Loss/val",
        val_loss,
        epoch
    )


    writer.add_scalar(
        "Accuracy/val",
        val_acc,
        epoch
    )


    # Affichage terminal

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.3f}"
    )


# ============================================================
# 13. FIN DE L'ENTRAÎNEMENT
# ============================================================

writer.flush()
writer.close()


# ============================================================
# 14. SAUVEGARDE DU MODÈLE
# ============================================================

torch.save(
    model.state_dict(),
    "mlp_model_tb.pth"
)


print(
    "Model saved in mlp_model_tb.pth"
)

print(
    "TensorBoard logs saved in:",
    logdir
)
