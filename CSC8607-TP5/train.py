
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from dataset import train_dataset, test_dataset
from text_utils import (
    vocabulary,
    create_embedding_layer,
    tokenize,
    untokenize
)
from model import ImageCaptioningModel, resnet_pretrained


def prepare_captions(captions, word2idx, device):
    """
    Flickr30k fournit 5 légendes par image.
    captions[0] contient la première légende de
    chaque image du batch.
    """
    sentences = captions[0]

    tokens = [
        tokenize(sentence, word2idx, max_length=20)
        for sentence in sentences
    ]

    return torch.tensor(
        tokens, dtype=torch.long, device=device
    )


@torch.no_grad()
def generate_caption(model, image, idx2word, max_words=19):
    """
    Génération réellement auto-régressive :
    chaque mot prédit devient l'entrée suivante.
    """
    model.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0)

    features = model.cnn(image)
    device = image.device

    # Indice initial du TP
    token = torch.zeros(
        1, 1, dtype=torch.long, device=device
    )

    states = None
    generated_words = []

    for _ in range(max_words):
        emb = model.embeddings(token)

        hidden_seq, states = model.lstm(
            emb,
            features,
            init_states=states
        )

        logits = model.fc_out(hidden_seq[:, -1, :])
        next_idx = int(logits.argmax(dim=-1).item())

        # Il n'existe pas de véritable EOS dans
        # cette version simplifiée.
        if next_idx == 0:
            break

        generated_words.append(idx2word[next_idx])

        token = torch.tensor(
            [[next_idx]],
            dtype=torch.long,
            device=device
        )

    return " ".join(generated_words)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/full")
    parser.add_argument("--log-dir", default="runs/captioning")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device :", device, flush=True)
    train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=4,
      pin_memory=(device.type == "cuda")
)

    test_loader = DataLoader(
      test_dataset,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=4,
      pin_memory=(device.type == "cuda")
)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Embeddings Word2Vec
    print("Chargement des embeddings...", flush=True)

    embedding_model, idx2word, word2idx = (
        create_embedding_layer(
            vocabulary,
            randomized=args.randomized
        )
    )

    vocab_size = len(idx2word)

    print("Vocab size :", vocab_size, flush=True)
    print(
        "Embedding shape :",
        embedding_model.weight.shape,
        flush=True
    )

    # 2. Modèle global
    model = ImageCaptioningModel(
        cnn_model=resnet_pretrained,
        embedding_model=embedding_model,
        lstm_hidden_dim=args.hidden_size
    ).to(device)

    # 3. Optimiseur, loss, scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    writer = SummaryWriter(args.log_dir)

    # 4. Checkpoints
    last_ckpt = output_dir / "last_checkpoint.pth"
    best_ckpt = output_dir / "best_model.pth"

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and last_ckpt.exists():
        checkpoint = torch.load(
            last_ckpt,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]

        print(
            f"Reprise à l'époque {start_epoch + 1}",
            flush=True
        )

    predictions_file = output_dir / "predictions.txt"
    metrics_file = output_dir / "metrics.csv"

    if start_epoch == 0:
        with open(metrics_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss", "lr"
            ])
        predictions_file.write_text("", encoding="utf-8")

    global_step = start_epoch * len(train_loader)

    # 5. Boucle principale
    for epoch in range(start_epoch, args.epochs):
        model.train()

        # Important : le CNN est gelé.
        # On conserve aussi BatchNorm en mode évaluation.
        model.cnn.eval()

        total_train_loss = 0.0
        train_steps = 0
        recent_losses = []

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n===== Epoch {epoch + 1}/{args.epochs} "
            f"| LR={current_lr:.6f} =====",
            flush=True
        )

        # ---- ENTRAINEMENT ----
        for i, (images, captions) in enumerate(train_loader):
            if (
                args.max_train_batches > 0
                and i >= args.max_train_batches
            ):
                break

            images = images.to(device)
            caps = prepare_captions(
                captions, word2idx, device
            )

            optimizer.zero_grad(set_to_none=True)

            # Teacher forcing
            outputs = model(images, caps[:, :-1])

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caps[:, 1:].reshape(-1)
            )

            loss.backward()

            # Sécurité contre les gradients très grands
            torch.nn.utils.clip_grad_norm_(
                filter(
                    lambda p: p.requires_grad,
                    model.parameters()
                ),
                max_norm=1.0
            )

            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1
            recent_losses.append(loss.item())
            global_step += 1

            if (i + 1) % 10 == 0:
                avg_recent = sum(recent_losses) / len(recent_losses)

                writer.add_scalar(
                    "Loss/Train_Step",
                    avg_recent,
                    global_step
                )

                print(
                    f"Batch {i+1}/{len(train_loader)} "
                    f"| Loss={avg_recent:.4f}",
                    flush=True
                )

                recent_losses = []

        avg_train_loss = total_train_loss / max(train_steps, 1)

        # ---- VALIDATION ----
        model.eval()

        total_val_loss = 0.0
        val_steps = 0

        example_image = None
        example_target = ""
        example_teacher = ""

        with torch.no_grad():
            for i, (images, captions) in enumerate(test_loader):
                if (
                    args.max_val_batches > 0
                    and i >= args.max_val_batches
                ):
                    break

                images = images.to(device)
                caps = prepare_captions(
                    captions, word2idx, device
                )

                outputs = model(images, caps[:, :-1])

                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    caps[:, 1:].reshape(-1)
                )

                total_val_loss += loss.item()
                val_steps += 1

                # Garder un exemple pour observer les progrès
                if example_image is None:
                    example_image = images[:1]

                    example_target = captions[0][0]

                    predicted_indices = (
                        outputs[0].argmax(dim=-1).cpu().tolist()
                    )

                    predicted_words = untokenize(
                        predicted_indices, idx2word
                    )

                    example_teacher = " ".join(
                        word for word in predicted_words if word
                    )

        avg_val_loss = total_val_loss / max(val_steps, 1)

        # Véritable génération à partir du seul token initial
        generated = generate_caption(
            model,
            example_image,
            idx2word
        )

        print(
            f"Train Loss : {avg_train_loss:.4f}",
            flush=True
        )
        print(
            f"Val Loss   : {avg_val_loss:.4f}",
            flush=True
        )
        print("Target     :", example_target, flush=True)
        print(
            "Prediction (teacher forcing) :",
            example_teacher,
            flush=True
        )
        print(
            "Generation (autoregressive)  :",
            generated,
            flush=True
        )

        # 6. TensorBoard
        writer.add_scalar(
            "Loss/Train", avg_train_loss, epoch + 1
        )
        writer.add_scalar(
            "Loss/Validation", avg_val_loss, epoch + 1
        )
        writer.add_scalar(
            "LR", current_lr, epoch + 1
        )

        # 7. Enregistrer les résultats textuels
        with open(predictions_file, "a", encoding="utf-8") as f:
            f.write(f"\n===== Epoch {epoch+1} =====\n")
            f.write(f"Target : {example_target}\n")
            f.write(f"Teacher forcing : {example_teacher}\n")
            f.write(f"Autoregressive : {generated}\n")

        with open(metrics_file, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                current_lr
            ])

        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                best_ckpt
            )

            print(
                "Nouveau meilleur modèle sauvegardé.",
                flush=True
            )

        # StepLR à la fin de chaque époque
        scheduler.step()

        # Checkpoint pour reprendre après interruption
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            },
            last_ckpt
        )

    writer.close()

    print("\nEntraînement terminé.", flush=True)
    print("Meilleure Val Loss :", best_val_loss, flush=True)


if __name__ == "__main__":
    main()
