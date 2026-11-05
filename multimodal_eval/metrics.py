import torch
import torch.nn.functional as F
import numpy as np

def compute_fusion_gain(acc_multimodal: float, acc_image_only: float, acc_text_only: float) -> float:
    best_single = max(acc_image_only, acc_text_only)
    fusion_gain = acc_multimodal - best_single
    return round(fusion_gain, 4)

def evaluate_imr_accuracy(model, dataloader, device="cpu", drop_modality=None) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            if drop_modality == 'image':
                images = torch.zeros_like(images)
            elif drop_modality == 'text':
                texts = torch.zeros_like(texts)

            outputs = model(images, texts)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return round(correct / total, 4) if total > 0 else 0.0

def compute_imr(acc_missing: float, acc_full: float) -> float:
    if acc_full == 0:
        return 0.0
    return round(acc_missing / acc_full, 4)

def compute_uncertainty_mc_dropout(model, dataloader, num_samples=20, device="cpu") -> float:
    def enable_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()

    model.eval()
    model.apply(enable_dropout)

    all_entropies = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)

            probs_list = []
            for _ in range(num_samples):
                outputs = model(images, texts)
                probs = F.softmax(outputs, dim=1)
                probs_list.append(probs.unsqueeze(0))

            probs_tensor = torch.cat(probs_list, dim=0)
            avg_probs = torch.mean(probs_tensor, dim=0)

            entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-12), dim=1)
            all_entropies.extend(entropy.cpu().numpy().tolist())

    mean_uncertainty = np.mean(all_entropies)
    return round(float(mean_uncertainty), 4)