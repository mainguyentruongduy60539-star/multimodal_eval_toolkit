from .metrics import compute_fusion_gain, evaluate_imr_accuracy, compute_imr, compute_uncertainty_mc_dropout

class MultimodalEvaluator:
    def __init__(self, model, test_loader, acc_image_only: float, acc_text_only: float, device="cpu"):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.acc_image_only = acc_image_only
        self.acc_text_only = acc_text_only
        self.device = device

    def run_full_evaluation(self, model_name="Multimodal Model"):
        acc_full = evaluate_imr_accuracy(self.model, self.test_loader, device=self.device, drop_modality=None)
        fg = compute_fusion_gain(acc_full, self.acc_image_only, self.acc_text_only)
        
        acc_no_img = evaluate_imr_accuracy(self.model, self.test_loader, device=self.device, drop_modality='image')
        acc_no_txt = evaluate_imr_accuracy(self.model, self.test_loader, device=self.device, drop_modality='text')
        
        imr_no_img = compute_imr(acc_no_img, acc_full)
        imr_no_txt = compute_imr(acc_no_txt, acc_full)
        
        uq_entropy = compute_uncertainty_mc_dropout(self.model, self.test_loader, num_samples=20, device=self.device)
        
        if fg <= 0:
            status = "PHÁT HIỆN HỢP NHẤT KÉM CÓ THẤT BẠI (FG <= 0)"
        elif fg < 0.02:
            status = "Hợp nhất Dư thừa / Tăng trưởng thấp"
        else:
            status = "Hợp nhất Hiệu quả (FG > 0)"

        return {
            "Model Name": model_name,
            "Full Accuracy": f"{acc_full * 100:.2f}%",
            "Fusion Gain (FG)": f"{fg * 100:+.2f}%",
            "IMR (Missing Image)": imr_no_img,
            "IMR (Missing Text)": imr_no_txt,
            "UQ (MC Dropout Entropy)": uq_entropy,
            "Fusion Status": status
        }