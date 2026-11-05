multimodal_eval_toolkit 🧪
Khung đánh giá đa tiêu chí cho mô hình hợp nhất y sinh dựa trên Fusion Gain (FG), Inter-Modality Reliability (IMR), và Uncertainty Quantification (UQ).

Đề tài Khóa luận Tốt nghiệp CNTT - Trường Đại học Kinh tế - Kỹ thuật Bình Dương.

📌 Tính năng chính
Fusion Gain (FG): Định lượng mức tăng trưởng hiệu năng so với mô hình đơn thức xuất sắc nhất.

Inter-Modality Reliability (IMR): Đo đạc độ bền vững và mức độ phụ thuộc dữ liệu khi khuyết thiếu phương thức.

Uncertainty Quantification (UQ): Sử dụng Monte Carlo Dropout trích xuất Predictive Entropy đo độ tự tin dự đoán.

Phân loại tự động: Tự động gán nhãn 04 trạng thái hợp nhất (Modality Redundancy, Modal Conflict, Low Gain, Optimal Fusion).

🛠️ Hướng dẫn cài đặt
# Clone dự án về máy
git clone https://github.com/mainguyentruongduy60539-star/multimodal_eval_toolkit.git

# Di chuyển vào thư mục dự án
cd multimodal_eval_toolkit

# Cài đặt gói thư viện
pip install -e .
🚀 Cách chạy Benchmark tự động
python run_evaluation.py --dataset PH2 --mc_samples 20
