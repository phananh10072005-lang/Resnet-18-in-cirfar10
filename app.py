import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn.functional as F

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & MODEL
# ==========================================
st.set_page_config(page_title="AI Model Comparison", layout="wide")
st.title("🔬 Phân loại ảnh: ResNet-18")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Máy bay', 'Ô tô', 'Chim', 'Mèo', 'Hươu', 'Chó', 'Ếch', 'Ngựa', 'Tàu thủy', 'Xe tải']

# Hàm tạo khung mô hình (Đã sửa lỗi size mismatch cho CIFAR-10)
@st.cache_resource # Lưu vào bộ nhớ đệm để không phải load lại mỗi lần nhấn nút
def load_my_model(arch_name, path):
    if arch_name == 'VGG-11':
        model = models.vgg11()
        model.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(), nn.Linear(512, 10)
        )
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    elif arch_name == 'ResNet-18':
        model = models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif arch_name == 'GoogLeNet':
        model = models.googlenet(num_classes=10, aux_logits=False)
        model.conv1.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

# Tiền xử lý ảnh (Cần khớp 100% với lúc train để tránh lỗi "đường thẳng")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# ==========================================
# 2. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
uploaded_file = st.sidebar.file_uploader("📤 Tải ảnh lên...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.image(img, caption="Ảnh gốc", use_container_width=True)
        btn_predict = st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True)

    if btn_predict:
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Chỉ hiển thị dự đoán từ ResNet-18
        cols = st.columns(1)
        model_configs = [
            ('ResNet-18', 'best_weights_ResNet-18.pth')
        ]

        for idx, (name, path) in enumerate(model_configs):
            with cols[idx]:
                st.markdown(f"### 📌 {name}")
                model = load_my_model(name, path)

                if model:
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = F.softmax(output, dim=1)[0] * 100

                        # Lấy top 1
                        prob_top, pred_top = torch.max(probabilities, 0)

                        # Hiển thị kết quả chính
                        st.metric("Dự đoán chính", class_names[pred_top.item()])
                        st.divider()

                        # Hiển thị danh sách xác suất chi tiết (Sắp xếp giảm dần)
                        st.write("📊 **Chi tiết các lớp:**")
                        sorted_probs, sorted_idx = torch.sort(probabilities, descending=True)

                        for p, i in zip(sorted_probs, sorted_idx):
                            label = class_names[i.item()]
                            val = p.item()
                            # Hiển thị thanh tiến trình nhỏ cho mỗi lớp
                            st.write(f"{label}: {val:.1f}%")
                            st.progress(min(val/100, 1.0))
else:
    st.info("Hãy tải một bức ảnh lên từ thanh bên trái để bắt đầu")