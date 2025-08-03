import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np
from realesrgan.utils import RealESRGANer
from gfpgan.utils import GFPGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact

# Khởi tạo RealESRGAN và GFPGAN
realesrgan_model_path = r'weights\realesr-general-x4v3.pth'
gfpgan_model_path = r'weights\GFPGANv1.4.pth'

# Load model RealESRGAN
sr_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
half = True if torch.cuda.is_available() else False
realesrganer = RealESRGANer(scale=4, model_path=realesrgan_model_path, model=sr_model, tile=0, tile_pad=10, pre_pad=0, half=half)

# Load model GFPGAN
face_enhancer = GFPGANer(model_path=gfpgan_model_path, upscale=10, arch='clean', channel_multiplier=2, bg_upsampler=realesrganer)

# Hàm áp dụng các bộ lọc (Canny, Gaussian, Bilateral)
def apply_filters(img):
    # Canny Edge Detection
    canny_edges = cv2.Canny(img, 80, 180)
    canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)  # Chuyển đổi từ ảnh grayscale sang RGB để dễ hiển thị
    
    # Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(img, (1, 1), 0)

    # Bilateral Filter
    bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)

    # Combine all filters (this is just one way to combine them, you can customize it)
    combined_result = cv2.addWeighted(bilateral_filter, 0.7, gaussian_blur, 0.6, 0)
    combined_result = cv2.addWeighted(combined_result, 0.8, canny_edges, 0.01, 0)

    return combined_result  # Chỉ hiển thị Bilateral Filter (bạn có thể thay đổi tùy chọn)

# Hàm xóa nhiễu và nâng cấp ảnh
def upscale_and_enhance_image(image_path, progress_callback=None):
    try:
        # Đọc ảnh và chuyển sang định dạng numpy
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Không thể đọc được ảnh từ tệp.")
        
        # Áp dụng các bộ lọc và chỉ giữ lại Bilateral Filter
        final_img = apply_filters(img)
        
        # Xóa nhiễu và nâng cấp ảnh
        output, _ = realesrganer.enhance(final_img, outscale=4)
        if progress_callback:
            progress_callback(50)  # Cập nhật tiến trình sau khi RealESRGAN xử lý ảnh
        
        # Tăng cường chất lượng khuôn mặt
        # Áp dụng GFPGAN trước khi RealESRGAN xử lý để làm rõ chi tiết khuôn mặt
        _, _, img_enhanced = face_enhancer.enhance(output, has_aligned=False, only_center_face=False, paste_back=True)
        
        if progress_callback:
            progress_callback(100)  # Hoàn thành tiến trình
        
        return img_enhanced
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))
        return None


# Hàm hiển thị ảnh trong Tkinter
def display_image(img, label):
    # Chuyển ảnh OpenCV BGR sang RGB và sử dụng Pillow để làm việc với Tkinter
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Tính toán tỷ lệ để thay đổi kích thước ảnh
    max_width = 400  # Kích thước tối đa cho chiều rộng
    max_height = 400  # Kích thước tối đa cho chiều cao
    
    # Lấy kích thước ảnh ban đầu
    img_width, img_height = img.size
    
    # Tính tỷ lệ thu nhỏ cho ảnh sao cho phù hợp với kích thước
    scale_width = max_width / img_width
    scale_height = max_height / img_height
    scale = min(scale_width, scale_height)  # Lấy tỷ lệ nhỏ nhất để giữ tỷ lệ ảnh
    
    # Tính kích thước mới sau khi thay đổi tỷ lệ
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    # Thay đổi kích thước ảnh mà không làm thay đổi tỷ lệ
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Chuyển ảnh thành định dạng Tkinter PhotoImage
    img_tk = ImageTk.PhotoImage(img_resized)
    
    # Cập nhật nhãn với ảnh mới
    label.config(image=img_tk)
    label.image = img_tk  # Giữ tham chiếu tới ảnh để tránh bị xóa

# Hàm tải ảnh và xử lý
def upload_and_display_image():
    file_path = filedialog.askopenfilename(title="Select photo", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if file_path:
        # Lưu ảnh đã tải lên để hiển thị
        global uploaded_image_path
        uploaded_image_path = file_path
        
        # Đọc ảnh bằng Pillow (PIL) để tránh lỗi kích thước ảnh
        try:
            pil_image = Image.open(uploaded_image_path)
            pil_image.thumbnail((1000, 1000))  # Giảm kích thước ảnh nếu quá lớn
            img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # Chuyển đổi từ PIL sang OpenCV
            display_image(img, uploaded_img_label)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {e}")

# Hàm nâng cấp ảnh khi nhấn nút
def upgrade_image():
    if not uploaded_image_path:
        messagebox.showerror("Lỗi", "Chưa có ảnh nào được tải lên!")
        return
    
    # Hiển thị phần trăm tiến trình
    progress_label.config(text="0%")  # Bắt đầu hiển thị 0%
    
    def update_progress(progress):
        progress_label.config(text=f"{progress}%")  # Cập nhật phần trăm
        window.update_idletasks()  # Cập nhật giao diện sau mỗi lần thay đổi tiến trình
    
    # Nâng cấp ảnh và áp dụng các bộ lọc
    processed_img = upscale_and_enhance_image(uploaded_image_path, progress_callback=update_progress)
    if processed_img is not None:
        display_image(processed_img, upgraded_img_label)
        global upgraded_image  # Lưu ảnh đã nâng cấp
        upgraded_image = processed_img
        
        # Kích hoạt nút "Tải ảnh xuống"
        save_button.config(state="normal")

# Hàm tải ảnh xuống
def save_image():
    if upgraded_image is None:
        messagebox.showerror("Lỗi", "Chưa có ảnh nào được nâng cấp!")
        return
    
    # Cửa sổ chọn nơi lưu ảnh
    save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if save_path:
        try:
            cv2.imwrite(save_path, upgraded_image)  # Lưu ảnh đã nâng cấp
            messagebox.showinfo("Thành công", f"Ảnh đã được lưu tại {save_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

# Giao diện chính Tkinter
window = tk.Tk()
window.title("Photo Enhancement")

# Chia cửa sổ thành hai cột để hiển thị ảnh
frame = tk.Frame(window)
frame.grid(row=0, column=0, padx=120, pady=30)

# Cột trái: Hiển thị ảnh đã tải lên
uploaded_img_label = tk.Label(frame)
uploaded_img_label.grid(row=0, column=0, padx=10, pady=10)

# Cột phải: Hiển thị ảnh đã nâng cấp
upgraded_img_label = tk.Label(frame)
upgraded_img_label.grid(row=0, column=1, padx=10, pady=10)

# Nút tải ảnh
upload_button = tk.Button(window, text="Upload photo", command=upload_and_display_image)
upload_button.grid(row=1, column=0, pady=10)

# Nút nâng cấp ảnh
upgrade_button = tk.Button(window, text="Upgrading photos", command=upgrade_image)
upgrade_button.grid(row=2, column=0, pady=10)

# Nút tải ảnh xuống (bắt đầu ở trạng thái "disabled")
save_button = tk.Button(window, text="Download photos", command=save_image, state="disabled")
save_button.grid(row=3, column=0, pady=10)

# Nhãn hiển thị phần trăm tiến trình
progress_label = tk.Label(window, text="0%", font=("Arial", 12))
progress_label.grid(row=4, column=0, columnspan=2, pady=5)

# Biến toàn cục để lưu đường dẫn ảnh đã tải lên và ảnh đã nâng cấp
uploaded_image_path = None
upgraded_image = None

# Cửa sổ chính
window.mainloop()
