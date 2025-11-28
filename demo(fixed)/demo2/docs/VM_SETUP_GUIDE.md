# Hướng Dẫn Setup VM và Test Malware

## ⚠️ CẢNH BÁO BẢO MẬT

**QUAN TRỌNG**: 
- Chỉ test malware trong môi trường cách ly (VM hoặc sandbox)
- KHÔNG BAO GIỜ chạy malware trên hệ thống production
- Tắt kết nối mạng khi test
- Tạo snapshot trước khi test

## Phần 1: Setup Virtual Machine

### 1.1. Chọn Virtualization Software

#### Option 1: VMware Workstation (Khuyến nghị)
- Download: https://www.vmware.com/products/workstation-pro.html
- Free trial 30 ngày
- Performance tốt, dễ sử dụng

#### Option 2: VirtualBox (Miễn phí)
- Download: https://www.virtualbox.org/
- Hoàn toàn miễn phí
- Performance kém hơn VMware một chút

#### Option 3: Hyper-V (Windows Pro/Enterprise)
- Built-in trên Windows
- Enable trong Windows Features

### 1.2. Tạo VM

#### Cấu hình khuyến nghị:
- **OS**: Windows 10/11 hoặc Linux (Ubuntu)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk**: Tối thiểu 50GB
- **Network**: NAT hoặc Internal (KHÔNG dùng Bridged)
- **Snapshot**: Tạo snapshot trước khi test

#### Steps:

1. **Tạo VM mới**
   ```
   - New Virtual Machine
   - Chọn OS (Windows 10/11)
   - Allocate RAM (4-8GB)
   - Create virtual disk (50GB+)
   ```

2. **Cấu hình Network**
   ```
   - Settings > Network Adapter
   - Chọn NAT hoặc Internal
   - KHÔNG dùng Bridged (tránh lây nhiễm)
   ```

3. **Cấu hình Isolation**
   ```
   - Settings > Options > Isolation
   - Enable: Copy/Paste, Drag and Drop (tùy chọn)
   - Disable: Shared Folders (nếu không cần)
   ```

4. **Tạo Snapshot**
   ```
   - Snapshot > Take Snapshot
   - Tên: "Clean State"
   - Mô tả: "Trước khi test malware"
   ```

### 1.3. Cài Đặt OS và Tools

#### Trong VM:

1. **Cài đặt OS** (Windows 10/11 hoặc Linux)

2. **Cài đặt Python**
   ```bash
   # Download Python 3.8+ từ python.org
   # Hoặc dùng package manager
   ```

3. **Clone/Copy project vào VM**
   ```bash
   # Option 1: Copy folder qua shared folder (nếu enable)
   # Option 2: Download từ git (nếu có)
   # Option 3: Copy qua USB (an toàn hơn)
   ```

4. **Cài đặt dependencies**
   ```bash
   cd /path/to/demo2
   pip install -r requirements.txt
   ```

5. **Tắt Windows Defender** (nếu cần, chỉ trong VM)
   ```
   Settings > Update & Security > Windows Security
   > Virus & threat protection > Manage settings
   > Tắt Real-time protection (tạm thời)
   ```

## Phần 2: Chuẩn Bị Test Environment

### 2.1. Tạo Cấu Trúc Thư Mục

```bash
python scripts/create_sample_structure.py
```

### 2.2. Train Models Trước (Nếu chưa có)

```bash
# 1. Chuẩn bị benign samples
# Đặt các file .exe hợp pháp vào data/benign/

# 2. Tạo obfuscated samples (nếu có)
# Đặt vào data/obfuscated/

# 3. Tạo dataset
python main.py generate-dataset

# 4. Train models
python main.py train
```

### 2.3. Tắt Network (Quan trọng!)

**Trong VM Settings:**
```
Settings > Network Adapter
> Disable Network Adapter
```

Hoặc **trong OS:**
```bash
# Windows
netsh interface set interface "Ethernet" disabled

# Linux
sudo ifconfig eth0 down
```

## Phần 3: Test Malware

### 3.1. Copy Malware vào VM

**Cách an toàn:**
1. Tắt network trong VM
2. Copy file qua shared folder (nếu enable) hoặc USB
3. Đặt vào thư mục test (ví dụ: `C:\malware_test\`)

### 3.2. Test với Script

```bash
# Test một file
python scripts/test_malware.py C:\malware_test\sample.exe \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --output results/malware_test.json

# Test với XGBoost
python scripts/test_malware.py C:\malware_test\sample.exe \
    --model models/xgboost_model.json \
    --model-type xgboost \
    --output results/malware_test_xgb.json

# Test với Neural Network
python scripts/test_malware.py C:\malware_test\sample.exe \
    --model models/neural_network_model.pt \
    --model-type neural_network \
    --output results/malware_test_nn.json
```

### 3.3. Test với Dashboard

```bash
# 1. Chạy dashboard
python main.py dashboard

# 2. Mở browser trong VM: http://localhost:5000

# 3. Upload file malware qua web interface

# 4. Xem kết quả
```

### 3.4. Batch Test (Nhiều files)

Tạo script batch test:

```python
# scripts/batch_test.py
import os
from pathlib import Path
from scripts.test_malware import extract_features_from_sample, predict_with_model

malware_dir = "C:/malware_test/"
model_path = "models/random_forest_model.pkl"
model_type = "random_forest"

results = []

for file_path in Path(malware_dir).glob("*.exe"):
    print(f"\nTesting: {file_path.name}")
    
    try:
        features = extract_features_from_sample(str(file_path))
        if len(features['features']) == 0:
            print(f"  ERROR: No features extracted")
            continue
        
        prediction = predict_with_model(model_path, model_type, features)
        
        results.append({
            'file': file_path.name,
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence']
        })
        
        print(f"  Result: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence']*100:.2f}%")
    
    except Exception as e:
        print(f"  ERROR: {e}")

# Save results
import json
with open("results/batch_test.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nTested {len(results)} files")
```

## Phần 4: Best Practices

### 4.1. Trước Khi Test

- [ ] Tạo snapshot VM
- [ ] Tắt network
- [ ] Backup dữ liệu quan trọng (nếu có)
- [ ] Đảm bảo models đã được train
- [ ] Chuẩn bị malware samples

### 4.2. Trong Khi Test

- [ ] Không mở file malware trực tiếp (chỉ dùng script)
- [ ] Monitor system resources (Task Manager)
- [ ] Ghi lại kết quả
- [ ] Không copy file ra ngoài VM

### 4.3. Sau Khi Test

- [ ] Restore snapshot về clean state
- [ ] Hoặc xóa VM và tạo lại
- [ ] Không copy files từ VM ra ngoài
- [ ] Document kết quả

## Phần 5: Troubleshooting

### 5.1. VM Chậm

**Giải pháp:**
- Tăng RAM allocation
- Enable hardware virtualization trong BIOS
- Đóng các ứng dụng không cần thiết

### 5.2. Không Extract Được Features

**Nguyên nhân:**
- File không phải binary hợp lệ
- Architecture không support (ARM, MIPS)
- File bị corrupt

**Giải pháp:**
- Kiểm tra file type: `file sample.exe`
- Thử với file khác
- Xem logs để biết chi tiết

### 5.3. Model Không Load

**Giải pháp:**
- Kiểm tra đường dẫn model
- Đảm bảo model đã được train
- Kiểm tra model type đúng

### 5.4. Network Vẫn Hoạt Động

**Kiểm tra:**
```bash
# Windows
ipconfig

# Linux
ifconfig

# Nếu vẫn có IP, disable network adapter trong VM settings
```

## Phần 6: Workflow Hoàn Chỉnh

```bash
# 1. Setup VM (một lần)
# - Tạo VM
# - Cài OS
# - Cài Python và dependencies
# - Copy project vào VM
# - Tạo snapshot

# 2. Mỗi lần test:
# - Restore snapshot về clean state
# - Tắt network
# - Copy malware vào VM
# - Test với script
# - Ghi lại kết quả
# - Restore snapshot lại

# 3. Ví dụ session:
cd /path/to/demo2

# Test một file
python scripts/test_malware.py \
    /path/to/malware/sample.exe \
    --model models/random_forest_model.pkl \
    --model-type random_forest \
    --output results/test_$(date +%Y%m%d_%H%M%S).json

# Xem kết quả
cat results/test_*.json

# Restore snapshot khi xong
```

## Phần 7: Lưu Ý Pháp Lý

- ✅ Chỉ test malware mà bạn sở hữu hoặc có quyền phân tích
- ✅ Tuân thủ các quy định về malware research
- ✅ Không chia sẻ malware samples
- ✅ Báo cáo findings một cách có trách nhiệm
- ❌ KHÔNG test malware từ nguồn không rõ ràng
- ❌ KHÔNG chia sẻ malware với người khác

## Tài Liệu Tham Khảo

- [VMware Documentation](https://docs.vmware.com/)
- [VirtualBox Manual](https://www.virtualbox.org/manual/)
- [Malware Analysis Best Practices](https://www.sans.org/white-papers/malware-analysis/)
- [Cuckoo Sandbox](https://cuckoo.sh/) - Alternative sandbox solution

