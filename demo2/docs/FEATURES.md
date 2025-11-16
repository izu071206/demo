# Chi Tiết Features

## Tổng Quan

Hệ thống sử dụng 3 loại features chính từ static analysis:

1. **Opcode N-grams**: Patterns trong instructions
2. **CFG Properties**: Thuộc tính của control flow graph
3. **API Calls**: Windows API calls và imports

## 1. Opcode N-grams

### Mô Tả
Trích xuất các sequences của opcodes (n-grams) từ disassembled code.

### Cách Hoạt Động
1. Disassemble binary sử dụng Capstone
2. Trích xuất opcode mnemonics
3. Tạo n-grams (2, 3, 4-grams)
4. Đếm frequency của mỗi n-gram
5. Tạo feature vector từ top n-grams

### Ví Dụ
```
Instructions: mov eax, 1; add eax, 2; mov ebx, eax; ret

2-grams: "mov add", "add mov", "mov ret"
3-grams: "mov add mov", "add mov ret"
4-grams: "mov add mov ret"
```

### Parameters
- `n_grams`: [2, 3, 4] - Sizes của n-grams
- `max_features`: 1000 - Số lượng features tối đa

### Tại Sao Quan Trọng
- Obfuscated code thường có patterns khác với code bình thường
- N-grams capture sequential patterns
- Resistant to một số obfuscation techniques

## 2. CFG Properties

### Mô Tả
Trích xuất các thuộc tính từ Control Flow Graph của binary.

### Metrics

#### Basic Metrics
- **num_nodes**: Số lượng nodes trong CFG
- **num_edges**: Số lượng edges trong CFG
- **avg_degree**: Độ trung bình của nodes

#### Complexity Metrics
- **cyclomatic_complexity**: Độ phức tạp cyclomatic
  - Formula: E - N + 2P
  - E: số edges, N: số nodes, P: số connected components

#### Loop Metrics
- **num_loops**: Số lượng loops (strongly connected components)

#### Depth Metrics
- **max_depth**: Độ sâu tối đa của graph
- **avg_path_length**: Độ dài đường đi trung bình

#### Clustering
- **clustering_coefficient**: Hệ số clustering

### Cách Hoạt Động
1. Sử dụng angr để trích xuất CFG
2. Convert sang NetworkX graph
3. Tính toán các metrics
4. Tạo feature vector

### Tại Sao Quan Trọng
- Obfuscation techniques như control flow flattening thay đổi CFG structure
- Phát hiện các patterns bất thường trong control flow
- Metrics như cyclomatic complexity tăng với obfuscation

## 3. API Calls

### Mô Tả
Trích xuất Windows API calls từ PE files.

### Cách Hoạt Động
1. Parse PE file sử dụng pefile
2. Trích xuất imports từ IAT (Import Address Table)
3. Tìm API calls trong strings
4. Đếm frequency của mỗi API
5. Tạo feature vector

### API Categories

#### File Operations
- CreateFile, ReadFile, WriteFile, DeleteFile
- CopyFile, MoveFile, FindFirstFile

#### Registry
- RegOpenKey, RegSetValue, RegGetValue
- RegDeleteKey

#### Network
- socket, connect, send, recv
- InternetOpen, HttpOpenRequest

#### Process/Thread
- CreateProcess, CreateThread
- OpenProcess, WriteProcessMemory

#### Memory
- VirtualAlloc, VirtualFree
- HeapAlloc, HeapFree

#### Crypto
- CryptEncrypt, CryptDecrypt
- CryptCreateHash

#### System
- GetSystemTime, GetTickCount
- Sleep, ExitProcess

### Tại Sao Quan Trọng
- Malware thường sử dụng specific APIs
- Obfuscated code có thể hide API calls
- Patterns trong API usage có thể chỉ ra obfuscation

## Feature Combination

### Feature Combiner
Module `FeatureCombiner` kết hợp tất cả features:

1. Flatten tất cả feature arrays
2. Concatenate thành một vector
3. Normalize (optional)

### Normalization Methods
- **L2**: L2 normalization
- **MinMax**: Min-max scaling
- **Standard**: Z-score normalization

## Feature Selection

### Current Approach
- Top-k features dựa trên frequency
- Fixed feature dimensions

### Future Improvements
- Feature selection (mutual information, chi-square)
- Dimensionality reduction (PCA, t-SNE)
- Feature importance từ models

## Handling Different Architectures

### Supported
- x86 (32-bit)
- x86-64 (64-bit)
- ARM (partial)

### Extending
Để thêm architecture mới:
1. Update Capstone arch/mode trong `OpcodeExtractor`
2. Đảm bảo angr support architecture
3. Test với samples của architecture đó

## Feature Alignment

### Problem
Different files có thể có different feature dimensions.

### Solution
1. **Padding**: Pad với zeros
2. **Truncation**: Cắt bớt nếu quá dài
3. **Fixed vocabulary**: Sử dụng fixed vocabulary từ training set

### Current Implementation
- Sử dụng padding/truncation trong `generate_dataset.py`
- Cần cải thiện với fixed vocabulary approach

## Performance Considerations

### Opcode Extraction
- Fast với Capstone
- Memory efficient

### CFG Extraction
- Slower với angr
- Có thể skip nếu không cần

### API Extraction
- Fast với pefile
- Chỉ hoạt động với PE files

## Best Practices

### 1. Feature Engineering
- Experiment với different n-gram sizes
- Tune max_features để balance accuracy và performance
- Combine multiple feature types

### 2. Feature Selection
- Remove redundant features
- Focus on discriminative features
- Use model feature importance

### 3. Normalization
- Normalize features trước khi train
- Use same normalization cho inference

### 4. Feature Alignment
- Ensure consistent feature dimensions
- Use fixed vocabulary từ training

