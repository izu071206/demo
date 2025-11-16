"""
Script để tạo obfuscated samples từ benign samples
Sử dụng các kỹ thuật obfuscation cơ bản

LƯU Ý: Script này chỉ là ví dụ. Trong thực tế, bạn nên sử dụng
các công cụ obfuscation chuyên nghiệp như OLLVM, UPX, etc.
"""

import argparse
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_obfuscated_samples(source_dir: str, output_dir: str, method: str = "copy"):
    """
    Tạo obfuscated samples từ benign samples
    
    Args:
        source_dir: Directory chứa benign samples
        output_dir: Directory để lưu obfuscated samples
        method: Method để tạo obfuscated samples
                - "copy": Chỉ copy (để test pipeline)
                - "upx": Sử dụng UPX packer (cần cài UPX)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return
    
    files = list(source_path.rglob('*'))
    files = [f for f in files if f.is_file()]
    
    logger.info(f"Found {len(files)} files in {source_dir}")
    
    if method == "copy":
        # Chỉ copy files (để test pipeline)
        # Trong thực tế, bạn cần obfuscate thật sự
        for file_path in files:
            dest_path = output_path / file_path.name
            shutil.copy2(file_path, dest_path)
            logger.info(f"Copied {file_path.name} to {dest_path}")
    
    elif method == "upx":
        # Sử dụng UPX packer
        import subprocess
        
        for file_path in files:
            dest_path = output_path / f"{file_path.stem}_packed{file_path.suffix}"
            try:
                # Pack với UPX
                subprocess.run(
                    ['upx', '-o', str(dest_path), str(file_path)],
                    check=True,
                    capture_output=True
                )
                logger.info(f"Packed {file_path.name} to {dest_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to pack {file_path.name}: {e}")
            except FileNotFoundError:
                logger.error("UPX not found. Please install UPX first.")
                break
    
    logger.info(f"Created {len(list(output_path.glob('*')))} obfuscated samples in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Tạo obfuscated samples từ benign samples"
    )
    
    parser.add_argument('--source', type=str, default='data/benign/',
                       help='Directory chứa benign samples')
    parser.add_argument('--output', type=str, default='data/obfuscated/',
                       help='Directory để lưu obfuscated samples')
    parser.add_argument('--method', type=str, default='copy',
                       choices=['copy', 'upx'],
                       help='Method để tạo obfuscated samples')
    
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("Tạo Obfuscated Samples")
    logger.info("="*50)
    logger.info(f"Source: {args.source}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Method: {args.method}")
    logger.info("="*50)
    
    create_obfuscated_samples(args.source, args.output, args.method)
    
    logger.info("\n✅ Hoàn thành!")
    logger.info("\n📝 Lưu ý:")
    logger.info("  - Method 'copy' chỉ copy files (để test)")
    logger.info("  - Để tạo obfuscated samples thật, sử dụng:")
    logger.info("    * OLLVM (Obfuscator-LLVM)")
    logger.info("    * UPX packer")
    logger.info("    * Các công cụ obfuscation khác")


if __name__ == "__main__":
    main()

