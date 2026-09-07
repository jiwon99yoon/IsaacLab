#!/usr/bin/env python3
"""
HDF5 파일을 텍스트 파일로 변환하는 간단한 스크립트
"""

import h5py
import argparse


def hdf5_to_text(input_file, output_file):
    """HDF5 파일의 전체 구조를 텍스트 파일로 저장"""

    with open(output_file, 'w', encoding='utf-8') as out:
        # 헤더 작성
        out.write("=" * 70 + "\n")
        out.write(f"HDF5 File Structure: {input_file}\n")
        out.write("=" * 70 + "\n\n")

        # HDF5 파일 열기
        with h5py.File(input_file, 'r') as f:
            # Top-level keys
            out.write(f"Top-level keys: {list(f.keys())}\n\n")

            # 재귀적으로 구조 탐색
            def write_structure(name, obj, indent=0):
                prefix = "  " * indent

                if isinstance(obj, h5py.Group):
                    out.write(f"{prefix}[Group] {name}/\n")
                    # Group attributes
                    if obj.attrs:
                        for attr_name, attr_val in obj.attrs.items():
                            out.write(f"{prefix}  @{attr_name}: {attr_val}\n")

                elif isinstance(obj, h5py.Dataset):
                    out.write(f"{prefix}[Dataset] {name}\n")
                    out.write(f"{prefix}  Shape: {obj.shape}, Dtype: {obj.dtype}\n")
                    # Dataset attributes
                    if obj.attrs:
                        for attr_name, attr_val in obj.attrs.items():
                            out.write(f"{prefix}  @{attr_name}: {attr_val}\n")

            # 모든 항목 순회
            f.visititems(lambda name, obj: write_structure(name, obj, name.count('/')))

        out.write("\n" + "=" * 70 + "\n")
        out.write("Conversion completed successfully\n")
        out.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 file structure to text file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 inspect_hdf5.py --input datasets/dataset.hdf5 --output datasets/dataset_structure.txt
        """
    )

    parser.add_argument('--input', type=str, required=True,
                        help='Input HDF5 file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output text file path')

    args = parser.parse_args()

    # 파일 존재 확인
    import os
    if not os.path.exists(args.input):
        print(f"✗ Error: Input file not found: {args.input}")
        return 1

    # 파일 크기 확인
    file_size = os.path.getsize(args.input)
    if file_size < 2048:
        print(f"✗ Error: HDF5 file is corrupted or incomplete")
        print(f"  File size: {file_size} bytes (minimum required: 2048 bytes)")
        print(f"  File: {args.input}")
        print(f"\nThis usually happens when dataset generation failed or was interrupted.")
        print(f"Please delete the file and regenerate it:")
        print(f"  rm {args.input}")
        return 1

    # 변환 실행
    print(f"Reading HDF5 file: {args.input}")
    try:
        hdf5_to_text(args.input, args.output)
        print(f"✓ Structure saved to: {args.output}")
        return 0
    except OSError as e:
        print(f"✗ Error: Failed to read HDF5 file")
        print(f"  {str(e)}")
        print(f"\nThe file might be corrupted. Try regenerating it.")
        return 1
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
