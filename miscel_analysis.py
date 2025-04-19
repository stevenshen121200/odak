import os
import re

base_dir = "ratios"

psnr_values = []
mse_values = []
tv_values = []


dir_pattern = re.compile(r"mse([\d\.]+)tv([\d\.]+)")
psnr_pattern = re.compile(r"PSNR:\s*([\d\.]+)")

for outer_dir in os.listdir(base_dir):
    outer_path = os.path.join(base_dir, outer_dir)
    if os.path.isdir(outer_path):
        dir_match = dir_pattern.match(outer_dir)
        if dir_match:
            mse_val = float(dir_match.group(1))
            tv_val = float(dir_match.group(2))
            mse_values.append(mse_val)
            tv_values.append(tv_val)

            inner_dirs = [d for d in os.listdir(outer_path) if os.path.isdir(os.path.join(outer_path, d))]
            if inner_dirs:
                inner_dir = os.path.join(outer_path, inner_dirs[0])
                psnr_file = os.path.join(inner_dir, "psnr.txt")

                # Check if psnr.txt exists
                if os.path.exists(psnr_file):
                    with open(psnr_file, "r") as f:
                        content = f.read()
                        # Search for the PSNR value in the file content
                        psnr_match = psnr_pattern.search(content)
                        if psnr_match:
                            psnr_val = float(psnr_match.group(1))
                            psnr_values.append(psnr_val)
                        else:
                            print(f"PSNR value not found in file: {psnr_file}")
                else:
                    print(f"File 'psnr.txt' not found in directory: {inner_dir}")
            else:
                print(f"No inner directory found in {outer_path}")
        else:
            print(f"Directory name {outer_dir} does not match the expected pattern.")

import matplotlib.pyplot as plt

ratio_threshold = 0.01
tv_to_mse_ratio = [tv/mse for tv, mse in zip(tv_values, mse_values)]
filtered_tv_to_mse = [ratio for ratio, psnr in zip(tv_to_mse_ratio, psnr_values) if ratio < ratio_threshold]
filtered_psnr = [psnr for ratio, psnr in zip(tv_to_mse_ratio, psnr_values) if ratio < ratio_threshold]

plt.figure(figsize=(8, 6))
plt.scatter(filtered_tv_to_mse, filtered_psnr, marker='o')

plt.xlabel('TV/MSE Ratio')
plt.ylabel('PSNR Value')
plt.title(f'PSNR vs. TV-to-MSE Ratio (Ratio < {ratio_threshold})')
plt.grid(True)
plt.show()

