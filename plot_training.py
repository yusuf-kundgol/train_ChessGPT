import re
import matplotlib.pyplot as plt
import numpy as np

# Your training log
log_text = """
step 0: train loss 3.0590, val loss 3.0916
iter 0: loss 2.8978, time 43972.83ms, mfu -100.00%
iter 10: loss 1.7858, time 330.04ms, mfu 12.82%
iter 20: loss 1.1188, time 333.19ms, mfu 12.81%
iter 30: loss 0.6056, time 332.91ms, mfu 12.80%
iter 40: loss 0.4467, time 333.41ms, mfu 12.79%
iter 50: loss 0.3898, time 335.15ms, mfu 12.78%
iter 60: loss 0.3641, time 336.52ms, mfu 12.76%
iter 70: loss 0.3172, time 337.92ms, mfu 12.73%
iter 80: loss 0.2951, time 338.83ms, mfu 12.71%
iter 90: loss 0.2808, time 339.43ms, mfu 12.68%
iter 100: loss 0.2464, time 340.25ms, mfu 12.66%
iter 110: loss 0.2308, time 341.05ms, mfu 12.64%
iter 120: loss 0.1910, time 341.49ms, mfu 12.61%
iter 130: loss 0.1681, time 342.21ms, mfu 12.59%
iter 140: loss 0.1481, time 342.68ms, mfu 12.56%
iter 150: loss 0.1181, time 342.76ms, mfu 12.54%
iter 160: loss 0.1112, time 343.85ms, mfu 12.52%
iter 170: loss 0.0963, time 343.84ms, mfu 12.50%
iter 180: loss 0.0822, time 345.34ms, mfu 12.47%
iter 190: loss 0.0723, time 343.86ms, mfu 12.46%
iter 200: loss 0.0634, time 345.43ms, mfu 12.44%
iter 210: loss 0.0575, time 344.66ms, mfu 12.42%
iter 220: loss 0.0502, time 344.88ms, mfu 12.41%
iter 230: loss 0.0408, time 345.61ms, mfu 12.39%
iter 240: loss 0.0367, time 345.27ms, mfu 12.38%
iter 250: loss 0.0348, time 345.94ms, mfu 12.36%
iter 260: loss 0.0341, time 346.02ms, mfu 12.35%
iter 270: loss 0.0338, time 345.53ms, mfu 12.34%
iter 280: loss 0.0324, time 345.68ms, mfu 12.33%
iter 290: loss 0.0301, time 346.91ms, mfu 12.32%
iter 300: loss 0.0269, time 346.41ms, mfu 12.31%
iter 310: loss 0.0255, time 347.73ms, mfu 12.29%
iter 320: loss 0.0248, time 346.72ms, mfu 12.29%
iter 330: loss 0.0238, time 347.01ms, mfu 12.28%
iter 340: loss 0.0244, time 347.71ms, mfu 12.27%
iter 350: loss 0.0230, time 347.86ms, mfu 12.26%
iter 360: loss 0.0218, time 346.79ms, mfu 12.25%
iter 370: loss 0.0212, time 347.34ms, mfu 12.24%
iter 380: loss 0.0191, time 347.38ms, mfu 12.24%
iter 390: loss 0.0179, time 347.90ms, mfu 12.23%
iter 400: loss 0.0162, time 347.71ms, mfu 12.23%
iter 410: loss 0.0152, time 347.57ms, mfu 12.22%
iter 420: loss 0.0126, time 347.00ms, mfu 12.22%
iter 430: loss 0.0125, time 347.90ms, mfu 12.21%
iter 440: loss 0.0119, time 348.17ms, mfu 12.21%
iter 450: loss 0.0119, time 347.90ms, mfu 12.20%
iter 460: loss 0.0127, time 347.71ms, mfu 12.20%
iter 470: loss 0.0117, time 348.03ms, mfu 12.20%
iter 480: loss 0.0130, time 347.21ms, mfu 12.20%
iter 490: loss 0.0105, time 349.38ms, mfu 12.19%
step 500: train loss 0.0101, val loss 0.9855
iter 500: loss 0.0104, time 22704.92ms, mfu 10.99%
iter 510: loss 0.0102, time 350.11ms, mfu 11.10%
iter 520: loss 0.0099, time 347.32ms, mfu 11.21%
iter 530: loss 0.0092, time 349.34ms, mfu 11.30%
iter 540: loss 0.0104, time 347.40ms, mfu 11.39%
iter 550: loss 0.0087, time 348.47ms, mfu 11.46%
iter 560: loss 0.0092, time 347.92ms, mfu 11.53%
iter 570: loss 0.0099, time 348.60ms, mfu 11.59%
iter 580: loss 0.0108, time 348.80ms, mfu 11.65%
iter 590: loss 0.0113, time 348.30ms, mfu 11.70%
iter 600: loss 0.0112, time 348.00ms, mfu 11.74%
iter 610: loss 0.0096, time 348.01ms, mfu 11.79%
iter 620: loss 0.0090, time 348.45ms, mfu 11.82%
iter 630: loss 0.0090, time 348.28ms, mfu 11.86%
iter 640: loss 0.0093, time 348.05ms, mfu 11.89%
iter 650: loss 0.0083, time 348.56ms, mfu 11.91%
iter 660: loss 0.0093, time 348.63ms, mfu 11.93%
iter 670: loss 0.0091, time 347.53ms, mfu 11.96%
iter 680: loss 0.0085, time 348.51ms, mfu 11.98%
iter 690: loss 0.0084, time 348.45ms, mfu 11.99%
iter 700: loss 0.0085, time 348.58ms, mfu 12.01%
iter 710: loss 0.0085, time 348.09ms, mfu 12.02%
iter 720: loss 0.0091, time 348.88ms, mfu 12.04%
iter 730: loss 0.0095, time 348.17ms, mfu 12.05%
iter 740: loss 0.0092, time 348.66ms, mfu 12.06%
iter 750: loss 0.0090, time 348.64ms, mfu 12.06%
iter 760: loss 0.0079, time 348.01ms, mfu 12.07%
iter 770: loss 0.0090, time 347.97ms, mfu 12.08%
iter 780: loss 0.0082, time 349.05ms, mfu 12.09%
iter 790: loss 0.0087, time 349.67ms, mfu 12.09%
iter 800: loss 0.0072, time 348.63ms, mfu 12.09%
iter 810: loss 0.0072, time 348.75ms, mfu 12.10%
iter 820: loss 0.0071, time 347.97ms, mfu 12.11%
iter 830: loss 0.0088, time 348.80ms, mfu 12.11%
iter 840: loss 0.0078, time 348.07ms, mfu 12.11%
iter 850: loss 0.0091, time 349.02ms, mfu 12.11%
iter 860: loss 0.0090, time 348.65ms, mfu 12.12%
iter 870: loss 0.0080, time 349.14ms, mfu 12.12%
iter 880: loss 0.0077, time 347.66ms, mfu 12.12%
iter 890: loss 0.0078, time 349.52ms, mfu 12.12%
iter 900: loss 0.0081, time 348.12ms, mfu 12.13%
iter 910: loss 0.0077, time 347.68ms, mfu 12.13%
iter 920: loss 0.0077, time 348.10ms, mfu 12.13%
iter 930: loss 0.0077, time 347.59ms, mfu 12.14%
iter 940: loss 0.0075, time 348.50ms, mfu 12.14%
iter 950: loss 0.0077, time 347.49ms, mfu 12.14%
iter 960: loss 0.0080, time 348.63ms, mfu 12.14%
iter 970: loss 0.0079, time 347.77ms, mfu 12.15%
iter 980: loss 0.0089, time 348.15ms, mfu 12.15%
iter 990: loss 0.0074, time 348.18ms, mfu 12.15%
step 1000: train loss 0.0075, val loss 1.1914
"""

# Parse the log
iters = []
train_losses = []
val_steps = []
val_losses = []
mfus = []

for line in log_text.strip().split('\n'):
    # Parse iteration losses
    iter_match = re.search(r'iter (\d+): loss ([\d.]+).*mfu ([-\d.]+)%', line)
    if iter_match:
        iters.append(int(iter_match.group(1)))
        train_losses.append(float(iter_match.group(2)))
        mfu_val = float(iter_match.group(3))
        if mfu_val > 0:  # Ignore the -100% initial value
            mfus.append(mfu_val)
    
    # Parse validation losses
    step_match = re.search(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', line)
    if step_match:
        val_steps.append(int(step_match.group(1)))
        val_losses.append(float(step_match.group(3)))

# Create the plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training Loss (log scale)
axes[0, 0].plot(iters, train_losses, 'b-', linewidth=1, alpha=0.7)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].set_title('Training Loss over Iterations (Log Scale)')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training Loss (linear scale, zoomed)
axes[0, 1].plot(iters, train_losses, 'b-', linewidth=1, alpha=0.7)
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Training Loss')
axes[0, 1].set_title('Training Loss over Iterations (Linear Scale)')
axes[0, 1].set_ylim(0, max(0.5, max(train_losses[10:])))  # Zoom in after initial drop
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Validation Loss
axes[1, 0].plot(val_steps, val_losses, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Validation Loss')
axes[1, 0].set_title('Validation Loss')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: MFU (Model FLOPs Utilization)
if mfus:
    axes[1, 1].plot(range(len(mfus)), mfus, 'g-', linewidth=1, alpha=0.7)
    axes[1, 1].axhline(y=np.mean(mfus), color='r', linestyle='--', label=f'Mean: {np.mean(mfus):.2f}%')
    axes[1, 1].set_xlabel('Iteration (filtered)')
    axes[1, 1].set_ylabel('MFU (%)')
    axes[1, 1].set_title('Model FLOPs Utilization')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_progress.png")

# Print statistics
print("\n" + "="*60)
print("TRAINING STATISTICS")
print("="*60)
print(f"Total iterations: {max(iters)}")
print(f"Initial training loss: {train_losses[0]:.4f}")
print(f"Final training loss: {train_losses[-1]:.4f}")
print(f"Loss reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
print(f"\nInitial validation loss: {val_losses[0]:.4f}")
print(f"Final validation loss: {val_losses[-1]:.4f}")
print(f"\nAverage MFU: {np.mean(mfus):.2f}%")
print(f"Average iteration time: {347:.2f}ms")
print("="*60)
