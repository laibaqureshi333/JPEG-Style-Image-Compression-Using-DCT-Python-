"""
Image Compression in Frequency and Spatial Domain
JPEG-style pipeline: 8x8 DCT -> quantization -> zigzag -> RLE -> Huffman
Course: Digital Image Processing, COMSATS University Islamabad, Abbottabad Campus
"""
import sys
import heapq
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import cv2

# ---------------------------------------------------------------- input image
# Usage: python image_compression.py path/to/image.png
path = sys.argv[1] if len(sys.argv) > 1 else "man.png"

img = plt.imread(path).astype(float)
if img.max() > 1:
    img /= 255
if img.ndim == 3:  # RGB -> grayscale
    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
img = img[:256, :256] * 255  # crop to 256x256

# Standard JPEG luminance quantization table
quant = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])


# ------------------------------------------------------------------ transforms
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def zigzag(block):
    out = []
    for diag in range(15):
        for r in range(8):
            c = diag - r
            if 0 <= c < 8:
                out.append(block[r, c] if diag % 2 == 0 else block[c, r])
    return out


def inverse_zigzag(arr):
    block = np.zeros((8, 8))
    idx = 0
    for diag in range(15):
        for r in range(8):
            c = diag - r
            if 0 <= c < 8:
                block[r, c] = arr[idx] if diag % 2 == 0 else arr[idx]
                idx += 1
    return block


# -------------------------------------------------------------- entropy coding
def run_length(seq):
    encoded = []
    count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            count += 1
        else:
            encoded.append((seq[i - 1], count))
            count = 1
    encoded.append((seq[-1], count))
    return encoded


class Node:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_codes(symbols):
    freq = defaultdict(int)
    for s in symbols:
        freq[s] += 1
    heap = [Node(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
        merged = Node(freq=n1.freq + n2.freq)
        merged.left, merged.right = n1, n2
        heapq.heappush(heap, merged)

    codes = {}

    def assign(n, code=""):
        if n.symbol is not None:
            codes[n.symbol] = code
            return
        assign(n.left, code + "0")
        assign(n.right, code + "1")

    assign(heap[0])
    return codes


# ----------------------------------------------------------------- compression
blocks_encoded = []
symbols_list = []
for r in range(0, 256, 8):
    for c in range(0, 256, 8):
        blk = img[r:r + 8, c:c + 8] - 128
        d = dct2(blk)
        q = np.round(d / quant)
        z = zigzag(q)
        rle = run_length(z)
        blocks_encoded.append(rle)
        symbols_list.extend(rle)

huff_table = huffman_codes(symbols_list)

# --------------------------------------------------------------- decompression
recon = np.zeros_like(img)
idx = 0
for r in range(0, 256, 8):
    for c in range(0, 256, 8):
        rle = blocks_encoded[idx]
        flat = [val for val, count in rle for _ in range(count)]
        blk = inverse_zigzag(flat)
        deq = blk * quant
        rec_blk = idct2(deq) + 128
        recon[r:r + 8, c:c + 8] = rec_blk
        idx += 1

recon = np.clip(recon, 0, 255).astype(np.uint8)
orig = np.clip(img, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------- figures
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(orig, cmap='gray')
plt.title("Original")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(recon, cmap='gray')
plt.title("Reconstructed")
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(orig.flatten(), bins=256)
plt.title("Original Histogram")
plt.subplot(1, 2, 2)
plt.hist(recon.flatten(), bins=256)
plt.title("Reconstructed Histogram")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------- metrics
original_bits = orig.size * 8
compressed_bits = sum(len(huff_table[s]) for blk in blocks_encoded for s in blk)
ratio = original_bits / compressed_bits
print(f"Compression Ratio: {ratio:.2f}")

psnr_val = cv2.PSNR(orig, recon)


def ssim_manual(im1, im2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    im1, im2 = im1.astype(np.float64), im2.astype(np.float64)
    mu1, mu2 = im1.mean(), im2.mean()
    sigma1, sigma2 = im1.var(), im2.var()
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))


ssim_val = ssim_manual(orig, recon)
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
