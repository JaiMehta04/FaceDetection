"""Quick analysis of what faces the enhanced detector is still missing."""
import pandas as pd

df = pd.read_csv("outputs/csv/quality_RetinaFace_insightface_det_MultiScale.csv")

print("=== Category counts ===")
print(df["category"].value_counts())
print()

missed = df[df["category"] == "undetected"]
detected = df[df["category"] == "detected"]
fp = df[df["category"] == "false_positive"]

print("=== Missed faces: size distribution ===")
print(f"  Mean area : {missed['face_area'].mean():.0f} px2")
print(f"  Median    : {missed['face_area'].median():.0f} px2")
tiny = (missed["face_area"] < 256).sum()
small = (missed["face_area"] < 1024).sum()
med = (missed["face_area"] < 4096).sum()
n = len(missed)
print(f"  < 256 px2 (16x16) : {tiny} ({tiny/n*100:.1f}%)")
print(f"  < 1024 px2 (32x32): {small} ({small/n*100:.1f}%)")
print(f"  < 4096 px2 (64x64): {med} ({med/n*100:.1f}%)")
print()

print("=== Detected faces: size distribution ===")
print(f"  Mean area : {detected['face_area'].mean():.0f} px2")
print(f"  Median    : {detected['face_area'].median():.0f} px2")
print()

print("=== Missed faces: brightness ===")
print(f"  Mean brightness: {missed['face_brightness'].mean():.1f}")
dark = (missed["face_brightness"] < 80).sum()
print(f"  Dark (< 80)    : {dark} ({dark/n*100:.1f}%)")
print()

print("=== Missed faces: image blur ===")
print(f"  Mean image blur: {missed['image_blur'].mean():.1f}")
blurry = (missed["image_blur"] < 50).sum()
print(f"  Blurry (< 50)  : {blurry} ({blurry/n*100:.1f}%)")
print()

print("=== False positives ===")
print(f"  Count: {len(fp)}")
if len(fp) > 0:
    print(f"  Mean area: {fp['face_area'].mean():.0f} px2")
    print(f"  Mean confidence: {fp['confidence'].mean():.3f}")
