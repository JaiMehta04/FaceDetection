@echo off
REM ─────────────────────────────────────────────────────────
REM  Run all 6 detector variants on full WIDER FACE dataset
REM  GPU is auto-detected (ONNX CUDA for RetinaFace)
REM ─────────────────────────────────────────────────────────

cd /d "%~dp0"
set PYTHON=venv\Scripts\python.exe

echo ============================================================
echo  Variant 1/6: RetinaFace Baseline
echo ============================================================
%PYTHON% main.py --detectors retinaface --no-anonymize --no-plots --device auto

echo ============================================================
echo  Variant 2/6: MTCNN Baseline
echo ============================================================
%PYTHON% main.py --detectors mtcnn --no-anonymize --no-plots --device auto

echo ============================================================
echo  Variant 3/6: RetinaFace + MultiScale
echo ============================================================
%PYTHON% main.py --detectors retinaface --enhance multiscale --scales 0.75 1.0 1.5 --no-anonymize --no-plots --device auto

echo ============================================================
echo  Variant 4/6: RetinaFace + Tiled + MultiScale
echo ============================================================
%PYTHON% main.py --detectors retinaface --enhance tiled multiscale --scales 0.75 1.0 1.5 --no-anonymize --no-plots --device auto

echo ============================================================
echo  Variant 5/6: MTCNN + MultiScale
echo ============================================================
%PYTHON% main.py --detectors mtcnn --enhance multiscale --scales 0.75 1.0 1.5 --no-anonymize --no-plots --device auto

echo ============================================================
echo  Variant 6/6: MTCNN + Tiled + MultiScale
echo ============================================================
%PYTHON% main.py --detectors mtcnn --enhance tiled multiscale --scales 0.75 1.0 1.5 --no-anonymize --no-plots --device auto

echo ============================================================
echo  Generating Report + All Plots
echo ============================================================
%PYTHON% scripts\generate_report.py

echo ============================================================
echo  DONE - All variants complete!
echo  CSVs:   outputs\csv\
echo  Plots:  outputs\plots\
echo  Report: outputs\reports\
echo  Dashboard: streamlit run app.py
echo ============================================================
pause
