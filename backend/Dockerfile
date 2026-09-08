FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# rapidocr-onnxruntime depends on `opencv-python`, a separate distribution that
# installs the same `cv2` module as the pinned headless build and overwrites it.
# Whichever lands last then wins, so the OpenCV the preprocessing runs on would
# depend on resolver ordering. Drop the GUI build and restore the pinned one.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.13.0.92 \
 && python -c "import cv2; assert cv2.__version__.startswith('4.13'), cv2.__version__"
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

