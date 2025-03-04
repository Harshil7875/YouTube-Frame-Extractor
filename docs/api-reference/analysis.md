# Analysis API Reference

The analysis modules enable advanced AI-powered evaluation of video frames. They provide methods for calculating similarity between images and text, detecting objects, and extracting text via OCR. This API reference covers the primary analysis components.

---

## CLIP Analyzer

The **CLIPAnalyzer** wraps OpenAI's CLIP model to compute the cosine similarity between a given image and a text prompt.

### Key Method

- **`calculate_similarity(image: Image.Image, text: str) -> float`**  
  - **Description:**  
    Preprocesses the input PIL image and tokenizes the text prompt, then computes and returns a similarity score. The score indicates how well the image matches the text, typically ranging from about -1.0 to +1.0.
  - **Usage Example:**
    ```python
    from youtube_frame_extractor.analysis import clip
    from PIL import Image

    analyzer = clip.CLIPAnalyzer(model_name="openai/clip-vit-large-patch14")
    image = Image.open("path/to/frame.jpg")
    similarity = analyzer.calculate_similarity(image, "A photo of a cat")
    print(f"Similarity: {similarity:.4f}")
    ```

---

## Object Detector

The **ObjectDetector** uses pre-trained object detection models (e.g., Faster R-CNN) from torchvision to identify objects within a frame.

### Key Methods

- **`detect_objects(image: Image.Image, score_threshold: float = 0.5) -> List[Dict[str, Any]]`**  
  - **Description:**  
    Runs inference on a single PIL image to detect objects. Returns a list of detections, where each detection includes a label, confidence score, and bounding box.
  - **Usage Example:**
    ```python
    from youtube_frame_extractor.analysis import object_detection
    from PIL import Image

    detector = object_detection.ObjectDetector(model_name="fasterrcnn_resnet50_fpn")
    image = Image.open("path/to/frame.jpg")
    detections = detector.detect_objects(image, score_threshold=0.6)
    for obj in detections:
        print(f"Detected label {obj['label']} with score {obj['score']:.2f} at bbox {obj['bbox']}")
    ```

- **`batch_detect(images: List[Image.Image], score_threshold: float = 0.5) -> List[List[Dict[str, Any]]]`**  
  - **Description:**  
    Processes a list of images and returns a corresponding list of detection results for each image. Useful for batch processing multiple frames.

---

## OCR Analyzer

The **OCRAnalyzer** leverages Tesseract (via pytesseract) to extract text from images.

### Key Methods

- **`extract_text_from_image(image: Image.Image, config_params: Optional[str] = None) -> str`**  
  - **Description:**  
    Extracts and returns recognized text from a single PIL image. Custom Tesseract configurations can be passed as an optional string.
  - **Usage Example:**
    ```python
    from youtube_frame_extractor.analysis import ocr
    from PIL import Image

    ocr_analyzer = ocr.OCRAnalyzer(lang="eng", psm=3)
    image = Image.open("path/to/frame.jpg")
    text = ocr_analyzer.extract_text_from_image(image)
    print("Extracted Text:", text)
    ```

- **`batch_extract_text(images: List[Image.Image], config_params: Optional[str] = None) -> List[str]`**  
  - **Description:**  
    Processes a list of images, returning a list of recognized text strings for each image.

- **`extract_data(image: Image.Image, output_type: str = "dict", config_params: Optional[str] = None) -> Any`**  
  - **Description:**  
    Extracts structured OCR data (such as bounding boxes and HOCR) in the specified format (e.g., `dict`, `hocr`). Useful when detailed OCR layout information is required.

---

## Generic VLM (Vision Language Model) Analyzer

The **VLMAnalyzer** provides a unified interface for text-image similarity analysis. By default, it delegates to the CLIP-based backend but is designed to be extendable to other models.

### Key Methods

- **`calculate_similarity(image: Image.Image, text: str) -> float`**  
  - **Description:**  
    Calculates and returns a similarity score between a given image and a text prompt using the underlying VLM backend.
  - **Usage Example:**
    ```python
    from youtube_frame_extractor.analysis import vlm
    from PIL import Image

    vlm_analyzer = vlm.VLMAnalyzer(model_name="openai/clip-vit-base-patch16")
    image = Image.open("path/to/frame.jpg")
    similarity = vlm_analyzer.calculate_similarity(image, "A person dancing")
    print(f"Similarity: {similarity:.4f}")
    ```

- **`batch_calculate_similarity(images: List[Image.Image], text: str) -> List[float]`**  
  - **Description:**  
    Optionally, processes a list of images in a loop to compute similarity scores for each image. This is useful when evaluating multiple frames at once.

---

These analysis modules are designed to work seamlessly with the extractors. After frames are extracted, they can be processed by the CLIPAnalyzer, ObjectDetector, or OCRAnalyzer to gain further insights from video content. This modular approach allows you to mix and match analysis techniques to suit your specific use case.
