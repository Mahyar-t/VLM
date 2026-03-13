package com.visionbox.api;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

@RestController
public class ApiController {

    private final PythonBridge bridge;

    /** Supported torchvision backbones (mirrors imgcls_ft/model.py _SUPPORTED) */
    private static final List<String> SUPPORTED_MODELS = List.of(
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "resnet18",
            "resnet50",
            "densenet121",
            "efficientnet_b0");

    public ApiController(PythonBridge bridge) {
        this.bridge = bridge;
    }

    // ── Health ───────────────────────────────────────────────────────────────

    @GetMapping("/api/health")
    public Map<String, String> health() {
        return Map.of("status", "ok");
    }

    // ── Models ───────────────────────────────────────────────────────────────

    @GetMapping("/api/models")
    public Map<String, Object> models() {
        return Map.of("models", SUPPORTED_MODELS);
    }

    // ── Predict ──────────────────────────────────────────────────────────────

    @PostMapping("/api/predict")
    public ResponseEntity<?> predict(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "weights", required = false) String weightsPath,
            @RequestParam(value = "class_map", required = false) String classMapPath,
            @RequestParam(value = "candidate_classes", required = false) String candidateClasses,
            @RequestParam(value = "model", defaultValue = "mobilenet_v3_small") String model,
            @RequestParam(value = "device", defaultValue = "cuda") String device,
            @RequestParam(value = "topk", defaultValue = "5") int topk) {
        try {
            // Save the uploaded image to a temp file
            String originalName = image.getOriginalFilename();
            String suffix = (originalName != null && originalName.contains("."))
                    ? originalName.substring(originalName.lastIndexOf('.'))
                    : ".jpg";
            Path tempImage = Files.createTempFile("imgcls_upload_", suffix);
            image.transferTo(tempImage.toFile());

            List<Map<String, Object>> predictions = bridge.predict(
                    tempImage.toAbsolutePath().toString(),
                    weightsPath,
                    classMapPath,
                    candidateClasses,
                    model,
                    device,
                    topk);

            // Clean up
            Files.deleteIfExists(tempImage);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("predictions", predictions);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
        }
    }

    // ── Detect ───────────────────────────────────────────────────────────────

    @PostMapping("/api/detect")
    public ResponseEntity<?> detect(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "model", defaultValue = "yolo11n.pt") String modelName,
            @RequestParam(value = "threshold", defaultValue = "0.5") float threshold,
            @RequestParam(value = "device", defaultValue = "cuda") String device) {
        try {
            String originalName = image.getOriginalFilename();
            String suffix = (originalName != null && originalName.contains("."))
                    ? originalName.substring(originalName.lastIndexOf('.'))
                    : ".jpg";
            Path tempImage = Files.createTempFile("detect_upload_", suffix);
            image.transferTo(tempImage.toFile());

            Map<String, Object> result = bridge.detect(
                    tempImage.toAbsolutePath().toString(),
                    modelName,
                    threshold,
                    device);

            Files.deleteIfExists(tempImage);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("result", result);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
        }
    }

    @PostMapping("/api/smart-detect")
    public ResponseEntity<?> smartDetect(
            @RequestParam("image") MultipartFile image,
            @RequestParam("user_query") String userQuery,
            @RequestParam(value = "threshold", defaultValue = "0.3") float threshold,
            @RequestParam(value = "device", defaultValue = "cuda") String device) {
        try {
            String originalName = image.getOriginalFilename();
            String suffix = (originalName != null && originalName.contains("."))
                    ? originalName.substring(originalName.lastIndexOf('.'))
                    : ".jpg";
            Path tempImage = Files.createTempFile("smart_detect_upload_", suffix);
            image.transferTo(tempImage.toFile());

            Map<String, Object> result = bridge.smartDetect(
                    tempImage.toAbsolutePath().toString(),
                    userQuery,
                    threshold,
                    device);

            Files.deleteIfExists(tempImage);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("result", result);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
        }
    }

    // ── Smart Detect Status ──────────────────────────────────────────────────

    @GetMapping("/api/smart-detect-status")
    public ResponseEntity<Map<String, Object>> smartDetectStatus() {
        try {
            Map<String, Object> status = bridge.smartDetectStatus();
            return ResponseEntity.ok(status);
        } catch (Exception e) {
            return ResponseEntity.ok(Map.of("stage", "error", "percent", 0, "label", e.getMessage()));
        }
    }

    // ── VQA ──────────────────────────────────────────────────────────────────

    @PostMapping("/api/vqa")
    public ResponseEntity<?> vqa(
            @RequestParam("image") MultipartFile image,
            @RequestParam("question") String question,
            @RequestParam(value = "model", required = false, defaultValue = "Salesforce/blip-vqa-base") String model,
            @RequestParam(value = "device", defaultValue = "cuda") String device) {
        try {
            // Save the uploaded image to a temp file
            String originalName = image.getOriginalFilename();
            String suffix = (originalName != null && originalName.contains("."))
                    ? originalName.substring(originalName.lastIndexOf('.'))
                    : ".jpg";
            Path tempImage = Files.createTempFile("vqa_upload_", suffix);
            image.transferTo(tempImage.toFile());

            String answer = bridge.vqa(
                    tempImage.toAbsolutePath().toString(),
                    question,
                    model,
                    device);

            // Clean up
            Files.deleteIfExists(tempImage);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("answer", answer);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
        }
    }

    // ── Caption ──────────────────────────────────────────────────────────────

    @PostMapping("/api/caption")
    public ResponseEntity<Map<String, Object>> createCaption(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "condition", required = false) String condition,
            @RequestParam(value = "model", required = false, defaultValue = "Salesforce/blip-image-captioning-large") String model,
            @RequestParam(value = "device", required = false, defaultValue = "cuda") String device,
            @RequestParam(value = "maxPixels", required = false) Integer maxPixels,
            @RequestParam(value = "precision", required = false, defaultValue = "4") String precision) {
        try {
            // Save the uploaded image to a temp file
            String originalName = image.getOriginalFilename();
            String suffix = (originalName != null && originalName.contains("."))
                    ? originalName.substring(originalName.lastIndexOf('.'))
                    : ".jpg";
            Path tempImage = Files.createTempFile("caption_upload_", suffix);
            image.transferTo(tempImage.toFile());

            String answer = bridge.caption(
                    tempImage.toAbsolutePath().toString(),
                    condition,
                    model,
                    device,
                    maxPixels,
                    precision);

            // Clean up
            Files.deleteIfExists(tempImage);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("caption", answer);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
        }
    }

    // ── Free Memory ──────────────────────────────────────────────────────────

    @PostMapping("/api/free-memory")
    public ResponseEntity<Map<String, Object>> freeMemory() {
        try {
            bridge.freeMemory();
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("message", "GPU memory cleared.");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
        }
    }

    @GetMapping("/api/gpu-stats")
    public ResponseEntity<Map<String, Object>> gpuStats() {
        try {
            return ResponseEntity.ok(bridge.getGpuStats());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("error", e.getMessage()));
        }
    }

    // ── Preload ──────────────────────────────────────────────────────────────

    @PostMapping("/api/preload")
    public ResponseEntity<Map<String, Object>> preloadModel(
            @RequestParam("model") String model,
            @RequestParam(value = "task", required = false, defaultValue = "caption") String task,
            @RequestParam(value = "device", required = false, defaultValue = "cuda") String device,
            @RequestParam(value = "precision", required = false, defaultValue = "4") String precision) {
        // Fire-and-forget: launch preload in a background thread and return
        // immediately. The frontend polls /api/preload-status for progress.
        final String m = model, t = task, d = device, p = precision;
        new Thread(() -> {
            try {
                bridge.preload(m, t, d, p);
            } catch (Exception e) {
                System.err.println("Background preload failed: " + e.getMessage());
            }
        }).start();

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status", "ok");
        response.put("message", "Preload started for " + model);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/api/preload-status")
    public ResponseEntity<Map<String, Object>> preloadStatus(
            @RequestParam("model_name") String modelName,
            @RequestParam(value = "device", defaultValue = "cuda") String device,
            @RequestParam(value = "precision", required = false, defaultValue = "4") String precision) {
        try {
            Map<String, Object> status = bridge.getPreloadStatus(modelName, device, precision);
            return ResponseEntity.ok(status);
        } catch (Exception e) {
            return ResponseEntity.ok(Map.of("stage", "error", "percent", 0, "label", e.getMessage()));
        }
    }

    // ── Train ────────────────────────────────────────────────────────────────

    @PostMapping("/api/train")
    public ResponseEntity<?> train(@RequestBody Map<String, Object> body) {
        try {
            String dataDir = (String) body.get("data_dir");
            if (dataDir == null || dataDir.isBlank()) {
                return ResponseEntity.badRequest().body(
                        Map.of("status", "error", "message", "data_dir is required"));
            }

            String model = (String) body.getOrDefault("model", "mobilenet_v3_small");
            Integer epochs = body.containsKey("epochs") ? ((Number) body.get("epochs")).intValue() : null;
            Integer batchSize = body.containsKey("batch_size") ? ((Number) body.get("batch_size")).intValue() : null;
            Double lr = body.containsKey("lr") ? ((Number) body.get("lr")).doubleValue() : null;
            String device = (String) body.getOrDefault("device", "cpu");
            String checkpoint = (String) body.getOrDefault("checkpoint", "best.pt");
            String classMap = (String) body.getOrDefault("save_class_map", "class_to_idx.json");

            String output = bridge.train(dataDir, model, epochs, batchSize, lr, device, checkpoint, classMap);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("output", output);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
        }
    }

    // ── Video Classify ────────────────────────────────────────────────────────

    @PostMapping("/api/video/classify")
    public ResponseEntity<?> videoClassify(
            @RequestParam("video") MultipartFile video,
            @RequestParam(value = "model", defaultValue = "facebook/vjepa2-vitl-fpc16-256-ssv2") String model,
            @RequestParam(value = "device", defaultValue = "cuda") String device,
            @RequestParam(value = "clip_len", defaultValue = "64") int clipLen,
            @RequestParam(value = "use_adaptive_step", defaultValue = "true") boolean adaptiveStep,
            @RequestParam(value = "use_overlap", defaultValue = "true") boolean overlap,
            @RequestParam(value = "aggregate_clips", defaultValue = "true") boolean aggregate) {
        try {
            String originalName = video.getOriginalFilename();
            String suffix = (originalName != null && originalName.contains("."))
                    ? originalName.substring(originalName.lastIndexOf('.'))
                    : ".mp4";
            Path tempVideo = Files.createTempFile("video_upload_", suffix);
            video.transferTo(tempVideo.toFile());

            java.util.List<Map<String, Object>> predictions = bridge.videoClassify(
                    tempVideo.toAbsolutePath().toString(), model, device,
                    clipLen, adaptiveStep, overlap, aggregate);

            Files.deleteIfExists(tempVideo);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("predictions", predictions);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "error", e.getMessage()));
        }
    }

    // ── Video Summarize ───────────────────────────────────────────────────────

    @PostMapping("/api/video/summarize")
    public ResponseEntity<?> videoSummarize(
            @RequestParam("video") MultipartFile video,
            @RequestParam(value = "vjepa_model", defaultValue = "facebook/vjepa2-vitl-fpc16-256-ssv2") String vjepaModel,
            @RequestParam(value = "vjepa_device", defaultValue = "cuda") String vjepaDevice,
            @RequestParam(value = "qwen_device", defaultValue = "cuda") String qwenDevice,
            @RequestParam(value = "clip_len", defaultValue = "64") int clipLen,
            @RequestParam(value = "k_clips", defaultValue = "3") int kClips) {
        try {
            String originalName = video.getOriginalFilename();
            String suffix = (originalName != null && originalName.contains("."))
                    ? originalName.substring(originalName.lastIndexOf('.'))
                    : ".mp4";
            Path tempVideo = Files.createTempFile("video_summarize_", suffix);
            video.transferTo(tempVideo.toFile());

            Map<String, Object> result = bridge.summarizeVideo(
                    tempVideo.toAbsolutePath().toString(), vjepaModel, vjepaDevice, qwenDevice, clipLen, kClips);

            Files.deleteIfExists(tempVideo);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("result", result);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "error", e.getMessage()));
        }
    }

    @GetMapping("/api/video/summarize-status")
    public ResponseEntity<?> videoSummarizeStatus() {
        try {
            Map<String, Object> result = bridge.getSummarizeStatus();
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.ok(Map.of("stage", "idle", "percent", 0, "label", "Waiting..."));
        }
    }

    // ── Video Models Listing ──────────────────────────────────────────────────

    @GetMapping("/api/video/models")
    public ResponseEntity<?> listVideoModels() {
        try {
            Map<String, Object> result = bridge.listVideoModels();
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "error", e.getMessage()));
        }
    }

    // ── Event Narration ──────────────────────────────────────────────────────

    @PostMapping("/api/video/narrate")
    public ResponseEntity<?> narrateVideo(
            @RequestParam("video") MultipartFile video,
            @RequestParam(value = "vjepa_model", defaultValue = "") String vjepaModel,
            @RequestParam(value = "vjepa_device", defaultValue = "cuda") String vjepaDevice,
            @RequestParam(value = "clip_len", defaultValue = "64") int clipLen,
            @RequestParam(value = "sensitivity", defaultValue = "1.5") float sensitivity,
            @RequestParam(value = "cooldown", defaultValue = "2") int cooldown,
            @RequestParam(value = "merge_gap", defaultValue = "3") int mergeGap) {

        try {
            byte[] bytes = video.getBytes();
            String base64 = java.util.Base64.getEncoder().encodeToString(bytes);

            Map<String, Object> result = bridge.narrateVideo(
                    base64, vjepaModel, vjepaDevice, clipLen, sensitivity, cooldown, mergeGap);
            return ResponseEntity.ok(result);

        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "error", e.getMessage()));
        }
    }

    @GetMapping("/api/video/narrate-status")
    public ResponseEntity<?> videoNarrateStatus() {
        try {
            Map<String, Object> result = bridge.getNarrateStatus();
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.ok(Map.of("stage", "idle", "percent", 0, "label", "Waiting..."));
        }
    }
}
