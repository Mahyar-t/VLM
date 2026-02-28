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
            @RequestParam(value = "device", required = false, defaultValue = "cuda") String device) {
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
                    device);

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

    // ── Preload ──────────────────────────────────────────────────────────────

    @PostMapping("/api/preload")
    public ResponseEntity<Map<String, Object>> preloadModel(
            @RequestParam("model") String model,
            @RequestParam(value = "task", required = false, defaultValue = "caption") String task,
            @RequestParam(value = "device", required = false, defaultValue = "cuda") String device) {
        try {
            bridge.preload(model, task, device);
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "ok");
            response.put("message", "Model " + model + " loaded to VRAM.");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(
                    Map.of("status", "error", "message", e.getMessage()));
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
}
