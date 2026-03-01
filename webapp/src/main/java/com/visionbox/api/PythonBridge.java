package com.visionbox.api;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Utility that communicates with the Python backend.
 * Uses HTTP (FastAPI) for VLMs (Captioning, VQA, CLIP) and ProcessBuilder for
 * training.
 */
@Component
public class PythonBridge {

    @Value("${python.executable:python3}")
    private String pythonExec;

    private final ObjectMapper mapper = new ObjectMapper();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
    private final String serverUrl = "http://localhost:8000/api";

    // Hold reference to the server process so it shuts down when Java shuts down
    private Process serverProcess;

    @PostConstruct
    public void startPythonServer() {
        try {
            // Kill any orphaned Python server from a previous session
            killExistingServerOnPort(8000);

            // Start the Uvicorn FastAPI server in the background
            List<String> cmd = List.of(resolveScriptPath("python"), "-m", "uvicorn", "visionbox.server:app", "--port",
                    "8000");
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            pb.redirectOutput(ProcessBuilder.Redirect.appendTo(new java.io.File("python_server.log")));
            serverProcess = pb.start();
            System.out.println(
                    "Started VisionBox FastAPI Server on port 8000 in the background. Logs in python_server.log");

            // Allow server a moment to start before requests hit it
            Thread.sleep(2000);
        } catch (Exception e) {
            System.err.println("Failed to start FastAPI server: " + e.getMessage());
        }
    }

    private void killExistingServerOnPort(int port) {
        try {
            // Use lsof to find all PIDs on the port, then kill them
            ProcessBuilder pb = new ProcessBuilder("bash", "-c",
                    "lsof -t -i:" + port + " | xargs -r kill -9");
            pb.redirectErrorStream(true);
            Process p = pb.start();
            p.waitFor(5, java.util.concurrent.TimeUnit.SECONDS);
            if (p.exitValue() == 0) {
                System.out.println("Killed orphaned process on port " + port);
                Thread.sleep(500); // Brief pause to let the port fully release
            }
        } catch (Exception e) {
            // Non-fatal: port may have been free already
        }
    }

    private String encodeImageFile(String imagePath) throws Exception {
        byte[] bytes = Files.readAllBytes(Path.of(imagePath));
        return Base64.getEncoder().encodeToString(bytes);
    }

    // ── predict ──────────────────────────────────────────────────────────────

    public List<Map<String, Object>> predict(String imagePath,
            String weightsPath,
            String classMapPath,
            String candidateClasses,
            String model,
            String device,
            int topk) throws Exception {

        // Use Fast HTTP API for CLIP
        if ("clip-vit-base-patch32".equals(model)) {
            Map<String, Object> payload = new HashMap<>();
            payload.put("image_base64", encodeImageFile(imagePath));
            if (candidateClasses != null && !candidateClasses.isBlank()) {
                payload.put("candidate_classes", candidateClasses);
            }
            if (device != null && !device.isBlank()) {
                payload.put("device", device);
            }
            payload.put("topk", topk);

            String jsonBytes = mapper.writeValueAsString(payload);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serverUrl + "/predict"))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofMinutes(5))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBytes))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new RuntimeException("Python server error: " + response.body());
            }

            Map<String, Object> respMap = mapper.readValue(response.body(), Map.class);
            return (List<Map<String, Object>>) respMap.get("predictions");
        }

        // Use ProcessBuilder for standard TorchVision models (they load very fast)
        String script = resolveScriptPath("visionbox-predict");
        List<String> cmd = new ArrayList<>(List.of(
                script,
                "--image", imagePath));
        if (weightsPath != null && !weightsPath.isBlank()) {
            cmd.addAll(List.of("--weights", weightsPath));
        }
        if (classMapPath != null && !classMapPath.isBlank()) {
            cmd.addAll(List.of("--class-map", classMapPath));
        }
        if (candidateClasses != null && !candidateClasses.isBlank()) {
            cmd.addAll(List.of("--classes", candidateClasses));
        }
        if (model != null && !model.isBlank()) {
            cmd.addAll(List.of("--model", model));
        }
        if (device != null && !device.isBlank()) {
            cmd.addAll(List.of("--device", device));
        }
        cmd.addAll(List.of("--topk", String.valueOf(topk)));

        String stdout = runProcess(cmd);

        List<Map<String, Object>> results = new ArrayList<>();
        for (String line : stdout.split("\n")) {
            line = line.strip();
            if (line.isEmpty())
                continue;
            String[] parts = line.split("\t");
            if (parts.length == 2) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("class", parts[0]);
                entry.put("probability", Double.parseDouble(parts[1]));
                results.add(entry);
            }
        }
        return results;
    }

    // ── train ────────────────────────────────────────────────────────────────

    public String train(String dataDir,
            String model,
            Integer epochs,
            Integer batchSize,
            Double lr,
            String device,
            String checkpoint,
            String classMapOut) throws Exception {
        String script = resolveScriptPath("visionbox-train");
        List<String> cmd = new ArrayList<>(List.of(
                script,
                "--data-dir", dataDir));
        if (model != null && !model.isBlank())
            cmd.addAll(List.of("--model", model));
        if (epochs != null)
            cmd.addAll(List.of("--epochs", epochs.toString()));
        if (batchSize != null)
            cmd.addAll(List.of("--batch-size", batchSize.toString()));
        if (lr != null)
            cmd.addAll(List.of("--lr", lr.toString()));
        if (device != null && !device.isBlank())
            cmd.addAll(List.of("--device", device));
        if (checkpoint != null && !checkpoint.isBlank())
            cmd.addAll(List.of("--checkpoint", checkpoint));
        if (classMapOut != null && !classMapOut.isBlank())
            cmd.addAll(List.of("--save-class-map", classMapOut));

        return runProcess(cmd);
    }

    // ── vqa ──────────────────────────────────────────────────────────────────

    public String vqa(String imagePath, String question, String model, String device) throws Exception {
        Map<String, String> payload = new HashMap<>();
        payload.put("image_base64", encodeImageFile(imagePath));
        payload.put("question", question);
        if (model != null && !model.isBlank()) {
            payload.put("model_name", model);
        }
        if (device != null && !device.isBlank()) {
            payload.put("device", device);
        }

        String jsonBytes = mapper.writeValueAsString(payload);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/vqa"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofMinutes(5))
                .POST(HttpRequest.BodyPublishers.ofString(jsonBytes))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("Python server error: " + response.body());
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> respMap = mapper.readValue(response.body(), Map.class);
        return (String) respMap.get("answer");
    }

    // ── caption ──────────────────────────────────────────────────────────────

    public String caption(String imagePath, String condition, String model, String device, Integer maxPixels)
            throws Exception {
        Map<String, Object> payload = new HashMap<>();
        payload.put("image_base64", encodeImageFile(imagePath));
        if (condition != null && !condition.isBlank()) {
            payload.put("condition", condition);
        }
        if (model != null && !model.isBlank()) {
            payload.put("model_name", model);
        }
        if (device != null && !device.isBlank()) {
            payload.put("device", device);
        }
        if (maxPixels != null) {
            payload.put("max_pixels", maxPixels);
        }

        String jsonBytes = mapper.writeValueAsString(payload);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/caption"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofMinutes(5))
                .POST(HttpRequest.BodyPublishers.ofString(jsonBytes))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("Python server error: " + response.body());
        }

        Map<String, Object> respMap = mapper.readValue(response.body(), Map.class);
        return (String) respMap.get("caption");
    }

    // ── free memory ───────────────────────────────────────────────────────────

    public void freeMemory() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/free-memory"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(15))
                .POST(HttpRequest.BodyPublishers.noBody())
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("Python server error during memory clear: " + response.body());
        }
    }

    public Map<String, Object> getGpuStats() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/gpu-stats"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(10))
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("Python server error: " + response.body());
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> respMap = mapper.readValue(response.body(), Map.class);
        return respMap;
    }

    // ── preload ──────────────────────────────────────────────────────────────

    public void preload(String model, String task, String device) throws Exception {
        Map<String, String> payload = new HashMap<>();
        if (model != null && !model.isBlank()) {
            payload.put("model_name", model);
        }
        if (task != null && !task.isBlank()) {
            payload.put("task", task);
        }
        if (device != null && !device.isBlank()) {
            payload.put("device", device);
        }

        String jsonBytes = mapper.writeValueAsString(payload);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl + "/preload"))
                .header("Content-Type", "application/json")
                .timeout(Duration.ofMinutes(3))
                .POST(HttpRequest.BodyPublishers.ofString(jsonBytes))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("Python server error during preload: " + response.body());
        }
    }

    public Map<String, Object> getPreloadStatus(String modelName, String device) throws Exception {
        String url = serverUrl + "/preload-status?model_name=" + java.net.URLEncoder.encode(modelName, "UTF-8")
                + "&device=" + java.net.URLEncoder.encode(device, "UTF-8");
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(5))
                .GET()
                .build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        return mapper.readValue(response.body(), Map.class);
    }

    // ── internal ─────────────────────────────────────────────────────────────

    private String resolveScriptPath(String scriptName) {
        if (!pythonExec.contains("/") && !pythonExec.contains("\\")) {
            return scriptName; // system path fallback
        }
        java.io.File pyFile = new java.io.File(pythonExec);
        return new java.io.File(pyFile.getParent(), scriptName).getAbsolutePath();
    }

    private String runProcess(List<String> command) throws Exception {
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        Process process = pb.start();

        String output;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            output = reader.lines().collect(Collectors.joining("\n"));
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Python command failed (exit " + exitCode + "):\n" + output);
        }
        return output;
    }
}
