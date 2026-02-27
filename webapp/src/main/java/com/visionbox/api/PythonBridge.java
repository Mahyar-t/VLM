package com.visionbox.api;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Utility that invokes the visionbox Python CLI commands (visionbox-predict,
 * visionbox-train)
 * via ProcessBuilder and parses their output.
 */
@Component
public class PythonBridge {

    @Value("${python.executable:python3}")
    private String pythonExec;

    // ── predict ──────────────────────────────────────────────────────────────

    /**
     * Runs: visionbox-predict --image <path> --weights <weights> --class-map
     * <classMap>
     * [--model <model>] [--device <device>] [--topk <topk>]
     *
     * @return list of maps: [{class: "cat", probability: 0.92}, ...]
     */
    public List<Map<String, Object>> predict(String imagePath,
            String weightsPath,
            String classMapPath,
            String candidateClasses,
            String model,
            String device,
            int topk) throws Exception {
        // Resolve the script path based on the configured pythonExec
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

        // visionbox-predict prints lines like: class_name 0.9234
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

    /**
     * Runs: visionbox-train --data-dir <dataDir> [--model <model>] [--epochs <n>]
     * ...
     *
     * @return the raw stdout output of the training process
     */
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
