package com.bilal.potatoleaf;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import org.json.JSONObject;
import org.json.JSONArray;

public class LeafClassifier {

    private static final String TAG = "LeafClassifier";
    private static final int IMG_SIZE = 224;
    private static final String MODEL_ASSET_NAME = "model_fp16.onnx";
    private static final String GATE_CONFIG_ASSET = "gate_config.json";
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD  = {0.229f, 0.224f, 0.225f};

    // Gate thresholds (loaded from gate_config.json)
    private float leafGateThreshold = 0.25f;
    private float confidenceThreshold = 0.30f;
    private float entropyThreshold = 1.8f;

    // Vegetation color gate — minimum fraction of green pixels to be considered a leaf
    private static final float MIN_GREEN_RATIO = 0.10f;
    // HSV bounds for vegetation green (hue in degrees)
    private static final float GREEN_HUE_LO = 60f;
    private static final float GREEN_HUE_HI = 155f;
    private static final float GREEN_SAT_MIN = 0.20f;
    private static final float GREEN_VAL_MIN = 0.15f;

    // Object detection model — pre-trained MobileNetV3-Small (ImageNet-1k)
    private static final String DETECTOR_ASSET_NAME = "leaf_detector_fp16.onnx";
    private static final String DETECTOR_CONFIG_ASSET = "detector_config.json";
    private static final int DETECTOR_TOP_K = 5;

    public static final String[] CLASS_NAMES = {
        "Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"
    };

    private static final Map<String, String> DISEASE_INFO = new HashMap<>();
    static {
        DISEASE_INFO.put("Bacteria",    "Bacterial leaf symptoms can appear as water-soaked or dark necrotic spots.");
        DISEASE_INFO.put("Fungi",       "Fungal infection often presents as expanding lesions with irregular boundaries.");
        DISEASE_INFO.put("Healthy",     "Leaf surface appears visually normal with no major disease symptoms.");
        DISEASE_INFO.put("Nematode",    "Nematode stress may appear through chlorosis, deformation, or weak tissue areas.");
        DISEASE_INFO.put("Pest",        "Pest damage can include chewing marks, punctures, or local tissue destruction.");
        DISEASE_INFO.put("Phytopthora", "Phytophthora symptoms often include dark blight lesions and rapid tissue collapse.");
        DISEASE_INFO.put("Virus",       "Virus infection may produce mottling, mosaic patterns, curling, or stunted growth.");
    }

    private OrtEnvironment env;
    private OrtSession session;

    // Per-class centroid vectors for leaf gate [7][1280]
    private float[][] centroids;
    private int featureDim = 1280;

    // Object detector (ImageNet-based leaf/object gate)
    private OrtSession detectorSession;
    private String[] imagenetClasses;
    private boolean[] isPlantClass;
    private boolean closed = false;

    public LeafClassifier(Context context) throws Exception {
        Context appContext = context.getApplicationContext();
        Log.i(TAG, "Initializing classifier.");

        env = OrtEnvironment.getEnvironment();
        File modelDir = new File(appContext.getFilesDir(), "onnx");
        File modelFile = copyAssetToFile(appContext, MODEL_ASSET_NAME, modelDir);

        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            // NO_OPT required: ALL_OPT strips the dual-output features node
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT);
            session = env.createSession(modelFile.getAbsolutePath(), opts);
        }

        loadGateConfig(appContext);

        // Load object detector model (MobileNetV3-Small for leaf/object gate)
        File detectorFile = copyAssetToFile(appContext, DETECTOR_ASSET_NAME, modelDir);
        try (OrtSession.SessionOptions detOpts = new OrtSession.SessionOptions()) {
            detectorSession = env.createSession(detectorFile.getAbsolutePath(), detOpts);
        }
        loadDetectorConfig(appContext);

        Log.i(TAG, "Classifier initialized successfully.");
    }

    private void loadGateConfig(Context context) {
        try (InputStream is = context.getAssets().open(GATE_CONFIG_ASSET);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }

            JSONObject config = new JSONObject(sb.toString());
            leafGateThreshold = (float) config.getDouble("leaf_gate_threshold");
            confidenceThreshold = (float) config.getDouble("confidence_threshold");
            entropyThreshold = (float) config.getDouble("entropy_threshold");
            featureDim = config.getInt("feature_dim");

            JSONArray classNames = config.getJSONArray("class_names");
            JSONObject centroidObj = config.getJSONObject("centroids");
            centroids = new float[classNames.length()][featureDim];

            for (int i = 0; i < classNames.length(); i++) {
                String cls = classNames.getString(i);
                JSONArray vec = centroidObj.getJSONArray(cls);
                for (int j = 0; j < featureDim; j++) {
                    centroids[i][j] = (float) vec.getDouble(j);
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load gate configuration. Falling back to defaults.", e);
            centroids = null;
        }
    }

    private void loadDetectorConfig(Context context) {
        try (InputStream is = context.getAssets().open(DETECTOR_CONFIG_ASSET);
             BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) sb.append(line);

            JSONObject config = new JSONObject(sb.toString());
            JSONArray names = config.getJSONArray("class_names");
            imagenetClasses = new String[names.length()];
            isPlantClass = new boolean[names.length()];
            for (int i = 0; i < names.length(); i++) {
                imagenetClasses[i] = names.getString(i);
            }
            JSONArray plantIdx = config.getJSONArray("plant_indices");
            for (int i = 0; i < plantIdx.length(); i++) {
                int idx = plantIdx.getInt(i);
                if (idx >= 0 && idx < isPlantClass.length) {
                    isPlantClass[idx] = true;
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load detector configuration. Detector gate will be skipped.", e);
            imagenetClasses = null;
            isPlantClass = null;
        }
    }

    public static class Prediction implements Comparable<Prediction> {
        public final String className;
        public final float probability;
        public final String diseaseNote;
        public final int rank;

        Prediction(String className, float probability, String diseaseNote, int rank) {
            this.className = className;
            this.probability = probability;
            this.diseaseNote = diseaseNote;
            this.rank = rank;
        }

        @Override
        public int compareTo(Prediction o) {
            return Float.compare(o.probability, this.probability);
        }
    }

    public enum GateStatus {
        PASS,
        REJECTED_NOT_VEGETATION,
        REJECTED_LEAF,
        LOW_CONFIDENCE,
        HIGH_ENTROPY
    }

    public static class Result {
        public final List<Prediction> predictions;
        public final long inferenceTimeMs;
        public final float meanBrightness;
        public final String brightnessWarning;
        public final GateStatus gateStatus;
        public final String gateMessage;
        public final float leafSimilarity;
        public final float entropy;
        public final float greenRatio;
        public final String detectedObject;

        Result(List<Prediction> predictions, long inferenceTimeMs,
               float meanBrightness, String brightnessWarning,
               GateStatus gateStatus, String gateMessage,
               float leafSimilarity, float entropy, float greenRatio,
               String detectedObject) {
            this.predictions = predictions;
            this.inferenceTimeMs = inferenceTimeMs;
            this.meanBrightness = meanBrightness;
            this.brightnessWarning = brightnessWarning;
            this.gateStatus = gateStatus;
            this.gateMessage = gateMessage;
            this.leafSimilarity = leafSimilarity;
            this.entropy = entropy;
            this.greenRatio = greenRatio;
            this.detectedObject = detectedObject;
        }
    }

    public synchronized Result classify(Bitmap bitmap) throws Exception {
        ensureUsable();
        if (bitmap == null) {
            throw new IllegalArgumentException("Bitmap is null.");
        }

        // Resize
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true);
        try {
            // Green vegetation ratio (Gate 0 — runs before inference)
            float greenRatio = computeGreenRatio(resized);

            // Brightness check
            float meanBrightness = computeMeanBrightness(resized);
            String brightnessWarning = null;
            if (meanBrightness < 60.0f) {
                brightnessWarning = String.format(Locale.US,
                    "\u26A0 Low-light warning: Mean brightness %.1f < 60. Predictions may be unreliable.", meanBrightness);
            } else if (meanBrightness > 220.0f) {
                brightnessWarning = String.format(Locale.US,
                    "\u26A0 Overexposure warning: Mean brightness %.1f > 220. Check image quality.", meanBrightness);
            }

            // Preprocess: NCHW float tensor with ImageNet normalization
            float[] inputData = new float[3 * IMG_SIZE * IMG_SIZE];
            int[] pixels = new int[IMG_SIZE * IMG_SIZE];
            resized.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE);

            for (int i = 0; i < pixels.length; i++) {
                int px = pixels[i];
                float r = ((Color.red(px)   / 255.0f) - MEAN[0]) / STD[0];
                float g = ((Color.green(px) / 255.0f) - MEAN[1]) / STD[1];
                float b = ((Color.blue(px)  / 255.0f) - MEAN[2]) / STD[2];
                inputData[i]                            = r;
                inputData[i + IMG_SIZE * IMG_SIZE]      = g;
                inputData[i + 2 * IMG_SIZE * IMG_SIZE]  = b;
            }

            long[] shape = {1, 3, IMG_SIZE, IMG_SIZE};

            // ── Gate 0: Object Detection (soft — informational only) ─────────
            // Run MobileNetV3-Small to label the object for user feedback.
            // This is NOT a hard gate — the disease model always runs.
            // Hard rejection is handled by Gate 1 (centroid similarity) which
            // is far more reliable for potato leaves than ImageNet classification.
            String detectedObj = null;
            boolean detectorSaysPlant = true;
            if (detectorSession != null && imagenetClasses != null) {
                FloatBuffer detBuf = FloatBuffer.wrap(inputData);
                try (OnnxTensor detTensor = OnnxTensor.createTensor(env, detBuf, shape);
                     OrtSession.Result detResults = detectorSession.run(
                             Collections.singletonMap("image", detTensor))) {

                    float[] detLogits;
                    try {
                        float[][] arr = (float[][]) detResults.get("logits").get().getValue();
                        detLogits = arr[0];
                    } catch (Exception e) {
                        float[][] arr = (float[][]) detResults.get(0).getValue();
                        detLogits = arr[0];
                    }

                    float[] detProbs = softmax(detLogits);

                    // Sort indices by probability to get top-K
                    Integer[] indices = new Integer[detProbs.length];
                    for (int i = 0; i < indices.length; i++) indices[i] = i;
                    java.util.Arrays.sort(indices, (a, b) -> Float.compare(detProbs[b], detProbs[a]));

                    detectedObj = imagenetClasses[indices[0]];

                    if (isPlantClass != null) {
                        detectorSaysPlant = false;
                        for (int k = 0; k < Math.min(DETECTOR_TOP_K, indices.length); k++) {
                            if (isPlantClass[indices[k]]) {
                                detectorSaysPlant = true;
                                break;
                            }
                        }
                    }
                }
                // Gate 0 no longer rejects — always proceed to the disease model
            }

            // ── Disease model inference ───────────────────────────────────────
            FloatBuffer buffer = FloatBuffer.wrap(inputData);

            float[] logits;
            float[] features = null;
            long inferenceTimeMs;
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, buffer, shape)) {
                long startTime = System.currentTimeMillis();
                try (OrtSession.Result results = session.run(Collections.singletonMap("image", tensor))) {
                    inferenceTimeMs = System.currentTimeMillis() - startTime;

                    // Extract outputs — try by name first, fall back to index
                    try {
                        float[][] logitsArr = (float[][]) results.get("logits").get().getValue();
                        logits = logitsArr[0];
                    } catch (Exception e) {
                        float[][] logitsArr = (float[][]) results.get(0).getValue();
                        logits = logitsArr[0];
                    }

                    if (results.size() >= 2) {
                        try {
                            float[][] featuresArr = (float[][]) results.get("features").get().getValue();
                            features = featuresArr[0];
                        } catch (Exception e) {
                            try {
                                float[][] featuresArr = (float[][]) results.get(1).getValue();
                                features = featuresArr[0];
                            } catch (Exception e2) {
                                Log.w(TAG, "Feature output unavailable. Leaf centroid gate will be skipped.", e2);
                            }
                        }
                    }
                }
            }

            float[] probs = softmax(logits);

            // Build sorted predictions
            List<Prediction> preds = new ArrayList<>();
            for (int i = 0; i < CLASS_NAMES.length; i++) {
                preds.add(new Prediction(
                    CLASS_NAMES[i], probs[i],
                    DISEASE_INFO.getOrDefault(CLASS_NAMES[i], ""), 0
                ));
            }
            Collections.sort(preds);

            List<Prediction> ranked = new ArrayList<>();
            for (int i = 0; i < preds.size(); i++) {
                Prediction p = preds.get(i);
                ranked.add(new Prediction(p.className, p.probability, p.diseaseNote, i + 1));
            }

            // ── 3-Stage Gate ──────────────────────────────────────────────

            // Gate 1: Leaf similarity (cosine similarity with class centroids)
            float maxSimilarity = -1.0f;
            if (features != null && centroids != null) {
                for (float[] centroid : centroids) {
                    float sim = cosineSimilarity(features, centroid);
                    if (sim > maxSimilarity) maxSimilarity = sim;
                }
            }

            // Gate 2: Confidence check
            float topConfidence = ranked.get(0).probability;

            // Gate 3: Shannon entropy
            float entropy = shannonEntropy(probs);

            // Evaluate gates in priority order
            GateStatus status = GateStatus.PASS;
            String gateMessage = null;

            // Gate 1: Feature similarity — reject if backbone features don't match any leaf class
            // Combined with detector signal: only reject when BOTH the centroid gate
            // AND the ImageNet detector agree this is not a plant/leaf.
            boolean obviousNonVegetation = !detectorSaysPlant && greenRatio < MIN_GREEN_RATIO;
            if (obviousNonVegetation) {
                String detectedLabel = detectedObj != null ? detectedObj : "unknown object";
                status = GateStatus.REJECTED_NOT_VEGETATION;
                gateMessage = String.format(Locale.US,
                    "\u26A0 This image does not contain a potato leaf.\n" +
                    "Detected object: %s\n" +
                    "Green content: %.1f%%\n" +
                    "Please capture a clear photo of a potato leaf.",
                    detectedLabel,
                    greenRatio * 100);
            } else if (features != null && centroids != null && maxSimilarity < leafGateThreshold) {
                String detectedLabel = detectedObj != null
                        ? detectedObj
                        : (detectorSaysPlant ? "plant-like object" : "unknown object");
                if (!detectorSaysPlant) {
                    // Both gates agree — high confidence this is NOT a leaf
                    status = GateStatus.REJECTED_NOT_VEGETATION;
                    gateMessage = String.format(Locale.US,
                        "\u26A0 This image does not contain a potato leaf.\n" +
                        "Detected object: %s\n" +
                        "Leaf similarity: %.1f%% (threshold: %.1f%%)\n" +
                        "Please capture a clear photo of a potato leaf.",
                        detectedLabel,
                        maxSimilarity * 100, leafGateThreshold * 100);
                } else {
                    // Detector says plant but centroid similarity is low — warn but don't reject
                    status = GateStatus.LOW_CONFIDENCE;
                    gateMessage = String.format(Locale.US,
                        "\u26A0 Leaf similarity is low (%.1f%%).\n" +
                        "Detected object: %s\n" +
                        "This may be a leaf, but not a recognizable potato-leaf sample.",
                        maxSimilarity * 100,
                        detectedLabel);
                }
            } else if (topConfidence < confidenceThreshold) {
                status = GateStatus.LOW_CONFIDENCE;
                gateMessage = String.format(Locale.US,
                    "\u26A0 Low confidence (%.1f%%). The model is uncertain.\n" +
                    "This may not be a recognizable potato leaf disease.",
                    topConfidence * 100);
            } else if (entropy > entropyThreshold) {
                status = GateStatus.HIGH_ENTROPY;
                gateMessage = String.format(Locale.US,
                    "\u26A0 Ambiguous prediction (entropy: %.2f > %.1f).\n" +
                    "The model cannot distinguish between multiple classes. " +
                    "Try a clearer or closer image.",
                    entropy, entropyThreshold);
            }

            return new Result(ranked, inferenceTimeMs, meanBrightness, brightnessWarning,
                    status, gateMessage, maxSimilarity, entropy, greenRatio, detectedObj);
        } finally {
            if (resized != bitmap && !resized.isRecycled()) {
                resized.recycle();
            }
        }
    }

    private float cosineSimilarity(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = (float)(Math.sqrt(normA) * Math.sqrt(normB));
        return denom > 0 ? dot / denom : 0;
    }

    private float shannonEntropy(float[] probs) {
        float entropy = 0;
        for (float p : probs) {
            if (p > 1e-9f) {
                entropy -= p * (float) Math.log(p);
            }
        }
        return entropy;
    }

    private float computeGreenRatio(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        int[] pixels = new int[w * h];
        bmp.getPixels(pixels, 0, w, 0, 0, w, h);
        float[] hsv = new float[3];
        int greenCount = 0;
        for (int px : pixels) {
            Color.RGBToHSV(Color.red(px), Color.green(px), Color.blue(px), hsv);
            // hsv[0]=hue 0-360, hsv[1]=saturation 0-1, hsv[2]=value 0-1
            if (hsv[0] >= GREEN_HUE_LO && hsv[0] <= GREEN_HUE_HI
                    && hsv[1] >= GREEN_SAT_MIN && hsv[2] >= GREEN_VAL_MIN) {
                greenCount++;
            }
        }
        return (float) greenCount / pixels.length;
    }

    private float computeMeanBrightness(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        int[] pixels = new int[w * h];
        bmp.getPixels(pixels, 0, w, 0, 0, w, h);
        double sum = 0;
        for (int px : pixels) {
            int r = Color.red(px), g = Color.green(px), b = Color.blue(px);
            sum += 0.299 * r + 0.587 * g + 0.114 * b;
        }
        return (float) (sum / pixels.length);
    }

    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v);
        float sum = 0;
        float[] exps = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - max);
            sum += exps[i];
        }
        for (int i = 0; i < exps.length; i++) exps[i] /= sum;
        return exps;
    }

    public synchronized boolean isClosed() {
        return closed;
    }

    private void ensureUsable() {
        if (closed || env == null || session == null) {
            throw new IllegalStateException("Classifier has been released and must be reinitialized.");
        }
    }

    private File copyAssetToFile(Context context, String assetName, File targetDir) throws Exception {
        if (!targetDir.exists() && !targetDir.mkdirs()) {
            throw new IllegalStateException("Unable to create model directory.");
        }

        File targetFile = new File(targetDir, assetName);
        File stampFile = new File(targetDir, assetName + ".stamp");
        String assetStamp = buildAssetStamp(context, assetName);
        String cachedStamp = readCachedStamp(stampFile);
        if (targetFile.exists() && targetFile.length() > 0 && assetStamp.equals(cachedStamp)) {
            return targetFile;
        }

        File tempFile = new File(targetDir, assetName + ".tmp");
        if (tempFile.exists() && !tempFile.delete()) {
            Log.w(TAG, "Could not delete stale temp asset file: " + tempFile.getAbsolutePath());
        }

        try (InputStream input = context.getAssets().open(assetName);
             OutputStream output = new FileOutputStream(tempFile)) {
            byte[] buf = new byte[8192];
            int read;
            while ((read = input.read(buf)) != -1) {
                output.write(buf, 0, read);
            }
        }

        if (targetFile.exists() && !targetFile.delete()) {
            throw new IllegalStateException("Unable to replace cached asset: " + targetFile.getAbsolutePath());
        }
        if (!tempFile.renameTo(targetFile)) {
            throw new IllegalStateException("Unable to finalize cached asset: " + targetFile.getAbsolutePath());
        }

        try (OutputStream output = new FileOutputStream(stampFile)) {
            output.write(assetStamp.getBytes(StandardCharsets.UTF_8));
        }

        return targetFile;
    }

    private String buildAssetStamp(Context context, String assetName) throws Exception {
        long lastUpdateTime = context.getPackageManager()
                .getPackageInfo(context.getPackageName(), 0)
                .lastUpdateTime;
        long assetLength = -1L;
        try (AssetFileDescriptor descriptor = context.getAssets().openFd(assetName)) {
            assetLength = descriptor.getLength();
        } catch (Exception ignored) {
            // Some asset types may not expose an fd; lastUpdateTime still forces refresh on app update.
        }
        return assetName + ":" + lastUpdateTime + ":" + assetLength;
    }

    private String readCachedStamp(File stampFile) {
        if (!stampFile.exists()) {
            return null;
        }
        try (FileInputStream input = new FileInputStream(stampFile)) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            return output.toString(StandardCharsets.UTF_8.name()).trim();
        } catch (Exception ignored) {
            return null;
        }
    }

    public synchronized void close() {
        if (closed) {
            return;
        }

        closed = true;
        Log.i(TAG, "Closing classifier sessions.");
        try { if (detectorSession != null) detectorSession.close(); } catch (Exception e) {
            Log.w(TAG, "Detector session close failed.", e);
        }
        try { if (session != null) session.close(); } catch (Exception e) {
            Log.w(TAG, "Disease session close failed.", e);
        }

        detectorSession = null;
        session = null;
        env = null;
    }
}
