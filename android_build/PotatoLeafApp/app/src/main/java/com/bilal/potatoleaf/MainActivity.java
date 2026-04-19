package com.bilal.potatoleaf;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.InputStream;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int PERM_REQUEST_CODE = 100;
    private static final int MAX_IMAGE_DIMENSION = 1600;

    private ImageView imagePreview;
    private TextView placeholderText;
    private TextView txtBrightnessWarning;
    private TextView txtInferenceTime;
    private CardView resultCard;
    private TextView txtPredictedClass;
    private TextView txtConfidence;
    private TextView txtDiseaseInfo;
    private TextView txtTop3Title;
    private TextView txtTop3;
    private TextView txtLeafConfidenceWarning;
    private TextView txtLeafConfidence;
    private ProgressBar progressBar;
    private Button btnCamera;
    private Button btnGallery;

    private LeafClassifierManager classifierManager;
    private Uri cameraImageUri;
    private boolean modelReady = false;
    private boolean inferenceRunning = false;
    private boolean activityDestroyed = false;

    private final ActivityResultLauncher<Uri> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicture(), success -> {
                Log.d(TAG, "Camera result received. success=" + success + ", uri=" + cameraImageUri);
                if (success && cameraImageUri != null) {
                    processImage(cameraImageUri);
                }
            });

    private final ActivityResultLauncher<Intent> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() != RESULT_OK
                        || result.getData() == null
                        || result.getData().getData() == null) {
                    Log.d(TAG, "Gallery selection cancelled or missing data.");
                    return;
                }

                Uri uri = result.getData().getData();
                final int takeFlags = result.getData().getFlags()
                        & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                        | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
                try {
                    if (takeFlags != 0) {
                        getContentResolver().takePersistableUriPermission(uri, takeFlags);
                    }
                } catch (Exception e) {
                    Log.w(TAG, "Persistable URI permission was not granted. Using transient access.", e);
                }
                processImage(uri);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "onCreate");
        setContentView(R.layout.activity_main);

        classifierManager = LeafClassifierManager.getInstance(getApplicationContext());
        activityDestroyed = false;

        imagePreview = findViewById(R.id.imagePreview);
        placeholderText = findViewById(R.id.placeholderText);
        txtBrightnessWarning = findViewById(R.id.txtBrightnessWarning);
        txtInferenceTime = findViewById(R.id.txtInferenceTime);
        resultCard = findViewById(R.id.resultCard);
        txtPredictedClass = findViewById(R.id.txtPredictedClass);
        txtConfidence = findViewById(R.id.txtConfidence);
        txtDiseaseInfo = findViewById(R.id.txtDiseaseInfo);
        txtTop3Title = findViewById(R.id.txtTop3Title);
        txtTop3 = findViewById(R.id.txtTop3);
        txtLeafConfidenceWarning = findViewById(R.id.txtLeafConfidenceWarning);
        txtLeafConfidence = findViewById(R.id.txtLeafConfidence);
        progressBar = findViewById(R.id.progressBar);
        btnCamera = findViewById(R.id.btnCamera);
        btnGallery = findViewById(R.id.btnGallery);

        btnCamera.setOnClickListener(v -> {
            if (!modelReady || inferenceRunning) {
                Log.d(TAG, "Camera click ignored because modelReady=" + modelReady
                        + ", inferenceRunning=" + inferenceRunning);
                return;
            }
            if (hasCameraPermission()) {
                launchCamera();
            } else {
                requestCameraPermission();
            }
        });

        btnGallery.setOnClickListener(v -> {
            if (!modelReady || inferenceRunning) {
                Log.d(TAG, "Gallery click ignored because modelReady=" + modelReady
                        + ", inferenceRunning=" + inferenceRunning);
                return;
            }
            launchGallery();
        });

        showLoadingState("Loading model...");
        initializeModel();
    }

    @Override
    protected void onStart() {
        super.onStart();
        Log.d(TAG, "onStart");
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
        updateButtonsForState();
    }

    @Override
    protected void onPause() {
        Log.d(TAG, "onPause");
        super.onPause();
    }

    @Override
    protected void onStop() {
        Log.d(TAG, "onStop");
        super.onStop();
    }

    private void initializeModel() {
        showLoadingState("Loading model...");
        classifierManager.ensureInitialized(new LeafClassifierManager.InitCallback() {
            @Override
            public void onSuccess() {
                if (!canSafelyUpdateUi()) {
                    Log.d(TAG, "Skipping model-ready UI update because activity is gone.");
                    return;
                }

                modelReady = true;
                progressBar.setVisibility(View.GONE);
                placeholderText.setText("Tap a button below to capture or select a leaf image");
                if (imagePreview.getDrawable() == null) {
                    placeholderText.setVisibility(View.VISIBLE);
                }
                updateButtonsForState();
                Log.i(TAG, "Model ready.");
            }

            @Override
            public void onError(Exception exception) {
                Log.e(TAG, "Model load failed.", exception);
                if (!canSafelyUpdateUi()) {
                    return;
                }

                modelReady = false;
                inferenceRunning = false;
                progressBar.setVisibility(View.GONE);
                placeholderText.setVisibility(View.VISIBLE);
                placeholderText.setText("Model load failed. Check logcat for details.");
                updateButtonsForState();
                Toast.makeText(MainActivity.this,
                        "Error loading model: " + safeMessage(exception),
                        Toast.LENGTH_LONG).show();
            }
        });
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        Log.d(TAG, "Requesting camera permission.");
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA}, PERM_REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERM_REQUEST_CODE && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.i(TAG, "Camera permission granted.");
            launchCamera();
        } else {
            Log.w(TAG, "Camera permission denied.");
            Toast.makeText(this, "Permission denied. Please grant permission to continue.",
                    Toast.LENGTH_SHORT).show();
        }
    }

    private void launchCamera() {
        try {
            File cacheDir = getExternalCacheDir();
            if (cacheDir == null) {
                throw new IllegalStateException("External cache directory is unavailable.");
            }

            File photoFile = new File(cacheDir, "leaf_photo.jpg");
            if (photoFile.exists() && !photoFile.delete()) {
                Log.w(TAG, "Unable to delete old camera cache file: " + photoFile.getAbsolutePath());
            }

            cameraImageUri = FileProvider.getUriForFile(this,
                    getPackageName() + ".fileprovider", photoFile);
            Log.d(TAG, "Launching camera with URI=" + cameraImageUri);
            cameraLauncher.launch(cameraImageUri);
        } catch (Exception e) {
            Log.e(TAG, "Cannot open camera.", e);
            Toast.makeText(this, "Cannot open camera: " + safeMessage(e),
                    Toast.LENGTH_SHORT).show();
        }
    }

    private void launchGallery() {
        Log.d(TAG, "Launching gallery chooser.");
        Intent documentsIntent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        documentsIntent.addCategory(Intent.CATEGORY_OPENABLE);
        documentsIntent.setType("image/*");
        documentsIntent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);

        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryIntent.setType("image/*");

        Intent chooser = Intent.createChooser(documentsIntent, getString(R.string.gallery_chooser_title));
        chooser.putExtra(Intent.EXTRA_INITIAL_INTENTS, new Intent[]{galleryIntent});
        galleryLauncher.launch(chooser);
    }

    private void processImage(Uri imageUri) {
        if (!modelReady) {
            Toast.makeText(this, "Model is still loading. Please wait.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (inferenceRunning) {
            Toast.makeText(this, "Please wait for the current prediction to finish.", Toast.LENGTH_SHORT).show();
            return;
        }

        Log.i(TAG, "Processing image URI=" + imageUri);
        try {
            Bitmap bitmap = loadBitmapSafely(imageUri);
            if (bitmap == null) {
                throw new IllegalStateException("Bitmap decode returned null.");
            }

            Bitmap rotatedBitmap = fixRotation(imageUri, bitmap);
            imagePreview.setImageBitmap(rotatedBitmap);
            placeholderText.setVisibility(View.GONE);
            resetResultViews();

            inferenceRunning = true;
            progressBar.setVisibility(View.VISIBLE);
            updateButtonsForState();

            classifierManager.classifyAsync(rotatedBitmap, new LeafClassifierManager.ResultCallback() {
                @Override
                public void onSuccess(LeafClassifier.Result result) {
                    inferenceRunning = false;
                    if (!canSafelyUpdateUi()) {
                        Log.d(TAG, "Skipping classification success UI update because activity is gone.");
                        return;
                    }

                    displayResult(result);
                    progressBar.setVisibility(View.GONE);
                    updateButtonsForState();
                    Log.i(TAG, "Classification finished successfully.");
                }

                @Override
                public void onError(Exception exception) {
                    inferenceRunning = false;
                    Log.e(TAG, "Inference failed.", exception);
                    if (!canSafelyUpdateUi()) {
                        return;
                    }

                    progressBar.setVisibility(View.GONE);
                    modelReady = false;
                    updateButtonsForState();
                    Toast.makeText(MainActivity.this,
                            "Inference error: " + safeMessage(exception),
                            Toast.LENGTH_LONG).show();
                    initializeModel();
                }
            });
        } catch (Exception e) {
            Log.e(TAG, "Image processing failed before inference.", e);
            inferenceRunning = false;
            progressBar.setVisibility(View.GONE);
            updateButtonsForState();
            Toast.makeText(this, "Error: " + safeMessage(e), Toast.LENGTH_LONG).show();
        }
    }

    private Bitmap loadBitmapSafely(Uri imageUri) throws Exception {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream boundsStream = getContentResolver().openInputStream(imageUri)) {
            if (boundsStream == null) {
                throw new IllegalStateException("Unable to open image stream for bounds.");
            }
            BitmapFactory.decodeStream(boundsStream, null, bounds);
        }

        BitmapFactory.Options decodeOptions = new BitmapFactory.Options();
        decodeOptions.inSampleSize = calculateInSampleSize(bounds, MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION);
        decodeOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

        try (InputStream decodeStream = getContentResolver().openInputStream(imageUri)) {
            if (decodeStream == null) {
                throw new IllegalStateException("Unable to open image stream for decoding.");
            }
            return BitmapFactory.decodeStream(decodeStream, null, decodeOptions);
        }
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        int height = Math.max(options.outHeight, 1);
        int width = Math.max(options.outWidth, 1);
        int inSampleSize = 1;

        while ((height / inSampleSize) > reqHeight || (width / inSampleSize) > reqWidth) {
            inSampleSize *= 2;
        }
        return Math.max(inSampleSize, 1);
    }

    private Bitmap fixRotation(Uri imageUri, Bitmap bitmap) {
        try (InputStream is = getContentResolver().openInputStream(imageUri)) {
            if (is == null) {
                return bitmap;
            }

            ExifInterface exif = new ExifInterface(is);
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

            int rotationDegrees = 0;
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotationDegrees = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotationDegrees = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotationDegrees = 270;
                    break;
                default:
                    break;
            }
            if (rotationDegrees == 0) {
                return bitmap;
            }

            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            return Bitmap.createBitmap(bitmap, 0, 0,
                    bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        } catch (Exception e) {
            Log.w(TAG, "Failed to read EXIF orientation. Using bitmap as-is.", e);
            return bitmap;
        }
    }

    private void resetResultViews() {
        resultCard.setVisibility(View.GONE);
        txtBrightnessWarning.setVisibility(View.GONE);
        txtInferenceTime.setVisibility(View.GONE);
        txtLeafConfidenceWarning.setVisibility(View.GONE);
    }

    private void showLoadingState(String message) {
        modelReady = false;
        progressBar.setVisibility(View.VISIBLE);
        placeholderText.setVisibility(View.VISIBLE);
        placeholderText.setText(message);
        updateButtonsForState();
    }

    private void updateButtonsForState() {
        boolean enableActions = modelReady && !inferenceRunning;
        btnCamera.setEnabled(enableActions);
        btnGallery.setEnabled(enableActions);
    }

    private boolean canSafelyUpdateUi() {
        return !activityDestroyed && !isFinishing() && !isDestroyed();
    }

    private String safeMessage(Exception exception) {
        String message = exception.getMessage();
        return message == null || message.trim().isEmpty()
                ? exception.getClass().getSimpleName()
                : message;
    }

    private void displayResult(LeafClassifier.Result result) {
        if (result.brightnessWarning != null) {
            txtBrightnessWarning.setText(result.brightnessWarning);
            txtBrightnessWarning.setVisibility(View.VISIBLE);
        } else {
            txtBrightnessWarning.setVisibility(View.GONE);
        }

        txtInferenceTime.setText(String.format(Locale.US,
                "Inference: %d ms", result.inferenceTimeMs));
        txtInferenceTime.setVisibility(View.VISIBLE);

        if (result.gateStatus != LeafClassifier.GateStatus.PASS && result.gateMessage != null) {
            txtLeafConfidenceWarning.setText(result.gateMessage);
            txtLeafConfidenceWarning.setVisibility(View.VISIBLE);
        } else {
            txtLeafConfidenceWarning.setVisibility(View.GONE);
        }

        LeafClassifier.Prediction top1 = result.predictions.get(0);
        float topConfidence = top1.probability;
        boolean hardRejected =
                result.gateStatus == LeafClassifier.GateStatus.REJECTED_NOT_VEGETATION
                        || result.gateStatus == LeafClassifier.GateStatus.REJECTED_LEAF;
        boolean uncertain =
                result.gateStatus == LeafClassifier.GateStatus.LOW_CONFIDENCE
                        || result.gateStatus == LeafClassifier.GateStatus.HIGH_ENTROPY;

        if (hardRejected) {
            txtPredictedClass.setText("Not a Potato Leaf");
            txtConfidence.setText(result.detectedObject != null
                    ? String.format(Locale.US, "Detected: %s", result.detectedObject)
                    : "Rejected by leaf gate");
            txtDiseaseInfo.setText(
                    "No disease prediction is shown because the image did not pass the potato-leaf gate.");
            txtTop3Title.setText("What To Do Next");
            txtTop3.setText(
                    "Capture a single, clear photo of one potato leaf.\n"
                            + "Use good lighting and avoid wood, cloth, hands, or cluttered backgrounds.");
        } else if (uncertain) {
            txtPredictedClass.setText("Uncertain");
            txtConfidence.setText(String.format(Locale.US,
                    "Best guess: %s (%.1f%%)", top1.className, topConfidence * 100));
            txtDiseaseInfo.setText(
                    "The model is not confident enough to return a reliable disease decision yet.");
            txtTop3Title.setText("Top Guesses");
            txtTop3.setText(buildTopPredictions(result));
        } else {
            txtPredictedClass.setText(top1.className);
            txtConfidence.setText(String.format(Locale.US,
                    "Confidence: %.1f%%", topConfidence * 100));
            txtDiseaseInfo.setText(top1.diseaseNote);
            txtTop3Title.setText(getString(R.string.top_predictions_title));
            txtTop3.setText(buildTopPredictions(result));
        }

        if (hardRejected && result.detectedObject != null && result.leafSimilarity >= 0) {
            txtLeafConfidence.setText(String.format(Locale.US,
                    "Detected: %s | Similarity: %.1f%% | Green: %.1f%%",
                    result.detectedObject, result.leafSimilarity * 100, result.greenRatio * 100));
        } else if (hardRejected && result.detectedObject != null) {
            txtLeafConfidence.setText(String.format(Locale.US,
                    "Detected: %s | Green: %.1f%%",
                    result.detectedObject, result.greenRatio * 100));
        } else if (result.leafSimilarity >= 0) {
            txtLeafConfidence.setText(String.format(Locale.US,
                    "Green: %.1f%% | Similarity: %.1f%% | Entropy: %.2f",
                    result.greenRatio * 100, result.leafSimilarity * 100, result.entropy));
        } else {
            txtLeafConfidence.setText(String.format(Locale.US,
                    "Green: %.1f%% | Entropy: %.2f",
                    result.greenRatio * 100, result.entropy));
        }

        resultCard.setVisibility(View.VISIBLE);
    }

    private String buildTopPredictions(LeafClassifier.Result result) {
        StringBuilder sb = new StringBuilder();
        int limit = Math.min(3, result.predictions.size());
        for (int i = 0; i < limit; i++) {
            LeafClassifier.Prediction prediction = result.predictions.get(i);
            sb.append(String.format(Locale.US, "%d. %s - %.1f%%",
                    prediction.rank, prediction.className, prediction.probability * 100));
            if (i < limit - 1) {
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    @Override
    protected void onDestroy() {
        Log.i(TAG, "onDestroy. finishing=" + isFinishing() + ", changingConfigurations=" + isChangingConfigurations());
        activityDestroyed = true;
        cameraImageUri = null;
        imagePreview.setImageDrawable(null);

        if (isFinishing()) {
            classifierManager.release("activity_finishing");
        }

        super.onDestroy();
    }
}
