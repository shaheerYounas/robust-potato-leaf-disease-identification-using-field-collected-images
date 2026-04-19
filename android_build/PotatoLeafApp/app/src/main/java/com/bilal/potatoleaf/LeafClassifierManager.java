package com.bilal.potatoleaf;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class LeafClassifierManager {

    private static final String TAG = "LeafClassifierManager";

    public interface InitCallback {
        void onSuccess();
        void onError(Exception exception);
    }

    public interface ResultCallback {
        void onSuccess(LeafClassifier.Result result);
        void onError(Exception exception);
    }

    private static volatile LeafClassifierManager instance;

    private final Context appContext;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final ExecutorService modelExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread thread = new Thread(r, "leaf-classifier-worker");
        thread.setDaemon(true);
        return thread;
    });
    private final Object classifierLock = new Object();

    private LeafClassifier classifier;

    private LeafClassifierManager(Context context) {
        this.appContext = context.getApplicationContext();
    }

    public static LeafClassifierManager getInstance(Context context) {
        if (instance == null) {
            synchronized (LeafClassifierManager.class) {
                if (instance == null) {
                    instance = new LeafClassifierManager(context);
                }
            }
        }
        return instance;
    }

    public void ensureInitialized(InitCallback callback) {
        modelExecutor.execute(() -> {
            try {
                getOrCreateClassifier();
                postInitSuccess(callback);
            } catch (Exception e) {
                Log.e(TAG, "Model initialization failed.", e);
                resetInternal("init_failure");
                postInitError(callback, e);
            }
        });
    }

    public void classifyAsync(Bitmap bitmap, ResultCallback callback) {
        modelExecutor.execute(() -> {
            try {
                LeafClassifier localClassifier = getOrCreateClassifier();
                LeafClassifier.Result result = localClassifier.classify(bitmap);
                postResultSuccess(callback, result);
            } catch (Exception e) {
                Log.e(TAG, "Classification failed.", e);
                resetInternal("classification_failure");
                postResultError(callback, e);
            }
        });
    }

    public void release(String reason) {
        modelExecutor.execute(() -> resetInternal(reason));
    }

    private LeafClassifier getOrCreateClassifier() throws Exception {
        synchronized (classifierLock) {
            if (classifier != null && !classifier.isClosed()) {
                Log.d(TAG, "Reusing already initialized classifier.");
                return classifier;
            }

            Log.i(TAG, "Creating classifier instance.");
            classifier = new LeafClassifier(appContext);
            return classifier;
        }
    }

    private void resetInternal(String reason) {
        synchronized (classifierLock) {
            if (classifier == null) {
                return;
            }

            Log.i(TAG, "Releasing classifier. Reason=" + reason);
            try {
                classifier.close();
            } catch (Exception e) {
                Log.w(TAG, "Classifier release raised an exception.", e);
            } finally {
                classifier = null;
            }
        }
    }

    private void postInitSuccess(InitCallback callback) {
        mainHandler.post(callback::onSuccess);
    }

    private void postInitError(InitCallback callback, Exception exception) {
        mainHandler.post(() -> callback.onError(exception));
    }

    private void postResultSuccess(ResultCallback callback, LeafClassifier.Result result) {
        mainHandler.post(() -> callback.onSuccess(result));
    }

    private void postResultError(ResultCallback callback, Exception exception) {
        mainHandler.post(() -> callback.onError(exception));
    }
}
