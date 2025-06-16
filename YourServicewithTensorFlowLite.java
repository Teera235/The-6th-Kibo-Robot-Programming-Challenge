package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.tensorflow.lite.Interpreter;

public class YourService extends KiboRpcService {
    private final int LOOP_MAX = 3;
    private final String TAG = getClass().getSimpleName();
    private final String[] TEMPLATE_NAMES = {"crystal", "emerald", "treasure_box", "coin", "compass", "coral", "fossil", "key", "letter", "shell"};
    private List<PatrolResult> patrolResults = new ArrayList();
    private List<Mat> templates = new ArrayList();

    // TensorFlow Lite variables
    private Interpreter tfliteInterpreter;
    private static final int INPUT_SIZE = 224; // Adjust based on your model
    private static final int NUM_CLASSES = 10; // Number of treasure items
    private static final int CHANNELS = 3; // RGB
    private String astronautTargetItem = ""; // What the astronaut is looking for

    private class PatrolResult {
        int areaId;
        String foundItemName;
        int itemCount;

        PatrolResult(int areaId2, String foundItemName2, int itemCount2) {
            this.areaId = areaId2;
            this.foundItemName = foundItemName2;
            this.itemCount = itemCount2;
        }
    }

    // Helper class for TensorFlow Lite results
    private class TFLiteResult {
        String className;
        float confidence;
        int classIndex;

        TFLiteResult(String className, float confidence, int classIndex) {
            this.className = className;
            this.confidence = confidence;
            this.classIndex = classIndex;
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        initializeTensorFlowLite();
    }

    /**
     * Initialize TensorFlow Lite interpreter
     */
    private void initializeTensorFlowLite() {
        try {
            Log.i(TAG, "Loading TensorFlow Lite model...");
            InputStream inputStream = getAssets().open("best.tflite");
            byte[] modelBytes = new byte[inputStream.available()];
            inputStream.read(modelBytes);
            inputStream.close();

            ByteBuffer modelBuffer = ByteBuffer.allocateDirect(modelBytes.length);
            modelBuffer.put(modelBytes);

            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);

            tfliteInterpreter = new Interpreter(modelBuffer, options);
            Log.i(TAG, "TensorFlow Lite model loaded successfully");
        } catch (IOException e) {
            Log.e(TAG, "Error loading TensorFlow Lite model", e);
            tfliteInterpreter = null;
        }
    }

    /**
     * Preprocess image for TensorFlow Lite
     */
    private ByteBuffer preprocessImage(Mat image) {
        // Convert to RGB
        Mat rgbImage = new Mat();
        if (image.channels() == 4) {
            Imgproc.cvtColor(image, rgbImage, Imgproc.COLOR_BGRA2RGB);
        } else if (image.channels() == 3) {
            Imgproc.cvtColor(image, rgbImage, Imgproc.COLOR_BGR2RGB);
        } else {
            rgbImage = image.clone();
        }

        // Resize to model input size
        Mat resizedImage = new Mat();
        Imgproc.resize(rgbImage, resizedImage, new Size(INPUT_SIZE, INPUT_SIZE));

        // Convert to ByteBuffer and normalize
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * CHANNELS);
        inputBuffer.order(ByteOrder.nativeOrder());

        byte[] imageData = new byte[(int) (resizedImage.total() * resizedImage.channels())];
        resizedImage.get(0, 0, imageData);

        for (int i = 0; i < imageData.length; i += CHANNELS) {
            float r = (imageData[i] & 0xFF) / 255.0f;
            float g = (imageData[i + 1] & 0xFF) / 255.0f;
            float b = (imageData[i + 2] & 0xFF) / 255.0f;

            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }

        return inputBuffer;
    }

    /**
     * Run TensorFlow Lite inference
     */
    private TFLiteResult classifyWithTFLite(Mat image) {
        if (tfliteInterpreter == null) {
            Log.w(TAG, "TensorFlow Lite interpreter not available");
            return null;
        }

        try {
            ByteBuffer inputBuffer = preprocessImage(image);
            float[][] output = new float[1][NUM_CLASSES];

            tfliteInterpreter.run(inputBuffer, output);

            // Find the class with highest confidence
            int maxIndex = 0;
            float maxConfidence = output[0][0];

            for (int i = 1; i < NUM_CLASSES; i++) {
                if (output[0][i] > maxConfidence) {
                    maxConfidence = output[0][i];
                    maxIndex = i;
                }
            }

            if (maxIndex < TEMPLATE_NAMES.length) {
                return new TFLiteResult(TEMPLATE_NAMES[maxIndex], maxConfidence, maxIndex);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error running TensorFlow Lite inference", e);
        }

        return null;
    }

    /**
     * Enhanced template matching with better accuracy
     */
    private TFLiteResult performTemplateMatching(Mat image) {
        if (templates.isEmpty()) {
            Log.w(TAG, "No templates loaded for matching");
            return null;
        }

        // Convert image to grayscale for template matching
        Mat grayImage = new Mat();
        if (image.channels() > 1) {
            Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);
        } else {
            grayImage = image.clone();
        }

        int[] matchCounts = new int[TEMPLATE_NAMES.length];

        // Perform template matching for each template
        for (int i = 0; i < templates.size() && i < TEMPLATE_NAMES.length; i++) {
            Mat template = templates.get(i);
            Mat result = new Mat();

            // Try multiple scales
            for (int scale = 20; scale <= 80; scale += 15) {
                Mat resizedTemplate = new Mat();
                int newHeight = (int)(template.rows() * ((double)scale / template.cols()));
                Imgproc.resize(template, resizedTemplate, new Size(scale, newHeight));

                Imgproc.matchTemplate(grayImage, resizedTemplate, result, Imgproc.TM_CCOEFF_NORMED);

                Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
                if (mmr.maxVal > 0.6) { // Lower threshold for better detection
                    matchCounts[i]++;
                }
            }
        }

        // Find the template with most matches
        int maxMatches = 0;
        int maxIndex = 0;
        for (int i = 0; i < matchCounts.length; i++) {
            if (matchCounts[i] > maxMatches) {
                maxMatches = matchCounts[i];
                maxIndex = i;
            }
        }

        if (maxMatches > 0) {
            return new TFLiteResult(TEMPLATE_NAMES[maxIndex], 0.7f, maxIndex);
        }

        return null;
    }

    /**
     * Enhanced item detection using both TensorFlow Lite and template matching
     */
    private TFLiteResult detectItems(Mat image) {
        // Save image for debugging
        api.saveMatImage(image, "detection_image_" + System.currentTimeMillis() + ".png");

        // Method 1: Try TensorFlow Lite first
        TFLiteResult tfliteResult = classifyWithTFLite(image);
        if (tfliteResult != null && tfliteResult.confidence > 0.6) {
            Log.i(TAG, "TensorFlow Lite detected: " + tfliteResult.className +
                    " (confidence: " + tfliteResult.confidence + ")");
            return tfliteResult;
        }

        // Method 2: Fallback to enhanced template matching
        TFLiteResult templateResult = performTemplateMatching(image);
        if (templateResult != null) {
            Log.i(TAG, "Template matching detected: " + templateResult.className);
            return templateResult;
        }

        // Method 3: Return most common item as fallback
        Log.w(TAG, "No items detected, using fallback");
        return new TFLiteResult("coin", 0.5f, 3); // Default to coin
    }

    protected void runPlan1() {
        Log.i(this.TAG, "start mission");
        this.api.startMission();
        loadTemplates();

        Point point = new Point(10.9d, -9.9d, 4.8d);
        Quaternion quatArea1 = new Quaternion(0.0f, 0.707f, 0.0f, 0.707f);
        Point pointArea2 = new Point(11.0d, -8.5d, 4.5d);
        Quaternion quatArea2 = new Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
        Point pointArea3 = new Point(10.9d, -7.2d, 4.8d);
        Quaternion quatArea3 = new Quaternion(0.0f, -0.707f, 0.0f, 0.707f);

        patrolAndRecognize(1, point, quatArea1);
        patrolAndRecognize(2, pointArea2, quatArea2);
        patrolAndRecognize(3, pointArea3, quatArea3);
        patrolAndRecognize(4, new Point(10.3d, -8.0d, 5.4d), new Quaternion(0.707f, 0.0f, -0.707f, 0.0f));

        Log.i(this.TAG, "Patrol complete. All areas have been scanned.");
        Log.i(this.TAG, "Moving to astronaut to report completion.");
        moveToWrapper(new Point(11.143d, -6.7607d, 4.9654d), new Quaternion(0.0f, 0.0f, 0.707f, 0.707f));
        this.api.reportRoundingCompletion();

        // Detect what the astronaut is looking for
        Log.i(TAG, "Analyzing what the astronaut wants...");
        Mat astronautImage = api.getMatNavCam();
        TFLiteResult astronautResult = detectItems(astronautImage);
        if (astronautResult != null) {
            astronautTargetItem = astronautResult.className;
            Log.i(TAG, "Astronaut is looking for: " + astronautTargetItem);
        }

        // Notify recognition
        api.notifyRecognitionItem();

        // Find the target item location
        Log.i(TAG, "Searching for target item: " + astronautTargetItem);
        Point targetLocation = findTargetItemLocation();

        if (targetLocation != null) {
            Log.i(TAG, "Moving to target item location.");
            moveToWrapper(targetLocation, quatArea1);
        } else {
            Log.i(TAG, "Target item location not found, using default location.");
            moveToWrapper(point, quatArea1);
        }

        Log.i(this.TAG, "Taking final snapshot.");
        this.api.takeTargetItemSnapshot();
        Log.i(this.TAG, "Mission finished.");
        this.api.shutdownFactory();
    }

    /**
     * Find the location where the target item was detected
     */
    private Point findTargetItemLocation() {
        for (PatrolResult result : patrolResults) {
            if (result.foundItemName.equals(astronautTargetItem)) {
                // Return the corresponding area position
                switch (result.areaId) {
                    case 1: return new Point(10.9d, -9.9d, 4.8d);
                    case 2: return new Point(11.0d, -8.5d, 4.5d);
                    case 3: return new Point(10.9d, -7.2d, 4.8d);
                    case 4: return new Point(10.3d, -8.0d, 5.4d);
                }
            }
        }
        return null; // Not found
    }

    private void patrolAndRecognize(int areaId, Point patrolPoint, Quaternion patrolQuat) {
        String str = this.TAG;
        Log.i(str, "Processing Area " + areaId);
        if (!moveToWrapper(patrolPoint, patrolQuat)) {
            String str2 = this.TAG;
            Log.e(str2, "Could not move to Area " + areaId + ". Skipping.");
            this.patrolResults.add(new PatrolResult(areaId, "move_failed", 0));
            return;
        }

        // Get camera image
        Mat matNavCam = this.api.getMatNavCam();

        // Detect AR markers to ensure we're in the right position
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        Aruco.detectMarkers(matNavCam, dictionary, corners, markerIds);

        if (!corners.isEmpty()) {
            Log.i(TAG, "AR markers detected in Area " + areaId);

            // Undistort the image for better detection
            Mat undistortedImage = undistortImage(matNavCam);

            // Detect items using enhanced detection
            TFLiteResult detectionResult = detectItems(undistortedImage);

            if (detectionResult != null) {
                String itemName = detectionResult.className;
                int itemCount = 1; // Assume 1 item for now

                Log.i(TAG, "In Area " + areaId + ", found " + itemCount + " of " + itemName);
                this.api.setAreaInfo(areaId, itemName, itemCount);
                this.patrolResults.add(new PatrolResult(areaId, itemName, itemCount));
            } else {
                Log.w(TAG, "No items detected in Area " + areaId);
                this.api.setAreaInfo(areaId, "unknown", 0);
                this.patrolResults.add(new PatrolResult(areaId, "unknown", 0));
            }
        } else {
            Log.w(TAG, "No AR markers detected in Area " + areaId + ", using fallback detection");
            // Fallback detection without AR markers
            TFLiteResult detectionResult = detectItems(matNavCam);
            String itemName = detectionResult != null ? detectionResult.className : "coin";
            int itemCount = 1;

            this.api.setAreaInfo(areaId, itemName, itemCount);
            this.patrolResults.add(new PatrolResult(areaId, itemName, itemCount));
        }
    }

    /**
     * Undistort camera image using intrinsic parameters
     */
    private Mat undistortImage(Mat image) {
        try {
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);

            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            distCoeffs.put(0, 0, api.getNavCamIntrinsics()[1]);

            Mat undistorted = new Mat();
            Calib3d.undistort(image, undistorted, cameraMatrix, distCoeffs);

            return undistorted;
        } catch (Exception e) {
            Log.w(TAG, "Could not undistort image, using original", e);
            return image;
        }
    }

    private boolean moveToWrapper(Point point, Quaternion quaternion) {
        int retry_count = 0;
        while (retry_count < 3) {
            if (this.api.moveTo(point, quaternion, true).hasSucceeded()) {
                Log.i(this.TAG, "Move successful.");
                return true;
            }
            retry_count++;
            String str = this.TAG;
        }
        Log.e(this.TAG, "Move failed after 3 retries.");
        return false;
    }

    private void loadTemplates() {
        Log.i(this.TAG, "Loading template images into memory.");
        for (String fileName : this.TEMPLATE_NAMES) {
            try {
                InputStream inputStream = getAssets().open(fileName + ".png");
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                Mat mat = new Mat();
                Utils.bitmapToMat(bitmap, mat);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
                this.templates.add(mat);
                inputStream.close();
            } catch (IOException e) {
                Log.e(this.TAG, "Error loading template: " + fileName, e);
            }
        }
        Log.i(this.TAG, "Finished loading " + this.templates.size() + " templates.");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (tfliteInterpreter != null) {
            tfliteInterpreter.close();
        }
    }

    protected void runPlan2() {
    }

    protected void runPlan3() {
    }
}
