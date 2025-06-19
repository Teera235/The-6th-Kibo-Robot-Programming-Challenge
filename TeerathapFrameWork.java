package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {
    private final String TAG = "KiboRPC_TemplateMatch";
    private final int LOOP_MAX = 3;

    // *** ปรับค่านี้เพื่อหาความแม่นยำที่พอดี ***
    // หากหาไม่เจอเลย -> ลดค่าลง (เช่น 0.65f)
    // หากเจอของมั่วบ่อย -> เพิ่มค่าขึ้น (เช่น 0.8f)
    private final float TEMPLATE_MATCH_THRESHOLD = 0.7f;

    private final String[] ITEM_CLASSES = {"crystal", "emerald", "treasure_box", "coin", "compass", "coral", "fossil", "key", "letter", "shell", "diamond"};
    private final List<String> TREASURE_ITEMS = Arrays.asList("crystal", "emerald", "diamond");

    private Mat cameraMatrix;
    private List<PatrolResult> patrolResults = new ArrayList<>();
    private BufferedWriter csvWriter;

    private Map<String, Mat> templates = new HashMap<>();

    // Inner classes to hold results
    private class PatrolResult {
        int areaId;
        List<String> foundItems;
        Point patrolPoint;
        Quaternion patrolQuat;
        PatrolResult(int id, List<String> items, Point p, Quaternion q) {
            this.areaId = id; this.foundItems = items; this.patrolPoint = p; this.patrolQuat = q;
        }
    }

    private class DetectionResult {
        String className;
        float confidence;
        Rect2d box;
        DetectionResult(String name, float conf, Rect2d box) {
            this.className = name; this.confidence = conf; this.box = box;
        }
    }

    @Override
    protected void runPlan1() {
        api.startMission();
        initialize();
        executePatrolPath();
        finalizeMission();
        closeCsvWriter();
        api.shutdownFactory();
    }

    private void initialize() {
        try {
            // Load camera intrinsics
            double[][] intrinsics = api.getNavCamIntrinsics();
            cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            if (intrinsics != null && intrinsics.length > 0) cameraMatrix.put(0, 0, intrinsics[0]);

            // Setup CSV writer for logging detections
            File file = new File(getFilesDir(), "detections.csv");
            csvWriter = new BufferedWriter(new FileWriter(file));
            csvWriter.write("Timestamp,Area,Item,Confidence,X,Y,Width,Height\n");

            // Load all item templates from assets folder
            Log.d(TAG, "Attempting to load templates...");
            for (String itemName : ITEM_CLASSES) {
                String assetPath = getAssetFilePath(this, itemName + ".png");
                Mat template = Imgcodecs.imread(assetPath, Imgcodecs.IMREAD_GRAYSCALE);
                if (!template.empty()) {
                    templates.put(itemName, template);
                    Log.d(TAG, "SUCCESS: Template loaded for: " + itemName);
                } else {
                    Log.e(TAG, "FAILURE: Template NOT loaded for: " + itemName + " at path: " + assetPath);
                }
            }
            Log.d(TAG, "Finished loading templates. Total loaded: " + templates.size());
            if (templates.isEmpty()){
                // This will now only trigger if the assets folder is empty or files are missing
                throw new IOException("Critical Error: No templates were loaded from assets.");
            }
        } catch (Exception e) {
            Log.e(TAG, "CRITICAL ERROR during initialization.", e);
        }
    }

    private void closeCsvWriter() {
        try {
            if (csvWriter != null) csvWriter.close();
        } catch (IOException ignored) {}
    }

    private void executePatrolPath() {
        // Points might need slight adjustments to avoid Keep Out Zones
        patrolAndAnalyze(1, new Point(11.15, -10.3, 5.05), new Quaternion(0f, 0f, -0.707f, 0.707f));
        patrolAndAnalyze(2, new Point(11.2, -9.0, 5.0), new Quaternion(0f, 0.707f, 0f, 0.707f));
        patrolAndAnalyze(3, new Point(10.7, -8.0, 5.4), new Quaternion(0f, 0.707f, 0f, 0.707f));
        patrolAndAnalyze(4, new Point(10.6, -6.74, 5.1), new Quaternion(0f, 1f, 0f, 0f));
    }

    private void patrolAndAnalyze(int areaId, Point p, Quaternion q) {
        Log.d(TAG, "Moving to Area " + areaId);
        if (!moveToWrapper(p, q)) {
            Log.e(TAG, "Failed to move to Area " + areaId + ". Skipping.");
            return;
        }

        Log.d(TAG, "Arrived at Area " + areaId + ". Adjusting camera toward AR tag.");
        Mat scene = api.getMatNavCam();
        adjustCameraTowardAR(scene, p);

        Log.d(TAG, "Analyzing Area " + areaId + " with Template Matching.");
        List<DetectionResult> detections = analyzeWithTemplateMatching(scene, areaId);

        List<String> found = new ArrayList<>();
        String landmark = "";
        for (DetectionResult d : detections) {
            found.add(d.className);
            if (landmark.isEmpty() && !TREASURE_ITEMS.contains(d.className)) {
                landmark = d.className;
            }
        }

        if (!detections.isEmpty()) api.notifyRecognitionItem();
        int count = landmark.isEmpty() ? 0 : 1;
        api.setAreaInfo(areaId, landmark, count);
        patrolResults.add(new PatrolResult(areaId, found, p, q));
        Log.d(TAG, "Finished analyzing Area " + areaId + ". Found items: " + found);
    }

    private void finalizeMission() {
        moveToWrapper(new Point(11.143, -6.7607, 4.9654), new Quaternion(0f, 0f, 0.707f, 0.707f));
        api.reportRoundingCompletion();
        Mat scene = api.getMatNavCam();

        List<DetectionResult> clues = analyzeWithTemplateMatching(scene, 99);

        String treasure = null;
        List<String> landmarks = new ArrayList<>();
        for (DetectionResult d : clues) {
            if (TREASURE_ITEMS.contains(d.className) && treasure == null) {
                treasure = d.className;
            } else if (!TREASURE_ITEMS.contains(d.className)) {
                landmarks.add(d.className);
            }
        }

        Log.d(TAG, "Finalizing mission. Treasure clue: " + treasure + ", Landmark clues: " + landmarks);

        for (PatrolResult r : patrolResults) {
            if (treasure != null && r.foundItems.contains(treasure) && !Collections.disjoint(r.foundItems, landmarks)) {
                Log.d(TAG, "Target area found: Area " + r.areaId);
                moveToWrapper(r.patrolPoint, r.patrolQuat);
                api.takeTargetItemSnapshot();
                return;
            }
        }

        Log.e(TAG, "No matching area found. Taking snapshot at fallback location.");
        api.takeTargetItemSnapshot();
    }

    private List<DetectionResult> analyzeWithTemplateMatching(Mat image, int areaId) {
        List<DetectionResult> results = new ArrayList<>();
        Mat grayImage = new Mat();

        if (image.channels() > 1) {
            Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);
        } else {
            image.copyTo(grayImage);
        }

        Mat imageToAnnotate = image.clone();

        for (Map.Entry<String, Mat> entry : templates.entrySet()) {
            String itemName = entry.getKey();
            Mat template = entry.getValue();

            if (grayImage.width() < template.width() || grayImage.height() < template.height()) {
                continue;
            }

            int resultWidth = grayImage.width() - template.width() + 1;
            int resultHeight = grayImage.height() - template.height() + 1;
            Mat resultMat = new Mat(resultHeight, resultWidth, CvType.CV_32FC1);

            Imgproc.matchTemplate(grayImage, template, resultMat, Imgproc.TM_CCOEFF_NORMED);

            Core.MinMaxLocResult mmr = Core.minMaxLoc(resultMat);
            double confidence = mmr.maxVal;
            org.opencv.core.Point matchLoc = mmr.maxLoc;

            if (confidence >= TEMPLATE_MATCH_THRESHOLD) {
                Log.d(TAG, "-> Match FOUND for " + itemName + String.format(Locale.US," with confidence %.3f", confidence));
                Rect2d box = new Rect2d(matchLoc.x, matchLoc.y, template.width(), template.height());
                results.add(new DetectionResult(itemName, (float)confidence, box));

                // Crop and save the detected object image
                Rect roi = new Rect((int)box.x, (int)box.y, (int)box.width, (int)box.height);
                Mat croppedImage = new Mat(image, roi);
                String croppedFilename = String.format(Locale.US, "Cropped_Area%d_%s_Conf%.2f.png", areaId, itemName, confidence);
                api.saveMatImage(croppedImage, croppedFilename);
                Log.d(TAG, "Saved cropped image: " + croppedFilename);

                // Draw bounding box on the annotated image
                Imgproc.rectangle(imageToAnnotate, box.tl(), box.br(), new Scalar(0, 255, 0), 2);
                Imgproc.putText(imageToAnnotate, itemName + String.format(" %.2f", confidence), new org.opencv.core.Point(box.x, box.y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 0, 0), 1);

                try {
                    String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date());
                    csvWriter.write(timestamp + "," + areaId + "," + itemName + "," + confidence + "," + box.x + "," + box.y + "," + box.width + "," + box.height + "\n");
                } catch (IOException ignored) {}
            }
        }

        api.saveMatImage(imageToAnnotate, "Annotated_Template_Area" + areaId + ".png");
        return results;
    }

    private boolean moveToWrapper(Point p, Quaternion q) {
        for (int i = 0; i < LOOP_MAX; i++) {
            if (api.moveTo(p, q, true).hasSucceeded()) return true;
        }
        return false;
    }

    private void adjustCameraTowardAR(Mat scene, Point robotPos) {
        if (cameraMatrix == null) return;
        Mat gray = new Mat();
        if (scene.channels() > 1) Imgproc.cvtColor(scene, gray, Imgproc.COLOR_BGR2GRAY);
        else scene.copyTo(gray);
        Mat thresh = new Mat();
        Imgproc.threshold(gray, thresh, 230, 255, Imgproc.THRESH_BINARY);
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(thresh, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        double maxArea = 0; MatOfPoint best = null;
        for (MatOfPoint c : contours) {
            double area = Imgproc.contourArea(c);
            if (area > maxArea) { maxArea = area; best = c; }
        }
        if (best == null) return;
        Rect roi = Imgproc.boundingRect(best);
        double fx = cameraMatrix.get(0, 0)[0];
        double fy = cameraMatrix.get(1, 1)[0];
        double cx = cameraMatrix.get(0, 2)[0];
        double cy = cameraMatrix.get(1, 2)[0];
        double estimatedZ = 4.45;
        double x = (roi.x + roi.width / 2.0 - cx) * estimatedZ / fx;
        double y = (roi.y + roi.height / 2.0 - cy) * estimatedZ / fy;
        Point targetInWorld = new Point(robotPos.getX() + x, robotPos.getY() + y, estimatedZ);

        Quaternion lookQ = computeLookAtQuaternion(robotPos, targetInWorld);
        api.moveTo(robotPos, lookQ, true);
    }

    private Quaternion computeLookAtQuaternion(Point p1, Point p2) {
        double[] from = { p1.getX(), p1.getY(), p1.getZ() };
        double[] to = { p2.getX(), p2.getY(), p2.getZ() };

        double[] z = { to[0] - from[0], to[1] - from[1], to[2] - from[2] };
        double zNorm = Math.sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
        if (zNorm < 1e-9) return new Quaternion(0, 0, 0, 1);
        z[0] /= zNorm; z[1] /= zNorm; z[2] /= zNorm;

        double[] up = {0, 0, 1};

        double[] x = { up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0] };
        double xNorm = Math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        if (xNorm < 1e-9) {
            x = new double[]{1, 0, 0};
        } else {
            x[0] /= xNorm; x[1] /= xNorm; x[2] /= xNorm;
        }

        double[] y = { z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0] };

        double m00 = x[0], m01 = y[0], m02 = z[0];
        double m10 = x[1], m11 = y[1], m12 = z[1];
        double m20 = x[2], m21 = y[2], m22 = z[2];

        double trace = m00 + m11 + m22;
        float qx, qy, qz, qw;
        if (trace > 0) {
            float s = 0.5f / (float)Math.sqrt(trace + 1.0);
            qw = 0.25f / s;
            qx = (float)((m21 - m12) * s);
            qy = (float)((m02 - m20) * s);
            qz = (float)((m10 - m01) * s);
        } else {
            if (m00 > m11 && m00 > m22) {
                float s = 2.0f * (float)Math.sqrt(1.0 + m00 - m11 - m22);
                qw = (float)((m21 - m12) / s);
                qx = 0.25f * s;
                qy = (float)((m01 + m10) / s);
                qz = (float)((m02 + m20) / s);
            } else if (m11 > m22) {
                float s = 2.0f * (float)Math.sqrt(1.0 + m11 - m00 - m22);
                qw = (float)((m02 - m20) / s);
                qx = (float)((m01 + m10) / s);
                qy = 0.25f * s;
                qz = (float)((m12 + m21) / s);
            } else {
                float s = 2.0f * (float)Math.sqrt(1.0 + m22 - m00 - m11);
                qw = (float)((m10 - m01) / s);
                qx = (float)((m02 + m20) / s);
                qy = (float)((m12 + m21) / s);
                qz = 0.25f * s;
            }
        }
        return new Quaternion(qx, qy, qz, qw);
    }

    private static String getAssetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) return file.getAbsolutePath();
        try (BufferedInputStream is = new BufferedInputStream(context.getAssets().open(assetName));
             FileOutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[1024];
            int read;
            while ((read = is.read(buffer)) != -1) os.write(buffer, 0, read);
        }
        return file.getAbsolutePath();
    }

    @Override protected void runPlan2() {}
    @Override protected void runPlan3() {}
}
