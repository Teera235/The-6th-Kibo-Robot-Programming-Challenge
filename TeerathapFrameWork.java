package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {

    private final String TAG = "KiboRPC_Service";
    private final int LOOP_MAX = 3;
    private final double MATCH_THRESHOLD = 0.45;

    private final String[] TEMPLATE_NAMES = {"crystal", "emerald", "treasure_box", "coin", "compass", "coral", "fossil", "key", "letter", "shell", "diamond"};
    private final List<String> TREASURE_ITEMS = Arrays.asList("crystal", "emerald", "diamond");

    private List<Mat> templates = new ArrayList<Mat>();
    private List<PatrolResult> patrolResults = new ArrayList<PatrolResult>();
    private Mat cameraMatrix, cameraCoefficients;
    private Dictionary arucoDictionary;

    private class PatrolResult {
        int areaId;
        List<String> foundItems;
        gov.nasa.arc.astrobee.types.Point patrolPoint;
        Quaternion patrolQuat;
        PatrolResult(int id, List<String> items, gov.nasa.arc.astrobee.types.Point p, Quaternion q) {
            this.areaId = id; this.foundItems = items; this.patrolPoint = p; this.patrolQuat = q;
        }
    }

    private class AnalysisResult {
        String name;
        double score;
        AnalysisResult(String name, double score) {
            this.name = name; this.score = score;
        }
    }

    @Override
    protected void runPlan1() {
        api.startMission();
        initialize();
        executePatrolPath();
        finalizeMission();
        api.shutdownFactory();
    }

    private void initialize() {
        Log.i(TAG, "Initializing...");
        AssetManager assetManager = getAssets();
        for (int i = 0; i < TEMPLATE_NAMES.length; i++) {
            String fileName = TEMPLATE_NAMES[i];
            try {
                InputStream istr = assetManager.open(fileName + ".png");
                Bitmap bitmap = BitmapFactory.decodeStream(istr);
                Mat mat = new Mat();
                Utils.bitmapToMat(bitmap, mat);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
                templates.add(mat);
                istr.close();
            } catch (Exception e) { Log.e(TAG, "Template loading error: " + fileName, e); }
        }

        cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraCoefficients = new Mat(1, 5, CvType.CV_64F);
        double[][] intrinsics = api.getNavCamIntrinsics();
        if (intrinsics != null && intrinsics.length >= 2) {
            cameraMatrix.put(0, 0, intrinsics[0]);
            cameraCoefficients.put(0, 0, intrinsics[1]);
        }
        arucoDictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
    }

    private void executePatrolPath() {
        Log.i(TAG, "==> Starting Patrol Path...");
        patrolAndAnalyze(1, new gov.nasa.arc.astrobee.types.Point(10.95, -10.0, 5.195), new Quaternion(0f, 0f, -0.707f, 0.707f));
        patrolAndAnalyze(2, new gov.nasa.arc.astrobee.types.Point(10.925, -9.0, 4.4), new Quaternion(0f, 0.707f, 0f, 0.707f));
        patrolAndAnalyze(3, new gov.nasa.arc.astrobee.types.Point(10.925, -7.9, 4.4), new Quaternion(0f, 0.707f, 0f, 0.707f));
        patrolAndAnalyze(4, new gov.nasa.arc.astrobee.types.Point(10.5, -6.85, 4.95), new Quaternion(0f, -0.707f, 0f, 0.707f));
    }

    private void patrolAndAnalyze(int areaId, gov.nasa.arc.astrobee.types.Point patrolPoint, Quaternion patrolQuat) {
        Log.i(TAG, "[Area " + areaId + "] Moving to patrol point...");
        if (!moveToWrapper(patrolPoint, patrolQuat)) {
            Log.e(TAG, "[Area " + areaId + "] Failed to move. Storing empty result.");
            patrolResults.add(new PatrolResult(areaId, new ArrayList<String>(), patrolPoint, patrolQuat));
            return;
        }
        try { Thread.sleep(500); } catch (InterruptedException ignored) {}

        Mat scene = api.getMatNavCam();
        if (scene == null || scene.empty()) {
            Log.e(TAG, "[Area " + areaId + "] Failed to capture image.");
            patrolResults.add(new PatrolResult(areaId, new ArrayList<String>(), patrolPoint, patrolQuat));
            return;
        }

        // The core analysis logic is now in a reusable function
        List<AnalysisResult> results = performImageAnalysis(scene, areaId);

        // Process and report results
        List<String> foundItemsInArea = new ArrayList<String>();
        String landmarkToReport = "";
        if (!results.isEmpty()) {
            for (AnalysisResult res : results) {
                if (res.score >= MATCH_THRESHOLD) {
                    foundItemsInArea.add(res.name);
                    if (landmarkToReport.equals("") && !TREASURE_ITEMS.contains(res.name)) {
                        landmarkToReport = res.name;
                    }
                }
            }
        }

        int reportCount = landmarkToReport.equals("") ? 0 : 1;
        Log.i(TAG, "[Area " + areaId + "] Reporting: '" + landmarkToReport + "', Count: " + reportCount + ". Found items: " + foundItemsInArea.toString());
        api.setAreaInfo(areaId, landmarkToReport, reportCount);
        patrolResults.add(new PatrolResult(areaId, foundItemsInArea, patrolPoint, patrolQuat));
    }

    private void finalizeMission() {
        Log.i(TAG, "==> Finalizing Mission...");
        moveToWrapper(new gov.nasa.arc.astrobee.types.Point(11.143, -6.7607, 4.9654), new Quaternion(0f, 0f, 0.707f, 0.707f));
        api.reportRoundingCompletion();
        try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

        Mat clueScene = api.getMatNavCam();
        if (clueScene == null || clueScene.empty()) {
            api.takeTargetItemSnapshot();
            return;
        }

        List<AnalysisResult> clueResults = performImageAnalysis(clueScene, 99); // Reuse analysis logic

        String treasure = null;
        List<String> landmarks = new ArrayList<String>();
        if (!clueResults.isEmpty()) {
            for (AnalysisResult res : clueResults) {
                if (res.score < MATCH_THRESHOLD) continue;
                if (TREASURE_ITEMS.contains(res.name) && treasure == null) {
                    treasure = res.name;
                } else if (!TREASURE_ITEMS.contains(res.name)) {
                    landmarks.add(res.name);
                }
            }
        }

        Log.i(TAG, "Clue decoded. Treasure: " + treasure + ", Landmarks: " + landmarks.toString());
        if (treasure == null) {
            Log.e(TAG, "Failed to identify treasure from clue. Taking snapshot.");
            api.takeTargetItemSnapshot();
            return;
        }

        for (int i = 0; i < patrolResults.size(); i++) {
            PatrolResult p = patrolResults.get(i);
            boolean hasTreasure = p.foundItems.contains(treasure);
            boolean hasLandmark = false;
            for (int j = 0; j < landmarks.size(); j++) {
                if (p.foundItems.contains(landmarks.get(j))) {
                    hasLandmark = true;
                    break;
                }
            }
            if (hasTreasure && hasLandmark) {
                Log.i(TAG, "Final Target Area is " + p.areaId + ". Moving to take snapshot.");
                moveToWrapper(p.patrolPoint, p.patrolQuat);
                api.takeTargetItemSnapshot();
                return;
            }
        }

        Log.e(TAG, "Could not match clue to any patrolled area. Taking snapshot at last known location.");
        api.takeTargetItemSnapshot();
    }

    // NEW: Central function for all image analysis tasks
    private List<AnalysisResult> performImageAnalysis(Mat scene, int areaId) {
        Mat undistorted = new Mat();
        Calib3d.undistort(scene, undistorted, cameraMatrix, cameraCoefficients);
        api.saveMatImage(undistorted, "Area" + areaId + "_Full_Capture.png");

        List<Mat> corners = new ArrayList<Mat>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(undistorted, arucoDictionary, corners, markerIds);

        List<AnalysisResult> results;

        if (markerIds.total() > 0) {
            Log.i(TAG, "[Area " + areaId + "] AR Tag detected! Using high-accuracy warp analysis.");
            Mat warpedSheet = warpItemSheetFromAR(undistorted, corners.get(0), areaId);
            if (warpedSheet != null && !warpedSheet.empty()) {
                results = analyzeROI(warpedSheet);
            } else {
                Log.e(TAG, "[Area " + areaId + "] Warping failed. Falling back to contour ROI detection.");
                results = fallbackAnalysis(undistorted, areaId);
            }
        } else {
            Log.w(TAG, "[Area " + areaId + "] No AR Tag detected. Falling back to contour ROI detection.");
            results = fallbackAnalysis(undistorted, areaId);
        }

        if (results != null && !results.isEmpty()) {
            Collections.sort(results, new Comparator<AnalysisResult>() {
                public int compare(AnalysisResult r1, AnalysisResult r2) {
                    return Double.compare(r2.score, r1.score);
                }
            });
            logAnalysisResults(results, areaId);
        }
        return results;
    }

    // NEW: Fallback analysis method
    private List<AnalysisResult> fallbackAnalysis(Mat scene, int areaId) {
        Rect roiRect = findItemPaperROI(scene, areaId);
        if (roiRect != null) {
            Mat roiScene = new Mat(scene, roiRect);
            return analyzeROI(roiScene);
        }
        Log.e(TAG, "[Area " + areaId + "] Fallback failed: Could not find ROI.");
        return new ArrayList<AnalysisResult>();
    }

    // NEW: ArUco-based perspective warp function
    private Mat warpItemSheetFromAR(Mat scene, Mat corners, int areaId) {
        final int outputWidth = 300;
        final int outputHeight = 210;

        MatOfPoint2f paperSourcePoints = new MatOfPoint2f();
        double[] tl = corners.get(0, 0);
        double[] tr = corners.get(0, 1);

        double vecX = tr[0] - tl[0];
        double vecY = tr[1] - tl[1];

        // Approximate paper corners based on AR tag position and orientation
        // These values are based on the visual layout in the rulebook
        double paperWidthFactor = 20.0 / 7.0; // Paper width is ~20cm, AR tag is 7cm
        double paperLeftOffsetFactor = paperWidthFactor - 1.0;

        Point paper_tl = new Point(tl[0] - vecX * paperLeftOffsetFactor, tl[1] - vecY * paperLeftOffsetFactor);
        Point paper_tr = new Point(tr[0], tr[1]);
        Point paper_br = new Point(corners.get(0, 2)[0], corners.get(0, 2)[1]);
        Point paper_bl = new Point(corners.get(0, 3)[0] - vecX * paperLeftOffsetFactor, corners.get(0, 3)[1] - vecY * paperLeftOffsetFactor);

        MatOfPoint2f realPaperCorners = new MatOfPoint2f(paper_tl, paper_tr, paper_br, paper_bl);
        MatOfPoint2f paperDestPoints = new MatOfPoint2f(
                new Point(0, 0), new Point(outputWidth, 0),
                new Point(outputWidth, outputHeight), new Point(0, outputHeight)
        );

        Mat transform = Imgproc.getPerspectiveTransform(realPaperCorners, paperDestPoints);
        Mat warpedSheet = new Mat();
        Imgproc.warpPerspective(scene, warpedSheet, transform, new Size(outputWidth, outputHeight));
        api.saveMatImage(warpedSheet, "Debug_Warped_" + areaId + ".png");

        return warpedSheet;
    }

    private Rect findItemPaperROI(Mat scene, int areaId) {
        Mat gray = new Mat();
        if (scene.channels() > 1) Imgproc.cvtColor(scene, gray, Imgproc.COLOR_BGR2GRAY);
        else scene.copyTo(gray);

        Mat binary = new Mat();
        Imgproc.adaptiveThreshold(gray, binary, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 21, 5);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        Rect bestRect = null;
        double sceneArea = scene.rows() * scene.cols();

        for (int i = 0; i < contours.size(); i++) {
            Rect rect = Imgproc.boundingRect(contours.get(i));
            double area = rect.area();
            if (area > 5000 && area < sceneArea * 0.75) {
                if (area > maxArea) {
                    maxArea = area;
                    bestRect = rect;
                }
            }
        }

        if (bestRect != null) {
            bestRect.x = Math.max(0, bestRect.x - 15);
            bestRect.y = Math.max(0, bestRect.y - 15);
            bestRect.width = Math.min(gray.cols() - bestRect.x, bestRect.width + 30);
            bestRect.height = Math.min(gray.rows() - bestRect.y, bestRect.height + 30);
            api.saveMatImage(new Mat(scene, bestRect), "Debug_ROI_" + areaId + ".png");
        }
        return bestRect;
    }

    private List<AnalysisResult> analyzeROI(Mat roiScene) {
        List<AnalysisResult> results = new ArrayList<AnalysisResult>();
        double[] scales = {0.8, 1.0, 1.2, 1.4};
        double[] angles = {-10.0, 0.0, 10.0};

        for (int i = 0; i < templates.size(); i++) {
            Mat baseTemplate = templates.get(i);
            double bestScore = -1;
            for (int s = 0; s < scales.length; s++) {
                Mat scaled = new Mat();
                Imgproc.resize(baseTemplate, scaled, new Size(), scales[s], scales[s]);
                for (int a = 0; a < angles.length; a++) {
                    Mat rotated = rotateMat(scaled, angles[a]);
                    if (roiScene.rows() < rotated.rows() || roiScene.cols() < rotated.cols()) continue;
                    Mat result = new Mat();
                    Imgproc.matchTemplate(roiScene, rotated, result, Imgproc.TM_CCOEFF_NORMED);
                    Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
                    if (mmr.maxVal > bestScore) {
                        bestScore = mmr.maxVal;
                    }
                }
            }
            results.add(new AnalysisResult(TEMPLATE_NAMES[i], bestScore));
        }
        return results;
    }

    private void logAnalysisResults(List<AnalysisResult> results, int areaId) {
        StringBuilder sb = new StringBuilder();
        sb.append("[Area ").append(areaId).append("] Top 5 Matches: ");
        for (int i = 0; i < Math.min(5, results.size()); i++) {
            AnalysisResult res = results.get(i);
            sb.append(res.name).append(String.format(Locale.US, " (%.2f)", res.score));
            if (i < 4) sb.append(", ");
        }
        Log.d(TAG, sb.toString());
    }

    private Mat rotateMat(Mat src, double angle) {
        if (angle == 0) return src;
        org.opencv.core.Point center = new org.opencv.core.Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat rotMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Rect bbox = new RotatedRect(new org.opencv.core.Point(0,0), src.size(), angle).boundingRect();
        rotMat.put(0, 2, rotMat.get(0, 2)[0] + (bbox.width / 2.0) - center.x);
        rotMat.put(1, 2, rotMat.get(1, 2)[0] + (bbox.height / 2.0) - center.y);
        Mat dst = new Mat();
        Imgproc.warpAffine(src, dst, rotMat, bbox.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
        return dst;
    }

    private boolean moveToWrapper(gov.nasa.arc.astrobee.types.Point p, Quaternion q) {
        int count = 0;
        while(count < LOOP_MAX) {
            if(api.moveTo(p, q, true).hasSucceeded()) {
                return true;
            }
            count++;
        }
        Log.e(TAG, "Move failed after " + LOOP_MAX + " retries to: " + p.toString());
        return false;
    }

    @Override protected void runPlan2() {}
    @Override protected void runPlan3() {}
}
