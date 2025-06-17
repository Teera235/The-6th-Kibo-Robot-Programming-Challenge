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
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {

    private final String TAG = getClass().getSimpleName();
    private final int LOOP_MAX = 3;

    private final double MATCH_THRESHOLD = 0.75;

    private final String[] TEMPLATE_NAMES = {"crystal", "emerald", "treasure_box", "coin", "compass", "coral", "fossil", "key", "letter", "shell", "diamond"};
    private final List<String> TREASURE_ITEMS = Arrays.asList("crystal", "emerald", "diamond");

    private List<Mat> templates = new ArrayList<>();
    private List<PatrolResult> patrolResults = new ArrayList<>();
    private Mat cameraMatrix, cameraCoefficients;
    private Dictionary arucoDictionary;

    private class PatrolResult {
        int areaId;
        String bestGuessItem;
        PatrolResult(int id, String item) { this.areaId = id; this.bestGuessItem = item; }
    }

    private class AnalysisResult {
        String name;
        double score;
        AnalysisResult(String name, double score) { this.name = name; this.score = score; }
    }

    @Override
    protected void runPlan1() {
        api.startMission();
        initialize();

        Log.i(TAG, "==> Starting Patrol Phase...");

        // [MODIFIED] กำหนด Point และ Quaternion แบบตายตัว
        // คุณสามารถทดลองปรับแก้ตัวเลขใน Point และ Quaternion นี้ได้โดยตรง
        Point p1 = new Point(11.2, -10.2, 5.0);
        Quaternion q1 = new Quaternion(0.0f, 0.707f, 0.0f, 0.707f); // ค่าเริ่มต้นสำหรับหันหน้าเข้ากำแพง Area 1

        patrolAndAnalyze(1, p1, q1);

        Log.i(TAG, "========== Mission Complete ==========");
        api.shutdownFactory();
    }

    private void initialize() {
        Log.i(TAG, "Initializing...");
        AssetManager assetManager = getAssets();
        for (String fileName : TEMPLATE_NAMES) {
            try (InputStream istr = assetManager.open(fileName + ".png")) {
                Bitmap bitmap = BitmapFactory.decodeStream(istr);
                Mat mat = new Mat();
                Utils.bitmapToMat(bitmap, mat);
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
                Imgproc.resize(mat, mat, new Size(100, 100));
                templates.add(mat);
            } catch (Exception e) {
                Log.e(TAG, "Error loading template: " + fileName, e);
            }
        }

        cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraCoefficients = new Mat(1, 5, CvType.CV_64F);
        double[][] intrinsics = api.getNavCamIntrinsics();
        if (intrinsics != null && intrinsics.length >= 2) {
            cameraMatrix.put(0, 0, intrinsics[0]);
            cameraCoefficients.put(0, 0, intrinsics[1]);
        }

        arucoDictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        Log.i(TAG, "Initialization complete.");
    }

    private void patrolAndAnalyze(int areaId, Point patrolPoint, Quaternion patrolQuat) {
        Log.i(TAG, "--- Patrolling Area " + areaId + " ---");
        if (!moveToWrapper(patrolPoint, patrolQuat)) return;

        try { Thread.sleep(1000); } catch (InterruptedException e) {}

        Mat scene = api.getMatNavCam();
        if (scene == null || scene.empty()) return;

        Mat undistortedScene = new Mat();
        Calib3d.undistort(scene, undistortedScene, cameraMatrix, cameraCoefficients);
        api.saveMatImage(undistortedScene, "Area" + areaId + "_capture.png");

        List<AnalysisResult> results = analyzeSceneWithARTag(undistortedScene);

        if (results.isEmpty()) {
            Log.w(TAG, "Area " + areaId + ": No items found after AR Tag analysis.");
            api.setAreaInfo(areaId, "", 0);
            patrolResults.add(new PatrolResult(areaId, "none"));
            return;
        }

        Collections.sort(results, new Comparator<AnalysisResult>() {
            @Override
            public int compare(AnalysisResult r1, AnalysisResult r2) {
                return Double.compare(r2.score, r1.score);
            }
        });

        Log.i(TAG, "--- Analysis Report for Area " + areaId + " ---");
        for(AnalysisResult res : results) {
            Log.i(TAG, String.format(" > Item: %-15s | Possibility: %.2f%%", res.name, res.score * 100));
        }

        AnalysisResult bestMatch = results.get(0);
        String foundItemName = "unknown";
        if (bestMatch.score >= MATCH_THRESHOLD) {
            foundItemName = bestMatch.name;
        }

        String reportItemName = TREASURE_ITEMS.contains(foundItemName) ? "" : foundItemName;
        int reportItemCount = (!foundItemName.equals("unknown") && !TREASURE_ITEMS.contains(foundItemName)) ? 1 : 0;

        Log.i(TAG, "Best match is '" + foundItemName + "'. Reporting to API...");
        api.setAreaInfo(areaId, reportItemName, reportItemCount);
        patrolResults.add(new PatrolResult(areaId, foundItemName));
    }

    private List<AnalysisResult> analyzeSceneWithARTag(Mat scene) {
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(scene, arucoDictionary, corners, markerIds);

        List<AnalysisResult> allPossibilities = new ArrayList<>();

        if (markerIds.total() > 0) {
            MatOfPoint2f sourceCorners = new MatOfPoint2f(corners.get(0));
            int roiSize = 100;
            Mat destinationCorners = new MatOfPoint2f(
                    new org.opencv.core.Point(0, 0),
                    new org.opencv.core.Point(roiSize, 0),
                    new org.opencv.core.Point(roiSize, roiSize),
                    new org.opencv.core.Point(0, roiSize)
            );

            Mat transform = Imgproc.getPerspectiveTransform(sourceCorners, destinationCorners);
            Mat croppedItem = new Mat();
            Imgproc.warpPerspective(scene, croppedItem, transform, new Size(roiSize, roiSize));

            api.saveMatImage(croppedItem, "cropped_item.png");

            for(int i = 0; i < templates.size(); i++) {
                Mat template = templates.get(i);
                Mat result = new Mat();
                Imgproc.matchTemplate(croppedItem, template, result, Imgproc.TM_CCOEFF_NORMED);
                Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
                allPossibilities.add(new AnalysisResult(TEMPLATE_NAMES[i], mmr.maxVal));
            }
        }
        return allPossibilities;
    }

    private boolean moveToWrapper(Point point, Quaternion quaternion) {
        int retry_count = 0;
        while (retry_count < LOOP_MAX) {
            gov.nasa.arc.astrobee.Result result = api.moveTo(point, quaternion, true);
            if (result.hasSucceeded()) {
                Log.i(TAG, "-> Move successful.");
                return true;
            }
            retry_count++;
            Log.w(TAG, "-> Move failed. Retrying... (" + retry_count + "/" + LOOP_MAX + ")");
        }
        Log.e(TAG, "-> Move failed after " + LOOP_MAX + " retries.");
        return false;
    }

    // [REMOVED] ลบเมธอด createLookAtQuaternion ออกไปแล้ว

    @Override
    protected void runPlan2() {}
    @Override
    protected void runPlan3() {}
}
