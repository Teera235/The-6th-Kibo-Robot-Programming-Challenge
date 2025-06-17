package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

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
    private final double MATCH_THRESHOLD = 0.70;

    private final String[] TEMPLATE_NAMES = {"crystal", "emerald", "treasure_box", "coin", "compass", "coral", "fossil", "key", "letter", "shell", "diamond"};
    private final List<String> TREASURE_ITEMS = Arrays.asList("crystal", "emerald", "diamond");

    private List<Mat> templates = new ArrayList<>();
    private List<PatrolResult> patrolResults = new ArrayList<>();
    private Mat cameraMatrix, cameraCoefficients;
    private Dictionary arucoDictionary;

    private class PatrolResult {
        int areaId;
        List<String> foundItems;
        Point patrolPoint;
        Quaternion patrolQuat;
        PatrolResult(int id, List<String> items, Point p, Quaternion q) {
            this.areaId = id; this.foundItems = items; this.patrolPoint = p; this.patrolQuat = q;
        }
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
        executePatrolPath();
        finalizeMission();
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
                templates.add(mat);
            } catch (Exception e) { Log.e(TAG, "Error loading template: " + fileName, e); }
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

    private void executePatrolPath() {
        Log.i(TAG, "==> Starting Strategic Patrol Path...");
        patrolAndAnalyze(1, new Point(11.20, -9.85, 4.35), new Quaternion(0f, 0f, -0.707f, 0.707f));
        patrolAndAnalyze(2, new Point(10.925, -8.9, 4.4), new Quaternion(0f, 0.707f , 0f, 0.707f));
        patrolAndAnalyze(3, new Point(10.925, -7.9, 4.4), new Quaternion(0f, 0.707f , 0f, 0.707f));
        moveToWrapper(new Point(10.7, -7.6, 5.0), new Quaternion(0f, 1f , 0f, 0f));
        patrolAndAnalyze(4, new Point(11.2, -6.8, 5.0), new Quaternion(0f, 0, 0, 1f));
    }

    private void patrolAndAnalyze(int areaId, Point patrolPoint, Quaternion patrolQuat) {
        if (!moveToWrapper(patrolPoint, patrolQuat)) {
            patrolResults.add(new PatrolResult(areaId, new ArrayList<String>(), patrolPoint, patrolQuat));
            return;
        }
        try { Thread.sleep(500); } catch (InterruptedException e) {}

        Mat scene = api.getMatNavCam();
        if (scene == null || scene.empty()) {
            patrolResults.add(new PatrolResult(areaId, new ArrayList<String>(), patrolPoint, patrolQuat));
            return;
        }

        Mat undistortedScene = new Mat();
        Calib3d.undistort(scene, undistortedScene, cameraMatrix, cameraCoefficients);
        api.saveMatImage(undistortedScene, "Area" + areaId + "_Full_Capture.png");

        List<AnalysisResult> results = analyzeSceneWithARTag(undistortedScene, areaId);
        if (results.isEmpty()) {
            results = analyzeFullScene_Fallback(undistortedScene);
        }

        if (results.isEmpty()) {
            api.setAreaInfo(areaId, "", 0);
            patrolResults.add(new PatrolResult(areaId, new ArrayList<String>(), patrolPoint, patrolQuat));
            return;
        }

        Collections.sort(results, new Comparator<AnalysisResult>() {
            @Override
            public int compare(AnalysisResult r1, AnalysisResult r2) {
                return Double.compare(r2.score, r1.score);
            }
        });

        List<String> foundItemsInArea = new ArrayList<>();
        String bestGuessForReport = "unknown";
        for (AnalysisResult res : results) {
            if (res.score >= MATCH_THRESHOLD) foundItemsInArea.add(res.name);
        }
        if (!foundItemsInArea.isEmpty()) bestGuessForReport = results.get(0).name;

        String reportItemName = TREASURE_ITEMS.contains(bestGuessForReport) ? "" : bestGuessForReport;
        int reportItemCount = (!bestGuessForReport.equals("unknown") && !TREASURE_ITEMS.contains(bestGuessForReport)) ? 1 : 0;

        api.setAreaInfo(areaId, reportItemName, reportItemCount);
        patrolResults.add(new PatrolResult(areaId, foundItemsInArea, patrolPoint, patrolQuat));
    }

    private void finalizeMission() {
        Log.i(TAG, "==> Finalizing Mission...");
        moveToWrapper(new Point(11.143, -6.7607, 4.80), new Quaternion(0f, 0f, 0.707f, 0.707f));
        api.reportRoundingCompletion();
        try { Thread.sleep(1000); } catch (InterruptedException e) {}

        Mat clueScene = api.getMatNavCam();
        if (clueScene == null) { return; }
        api.saveMatImage(clueScene, "Clue_Capture.png");

        List<AnalysisResult> clueResults = analyzeSceneWithARTag(clueScene, 0); // areaId 0 for clue
        if (clueResults.isEmpty()) clueResults = analyzeFullScene_Fallback(clueScene);

        String clueTreasure = null;
        List<String> clueLandmarks = new ArrayList<>();
        if (!clueResults.isEmpty()){
            Collections.sort(clueResults, new Comparator<AnalysisResult>() {
                @Override public int compare(AnalysisResult r1, AnalysisResult r2) { return Double.compare(r2.score, r1.score); }
            });
            for (AnalysisResult res : clueResults) {
                if (res.score > MATCH_THRESHOLD) {
                    if (TREASURE_ITEMS.contains(res.name) && clueTreasure == null) clueTreasure = res.name;
                    else if (!TREASURE_ITEMS.contains(res.name)) clueLandmarks.add(res.name);
                }
            }
        }

        if (clueTreasure == null || clueLandmarks.isEmpty()) { return; }

        PatrolResult finalTargetArea = null;
        for (PatrolResult patrol : patrolResults) {
            boolean hasTreasure = patrol.foundItems.contains(clueTreasure);
            boolean hasLandmark = false;
            for (String landmark : clueLandmarks) {
                if (patrol.foundItems.contains(landmark)) { hasLandmark = true; break; }
            }
            if (hasTreasure && hasLandmark) {
                finalTargetArea = patrol;
                break;
            }
        }

        if (finalTargetArea != null) {
            moveToWrapper(finalTargetArea.patrolPoint, finalTargetArea.patrolQuat);
            api.takeTargetItemSnapshot();
        } else {
            api.takeTargetItemSnapshot();
        }
    }

    private List<AnalysisResult> analyzeSceneWithARTag(Mat scene, int areaId) {
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(scene, arucoDictionary, corners, markerIds);
        List<AnalysisResult> allPossibilities = new ArrayList<>();
        if (markerIds.total() > 0) {
            MatOfPoint2f sourceCorners = new MatOfPoint2f(corners.get(0));
            int roiSize = 100;
            Mat destCorners = new MatOfPoint2f(
                    new org.opencv.core.Point(0, 0), new org.opencv.core.Point(roiSize, 0),
                    new org.opencv.core.Point(roiSize, roiSize), new org.opencv.core.Point(0, roiSize)
            );
            Mat transform = Imgproc.getPerspectiveTransform(sourceCorners, destCorners);
            Mat croppedItem = new Mat();
            Imgproc.warpPerspective(scene, croppedItem, transform, new Size(roiSize, roiSize));

            api.saveMatImage(croppedItem, "Area" + areaId + "_Cropped_Item.png");

            for (int i = 0; i < templates.size(); i++) {
                Mat template = new Mat();
                Imgproc.resize(templates.get(i), template, new Size(100,100));
                Mat result = new Mat();
                Imgproc.matchTemplate(croppedItem, template, result, Imgproc.TM_CCOEFF_NORMED);
                allPossibilities.add(new AnalysisResult(TEMPLATE_NAMES[i], Core.minMaxLoc(result).maxVal));
            }
        }
        return allPossibilities;
    }

    private List<AnalysisResult> analyzeFullScene_Fallback(Mat scene) {
        List<AnalysisResult> allPossibilities = new ArrayList<>();
        List<Double> scales = Arrays.asList(0.4, 0.6, 0.8);
        List<Double> angles = Arrays.asList(0.0, -10.0, 10.0);
        for (int i = 0; i < templates.size(); i++) {
            Mat template = templates.get(i);
            double maxValForThisTemplate = -1.0;
            for (double scale : scales) {
                Mat scaledTemplate = new Mat();
                Imgproc.resize(template, scaledTemplate, new Size(), scale, scale);
                for (double angle : angles) {
                    Mat rotatedTemplate = rotateTemplate(scaledTemplate, angle);
                    if (scene.rows() < rotatedTemplate.rows() || scene.cols() < rotatedTemplate.cols()) continue;
                    Mat result = new Mat();
                    Imgproc.matchTemplate(scene, rotatedTemplate, result, Imgproc.TM_CCOEFF_NORMED);
                    Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
                    if (mmr.maxVal > maxValForThisTemplate) maxValForThisTemplate = mmr.maxVal;
                }
            }
            allPossibilities.add(new AnalysisResult(TEMPLATE_NAMES[i], maxValForThisTemplate));
        }
        return allPossibilities;
    }

    private Mat rotateTemplate(Mat source, double angle) {
        if (angle == 0.0) return source;
        org.opencv.core.Point center = new org.opencv.core.Point(source.cols() / 2.0, source.rows() / 2.0);
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Rect bbox = new RotatedRect(new org.opencv.core.Point(0, 0), source.size(), angle).boundingRect();
        rotationMatrix.put(0, 2, rotationMatrix.get(0, 2)[0] + (bbox.width / 2.0) - center.x);
        rotationMatrix.put(1, 2, rotationMatrix.get(1, 2)[0] + (bbox.height / 2.0) - center.y);
        Mat rotated = new Mat();
        Imgproc.warpAffine(source, rotated, rotationMatrix, bbox.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));
        return rotated;
    }

    private boolean moveToWrapper(Point point, Quaternion quaternion) {
        int retry_count = 0;
        while (retry_count < LOOP_MAX) {
            if (api.moveTo(point, quaternion, true).hasSucceeded()) return true;
            retry_count++;
            Log.w(TAG, "-> Move failed. Retrying...");
        }
        Log.e(TAG, "-> Move failed after " + LOOP_MAX + " retries.");
        return false;
    }

    private Quaternion createLookAtQuaternion(Point pos, Point target, Point up) {
        Point dir = new Point(target.getX() - pos.getX(), target.getY() - pos.getY(), target.getZ() - pos.getZ());
        double norm = Math.sqrt(Math.pow(dir.getX(), 2) + Math.pow(dir.getY(), 2) + Math.pow(dir.getZ(), 2));
        float zx = (float) (dir.getX() / norm), zy = (float) (dir.getY() / norm), zz = (float) (dir.getZ() / norm);
        float xx = (float) (up.getY() * zz - up.getZ() * zy), xy = (float) (up.getZ() * zx - up.getX() * zz), xz = (float) (up.getX() * zy - up.getY() * zx);
        norm = Math.sqrt(xx * xx + xy * xy + xz * xz);
        xx /= norm; xy /= norm; xz /= norm;
        float yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
        float tr = xx + yy + zz;
        float qw, qx, qy, qz;
        if (tr > 0) {
            float S = (float) (Math.sqrt(tr + 1.0) * 2);
            qw = 0.25f * S; qx = (yz - zy) / S; qy = (zx - xz) / S; qz = (xy - yx) / S;
        } else if ((xx > yy) & (xx > zz)) {
            float S = (float) (Math.sqrt(1.0 + xx - yy - zz) * 2);
            qw = (yz - zy) / S; qx = 0.25f * S; qy = (yx + xy) / S; qz = (zx + xz) / S;
        } else if (yy > zz) {
            float S = (float) (Math.sqrt(1.0 + yy - xx - zz) * 2);
            qw = (zx - xz) / S; qx = (yx + xy) / S; qy = 0.25f * S; qz = (zy + yz) / S;
        } else {
            float S = (float) (Math.sqrt(1.0 + zz - xx - yy) * 2);
            qw = (xy - yx) / S; qx = (zx + xz) / S; qy = (zy + yz) / S; qz = 0.25f * S;
        }
        return new Quaternion(qx, qy, qz, qw);
    }

    @Override
    protected void runPlan2() {}
    @Override
    protected void runPlan3() {}
}
