package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {
    private final int LOOP_MAX = 3;
    private final String TAG = getClass().getSimpleName();
    private final String[] TEMPLATE_NAMES = {"crystal", "emerald", "treasure_box", "coin", "compass", "coral", "fossil", "key", "letter", "shell"};
    private List<PatrolResult> patrolResults = new ArrayList();
    private List<Mat> templates = new ArrayList();

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

    @Override
    protected void runPlan1() {
        // ========== Mission Start ==========
        Log.i(TAG, "========== MISSION PLAN 1 START ==========");
        api.startMission();

        // ========== Step 1: Preparation ==========
        Log.i(TAG, "==> [Step 1] Initializing mission: Loading templates...");
        loadTemplates();

        Point pointArea1 = new Point(10.9d, -9.9d, 4.8d);
        Quaternion quatArea1 = new Quaternion(0.0f, 0.707f, 0.0f, 0.707f);
        Point pointArea2 = new Point(11.0d, -8.5d, 4.5d);
        Quaternion quatArea2 = new Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
        Point pointArea3 = new Point(10.9d, -7.2d, 4.8d);
        Quaternion quatArea3 = new Quaternion(0.0f, -0.707f, 0.0f, 0.707f);
        Point pointArea4 = new Point(10.3d, -8.0d, 5.4d);
        Quaternion quatArea4 = new Quaternion(0.707f, 0.0f, -0.707f, 0.0f);

        // ========== Step 2: Patrolling ==========
        Log.i(TAG, "==> [Step 2] Starting patrol phase...");
        patrolAndRecognize(1, pointArea1, quatArea1);
        patrolAndRecognize(2, pointArea2, quatArea2);
        patrolAndRecognize(3, pointArea3, quatArea3);
        patrolAndRecognize(4, pointArea4, quatArea4);
        Log.i(TAG, "==> Patrol phase complete. All areas have been scanned.");

        // ========== Step 3: Reporting ==========
        Log.i(TAG, "==> [Step 3] Reporting completion to astronaut...");
        Point astronautPoint = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion astronautQuat = new Quaternion(0.0f, 0.0f, 0.707f, 0.707f);
        moveToWrapper(astronautPoint, astronautQuat);

        Log.i(TAG, "   -> Sending rounding completion report...");
        api.reportRoundingCompletion();
        Log.i(TAG, "   -> Report sent.");

        // ========== Step 4: Final Target ==========
        Log.i(TAG, "==> [Step 4] Proceeding to final target...");
        moveToWrapper(pointArea1, quatArea1);

        Log.i(TAG, "   -> Taking final snapshot...");
        api.takeTargetItemSnapshot();
        Log.i(TAG, "   -> Final snapshot taken.");

        // ========== Step 5: Shutdown ==========
        Log.i(TAG, "==> [Step 5] Shutting down...");
        api.shutdownFactory();
        Log.i(TAG, "========== MISSION COMPLETE ==========");
    }

    private void patrolAndRecognize(int areaId, Point patrolPoint, Quaternion patrolQuat) {
        Log.i(TAG, "-------------------------------------------------");
        Log.i(TAG, "--- Starting Area " + areaId + " ---");

        Log.i(TAG, "   -> Moving to patrol point...");
        if (!moveToWrapper(patrolPoint, patrolQuat)) {
            Log.e(TAG, "Could not move to Area " + areaId + ". Skipping.");
            this.patrolResults.add(new PatrolResult(areaId, "move_failed", 0));
            return;
        }
        Log.i(TAG, "   -> Arrived at Area " + areaId + ".");

        Log.i(TAG, "   -> Capturing image and starting recognition...");

        // [REVISED] Get the image from NavCam. It's already Grayscale (1 channel).
        // No conversion is needed.
        Mat matNavCam = this.api.getMatNavCam();

        int bestMatchIndex = -1;
        double bestMatchValue = -1.0;

        for (int i = 0; i < this.templates.size(); i++) {
            Mat template = this.templates.get(i);
            Mat result = new Mat();

            if (matNavCam.rows() < template.rows() || matNavCam.cols() < template.cols()) {
                continue;
            }

            // [REVISED] Use matNavCam directly for matching.
            Imgproc.matchTemplate(matNavCam, template, result, Imgproc.TM_CCOEFF_NORMED);
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
            double currentMatchValue = mmr.maxVal;

            if (currentMatchValue > bestMatchValue) {
                bestMatchValue = currentMatchValue;
                bestMatchIndex = i;
            }
        }

        Log.i(TAG, "   -> Recognition process finished.");

        final double MATCH_THRESHOLD = 0.2;
        String foundItemName = "unknown";
        int itemCount = 0;

        if (bestMatchValue >= MATCH_THRESHOLD) {
            foundItemName = this.TEMPLATE_NAMES[bestMatchIndex];
            itemCount = 1;
            Log.i(TAG, "   -> RESULT: Found '" + foundItemName + "' (Score: " + String.format("%.2f", bestMatchValue) + ")");
        } else {
            String bestGuess = (bestMatchIndex != -1) ? this.TEMPLATE_NAMES[bestMatchIndex] : "none";
            Log.w(TAG, "   -> RESULT: No item met threshold. Best guess was '" + bestGuess + "' (Score: " + String.format("%.2f", bestMatchValue) + ")");
        }

        Log.i(TAG, "   -> Reporting area info to API...");
        this.api.setAreaInfo(areaId, foundItemName, itemCount);
        this.patrolResults.add(new PatrolResult(areaId, foundItemName, itemCount));
        Log.i(TAG, "--- Finished Area " + areaId + " ---");
    }

    private boolean moveToWrapper(Point point, Quaternion quaternion) {
        int retry_count = 0;
        while (retry_count < LOOP_MAX) {
            if (this.api.moveTo(point, quaternion, true).hasSucceeded()) {
                Log.i(this.TAG, "   -> Move successful.");
                return true;
            }
            retry_count++;
            Log.w(this.TAG, "   -> Move failed. Retrying... (" + retry_count + "/" + LOOP_MAX + ")");
        }
        Log.e(this.TAG, "   -> Move failed after " + LOOP_MAX + " retries.");
        return false;
    }

    private void loadTemplates() {
        Log.i(this.TAG, "   -> Loading template images into memory...");
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
        Log.i(this.TAG, "   -> Finished loading " + this.templates.size() + " templates.");
    }

    @Override
    protected void runPlan2() {
        // Not implemented
    }

    @Override
    protected void runPlan3() {
        // Not implemented
    }
}
