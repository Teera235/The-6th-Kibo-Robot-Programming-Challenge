  package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import org.apache.commons.httpclient.cookie.CookieSpec;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

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

    /* access modifiers changed from: protected */
    public void runPlan1() {
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
        Log.i(this.TAG, "[PLACEHOLDER] Moving to final target location.");
        moveToWrapper(point, quatArea1);
        Log.i(this.TAG, "Taking final snapshot.");
        this.api.takeTargetItemSnapshot();
        Log.i(this.TAG, "Mission finished.");
        this.api.shutdownFactory();
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
        Mat matNavCam = this.api.getMatNavCam();
        String str3 = this.TAG;
        Log.i(str3, "[PLACEHOLDER] In Area " + areaId + ", found " + 1 + " of " + "coin");
        this.api.setAreaInfo(areaId, "coin", 1);
        this.patrolResults.add(new PatrolResult(areaId, "coin", 1));
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
            Log.w(str, "Move failed. Retrying... (" + retry_count + CookieSpec.PATH_DELIM + 3 + ")");
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
                Imgproc.cvtColor(mat, mat, 6);
                this.templates.add(mat);
                inputStream.close();
            } catch (IOException e) {
                Log.e(this.TAG, "Error loading template: " + fileName, e);
            }
        }
        Log.i(this.TAG, "Finished loading " + this.templates.size() + " templates.");
    }

    /* access modifiers changed from: protected */
    public void runPlan2() {
    }

    /* access modifiers changed from: protected */
    public void runPlan3() {
    }
}
