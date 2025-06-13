package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {
    private final int LOOP_MAX = 3;
    private final String TAG = getClass().getSimpleName();
    private final String[] TEMPLATE_FILE_NAMES = {"coin.png", "compass.png", "coral.png", "crystal.png", "emerald.png", "fossil.png", "key.png", "letter.png", "shell.png", "treasure_box.png"};
    private final String[] TEMPLATE_NAMES = {"coin", "compass", "coral", "crystal", "emerald", "fossil", "key", "letter", "shell", "treasure_box"};
    private List<PatrolResult> patrolResults = new ArrayList();
    private Mat[] templates;

    private class PatrolResult {
        int areaId;
        int itemCount;
        String itemName;

        PatrolResult(int areaId2, String itemName2, int itemCount2) {
            this.areaId = areaId2;
            this.itemName = itemName2;
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
        executePatrolForArea(1, point, quatArea1);
        executePatrolForArea(2, pointArea2, quatArea2);
        executePatrolForArea(3, pointArea3, quatArea3);
        executePatrolForArea(4, new Point(10.3d, -8.0d, 5.4d), new Quaternion(0.707f, 0.0f, -0.707f, 0.0f));
        Log.i(this.TAG, "Patrol complete. All areas have been scanned.");
        moveToWrapper(new Point(11.143d, -6.7607d, 4.9654d), new Quaternion(0.0f, 0.0f, 0.707f, 0.707f));
        this.api.reportRoundingCompletion();
        this.api.takeTargetItemSnapshot();
        this.api.shutdownFactory();
    }

    private void executePatrolForArea(int areaId, Point point, Quaternion quat) {
        if (!moveToWrapper(point, quat)) {
            String str = this.TAG;
            Log.e(str, "Skipping Area " + areaId + " due to movement failure.");
            return;
        }
        PatrolResult result = recognizeItemsInFrame(areaId, this.api.getMatNavCam());
        this.patrolResults.add(result);
        this.api.setAreaInfo(result.areaId, result.itemName, result.itemCount);
        String str2 = this.TAG;
        Log.i(str2, "Area " + result.areaId + ": Found " + result.itemCount + " of " + result.itemName);
    }

    private PatrolResult recognizeItemsInFrame(int areaId, Mat image) {
        Dictionary dictionary;
        Mat mat = image;
        Dictionary dictionary2 = Aruco.getPredefinedDictionary(6);
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(mat, dictionary2, corners, markerIds);
        Mat cameraMatrix = new Mat(3, 3, 6);
        cameraMatrix.put(0, 0, this.api.getNavCamIntrinsics()[0]);
        Mat cameraCoefficients = new Mat(1, 5, 6);
        cameraCoefficients.put(0, 0, this.api.getNavCamIntrinsics()[1]);
        Mat undistortImg = new Mat();
        Calib3d.undistort(mat, undistortImg, cameraMatrix, cameraCoefficients);
        int[] templateMatchCnt = new int[this.templates.length];
        int tempNum = 0;
        while (tempNum < this.templates.length) {
            List<org.opencv.core.Point> matches = new ArrayList<>();
            Mat template = this.templates[tempNum].clone();
            Mat targetImg = undistortImg.clone();
            int size = 20;
            while (size <= 100) {
                Mat cameraCoefficients2 = cameraCoefficients;
                int angle = 0;
                while (true) {
                    dictionary = dictionary2;
                    if (angle >= 360) {
                        break;
                    }
                    Mat resizedTemp = resizeImg(template, size);
                    List<Mat> corners2 = corners;
                    Mat rotResizedTemp = rotImg(resizedTemp, angle);
                    Mat mat2 = resizedTemp;
                    Mat markerIds2 = markerIds;
                    Mat result = new Mat();
                    Imgproc.matchTemplate(targetImg, rotResizedTemp, result, 5);
                    Core.MinMaxLocResult mmlr = Core.minMaxLoc(result);
                    Mat mat3 = result;
                    Mat mat4 = rotResizedTemp;
                    if (mmlr.maxVal >= 0.7d) {
                        matches.add(mmlr.maxLoc);
                    }
                    angle += 45;
                    markerIds = markerIds2;
                    dictionary2 = dictionary;
                    corners = corners2;
                }
                Mat mat5 = markerIds;
                size += 5;
                dictionary2 = dictionary;
                cameraCoefficients = cameraCoefficients2;
            }
            Dictionary dictionary3 = dictionary2;
            List<Mat> list = corners;
            Mat mat6 = markerIds;
            templateMatchCnt[tempNum] = 0 + removeDuplicates(matches).size();
            tempNum++;
            Mat mat7 = image;
            cameraCoefficients = cameraCoefficients;
        }
        Dictionary dictionary4 = dictionary2;
        List<Mat> list2 = corners;
        Mat mat8 = markerIds;
        int mostMatchTemplateNum = getMaxIndex(templateMatchCnt);
        return new PatrolResult(areaId, this.TEMPLATE_NAMES[mostMatchTemplateNum], templateMatchCnt[mostMatchTemplateNum]);
    }

    private boolean moveToWrapper(Point point, Quaternion quaternion) {
        for (int retry_count = 0; retry_count < 3; retry_count++) {
            if (this.api.moveTo(point, quaternion, true).hasSucceeded()) {
                return true;
            }
        }
        return false;
    }

    private void loadTemplates() {
        this.templates = new Mat[this.TEMPLATE_FILE_NAMES.length];
        for (int i = 0; i < this.TEMPLATE_FILE_NAMES.length; i++) {
            try {
                InputStream is = getAssets().open(this.TEMPLATE_FILE_NAMES[i]);
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                Mat mat = new Mat();
                Utils.bitmapToMat(bitmap, mat);
                Imgproc.cvtColor(mat, mat, 6);
                this.templates[i] = mat;
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private Mat resizeImg(Mat img, int width) {
        Mat resizedImg = new Mat();
        Imgproc.resize(img, resizedImg, new Size((double) width, (double) ((int) (((double) img.rows()) * (((double) width) / ((double) img.cols()))))));
        return resizedImg;
    }

    private Mat rotImg(Mat img, int angle) {
        Mat rotatedMat = Imgproc.getRotationMatrix2D(new org.opencv.core.Point(((double) img.cols()) / 2.0d, ((double) img.rows()) / 2.0d), (double) angle, 1.0d);
        Mat rotatedImg = new Mat();
        Imgproc.warpAffine(img, rotatedImg, rotatedMat, img.size());
        return rotatedImg;
    }

    private List<org.opencv.core.Point> removeDuplicates(List<org.opencv.core.Point> points) {
        List<org.opencv.core.Point> filteredList = new ArrayList<>();
        for (org.opencv.core.Point point : points) {
            boolean isInclude = false;
            Iterator<org.opencv.core.Point> it = filteredList.iterator();
            while (true) {
                if (it.hasNext()) {
                    if (calculateDistance(point, it.next()) <= 10.0d) {
                        isInclude = true;
                        break;
                    }
                } else {
                    break;
                }
            }
            if (!isInclude) {
                filteredList.add(point);
            }
        }
        return filteredList;
    }

    private double calculateDistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2.0d) + Math.pow(p1.y - p2.y, 2.0d));
    }

    private int getMaxIndex(int[] array) {
        int max = 0;
        int maxIndex = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /* access modifiers changed from: protected */
    public void runPlan2() {
    }

    /* access modifiers changed from: protected */
    public void runPlan3() {
    }
}
