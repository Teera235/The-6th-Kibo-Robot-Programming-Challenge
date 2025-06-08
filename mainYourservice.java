package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import org.opencv.core.Mat;
import java.util.HashMap;
import java.util.Map;

public class YourService extends KiboRpcService {
    private final Point[] areaPoints = {
        new Point(10.95, -10.58, 5.195),
        new Point(10.925, -8.875, 3.76203),
        new Point(10.925, -7.925, 3.76093),
        new Point(9.866984, -6.8525, 4.945)
    };
    private final Quaternion areaQuaternion = new Quaternion(0f, 0f, -0.707f, 0.707f);

    private final Point[] oasisPoints = {
        new Point(10.925, -9.85, 4.695),
        new Point(11.175, -8.975, 5.195),
        new Point(10.7, -7.925, 5.195),
        new Point(11.175, -6.875, 4.685)
    };

    private final Map<Integer, ItemRecognitionResult> areaItems = new HashMap<>();
    private String targetItemName = null;
    private int targetArea = -1;

    @Override
    protected void runPlan1() {
        api.startMission();

        for (int i = 0; i < 4; i++) {
            if (i > 0) {
                moveThroughOasis(i - 1);
            }
            api.moveTo(areaPoints[i], areaQuaternion, false);
            Mat image = api.getMatNavCam();
            ItemRecognitionResult result = recognizeItems(image);
            areaItems.put(i + 1, result);
            api.setAreaInfo(i + 1, result.landmarkName, result.landmarkQuantity);
            if (result.treasureName != null && targetItemName == null) {
                targetItemName = result.treasureName;
                targetArea = i + 1;
            }
        }

        moveThroughOasis(3);
        Point astronautPoint = new Point(11.143, -6.7607, 4.9654);
        Quaternion astronautQuaternion = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(astronautPoint, astronautQuaternion, false);
        api.reportRoundingCompletion();

        Mat astronautImage = api.getMatNavCam();
        String astronautTarget = recognizeTargetItem(astronautImage);
        api.notifyRecognitionItem();

        if (astronautTarget != null) {
            targetItemName = astronautTarget;
            targetArea = findTargetArea(targetItemName);
        }

        if (targetArea > 0) {
            Point targetPoint = areaPoints[targetArea - 1];
            api.moveTo(targetPoint, areaQuaternion, false);
            api.takeTargetItemSnapshot();
        }
    }

    private void moveThroughOasis(int index) {
        api.moveTo(oasisPoints[index], areaQuaternion, false);
    }

    private ItemRecognitionResult recognizeItems(Mat image) {
        ItemRecognitionResult result = new ItemRecognitionResult();
        result.landmarkName = "shell";
        result.landmarkQuantity = 1;
        if (Math.random() > 0.3) {
            result.treasureName = "diamond";
        }
        return result;
    }

    private String recognizeTargetItem(Mat image) {
        return "diamond";
    }

    private int findTargetArea(String targetItemName) {
        for (Map.Entry<Integer, ItemRecognitionResult> entry : areaItems.entrySet()) {
            if (targetItemName.equals(entry.getValue().treasureName)) {
                return entry.getKey();
            }
        }
        return -1;
    }

    private static class ItemRecognitionResult {
        String landmarkName;
        int landmarkQuantity;
        String treasureName;
    }

    @Override
    protected void runPlan2() {}
    @Override
    protected void runPlan3() {}
}
