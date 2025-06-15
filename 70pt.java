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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

public class YourService extends KiboRpcService {
    // Constants
    private final int LOOP_MAX = 3;
    private final String TAG = getClass().getSimpleName();
    private final double TEMPLATE_MATCHING_THRESHOLD = 0.8; // Threshold for matching

    // List of item names corresponding to the template images
    private final String[] TEMPLATE_NAMES = {
            "crystal", "emerald", "treasure_box", "coin", "compass",
            "coral", "fossil", "key", "letter", "shell"
    };

    // Data storage for mission results
    private List<PatrolResult> patrolResults = new ArrayList<>();
    private List<Mat> templates = new ArrayList<>();
    private Map<Integer, Point> areaPoints = new HashMap<>();
    private Map<Integer, Quaternion> areaQuats = new HashMap<>();


    // Inner class to store patrol results for each area
    private class PatrolResult {
        int areaId;
        // Stores found items and their counts, e.g., {"coin": 2, "shell": 1}
        Map<String, Integer> foundItems;

        PatrolResult(int areaId, Map<String, Integer> foundItems) {
            this.areaId = areaId;
            this.foundItems = new HashMap<>(foundItems);
        }

        @Override
        public String toString() {
            return "Area " + areaId + ": " + foundItems.toString();
        }
    }

    @Override
    protected void runPlan1() {
        Log.i(TAG, "START MISSION");
        api.startMission();

        // Load item templates for image recognition
        loadTemplates();

        // Define patrol points and orientations for each area
        initializePatrolPoints();

        // Patrol each area and perform image recognition
        patrolAndRecognize(1);
        patrolAndRecognize(2);
        patrolAndRecognize(3);
        patrolAndRecognize(4);

        Log.i(TAG, "Patrol complete. All areas have been scanned.");
        for (PatrolResult result : patrolResults) {
            Log.i(TAG, "Result -> " + result.toString());
        }

        // Move to astronaut to report completion
        Log.i(TAG, "Moving to astronaut to report completion.");
        Point astronautPoint = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion astronautQuat = new Quaternion(0.0f, 0.0f, 0.707f, 0.707f);
        moveToWrapper(astronautPoint, astronautQuat);
        api.reportRoundingCompletion();

        // =========================================================================
        // LOGIC TO FIND THE FINAL TARGET
        // In a real run, you would take a picture of the astronaut's clue here.
        // For this example, we will simulate the result of that recognition.
        // =========================================================================
        Log.i(TAG, "Analyzing clues from astronaut to find the final target.");
        String targetTreasure = "emerald"; // Example: The treasure we're looking for
        String clueLandmark = "coral";   // Example: The landmark that was with the treasure

        int finalAreaId = -1;
        // Search through our patrol results
        for (PatrolResult result : patrolResults) {
            // Check if this area contains BOTH the target treasure AND the clue landmark
            if (result.foundItems.containsKey(targetTreasure) && result.foundItems.containsKey(clueLandmark)) {
                finalAreaId = result.areaId;
                Log.i(TAG, "Target location identified! Area: " + finalAreaId);
                break;
            }
        }

        if (finalAreaId != -1) {
            // Move to the identified final target location
            Log.i(TAG, "Moving to final target location in Area " + finalAreaId);
            Point finalPoint = areaPoints.get(finalAreaId);
            Quaternion finalQuat = areaQuats.get(finalAreaId);
            moveToWrapper(finalPoint, finalQuat);

            // Take final snapshot
            Log.i(TAG, "Taking final snapshot of the target item.");
            api.takeTargetItemSnapshot();
        } else {
            Log.e(TAG, "Could not identify the final target area based on clues. Mission might fail.");
            // As a fallback, maybe move to a default location, though this is not ideal.
        }

        Log.i(TAG, "MISSION FINISHED");
        api.shutdownFactory();
    }

    /**
     * Initializes the patrol points and quaternions for all areas.
     */
    private void initializePatrolPoints() {
        areaPoints.put(1, new Point(10.9d, -9.9d, 4.8d));
        areaQuats.put(1, new Quaternion(0.0f, 0.707f, 0.0f, 0.707f));

        areaPoints.put(2, new Point(11.0d, -8.5d, 4.5d));
        areaQuats.put(2, new Quaternion(0.0f, 0.0f, 0.0f, 1.0f));

        areaPoints.put(3, new Point(10.9d, -7.2d, 4.8d));
        areaQuats.put(3, new Quaternion(0.0f, -0.707f, 0.0f, 0.707f));

        areaPoints.put(4, new Point(10.3d, -8.0d, 5.4d));
        areaQuats.put(4, new Quaternion(0.707f, 0.0f, -0.707f, 0.0f));
    }


    /**
     * Moves to a patrol area, captures an image, and recognizes items.
     * @param areaId The ID of the area to process (1, 2, 3, or 4).
     */
    private void patrolAndRecognize(int areaId) {
        Log.i(TAG, "Processing Area " + areaId);
        Point patrolPoint = areaPoints.get(areaId);
        Quaternion patrolQuat = areaQuats.get(areaId);

        if (patrolPoint == null || patrolQuat == null) {
            Log.e(TAG, "Coordinates for Area " + areaId + " not found. Skipping.");
            patrolResults.add(new PatrolResult(areaId, new HashMap<String, Integer>()));
            return;
        }

        // Move to the designated patrol point
        if (!moveToWrapper(patrolPoint, patrolQuat)) {
            Log.e(TAG, "Could not move to Area " + areaId + ". Skipping.");
            patrolResults.add(new PatrolResult(areaId, new HashMap<String, Integer>())); // Add empty result
            return;
        }

        // Get the camera image
        Mat sceneImage = api.getMatNavCam();
        if (sceneImage == null) {
            Log.e(TAG, "Failed to get NavCam image for Area " + areaId);
            patrolResults.add(new PatrolResult(areaId, new HashMap<String, Integer>()));
            return;
        }
        // Convert the scene to grayscale for template matching
        Imgproc.cvtColor(sceneImage, sceneImage, Imgproc.COLOR_BGR2GRAY);

        Map<String, Integer> itemsFoundInArea = new HashMap<>();

        // Iterate through each loaded template to find matches
        for (int i = 0; i < templates.size(); i++) {
            Mat template = templates.get(i);
            String itemName = TEMPLATE_NAMES[i];

            // Create a result matrix
            int result_cols = sceneImage.cols() - template.cols() + 1;
            int result_rows = sceneImage.rows() - template.rows() + 1;
            Mat result = new Mat(result_rows, result_cols, 32); // CV_32F

            // Perform template matching
            Imgproc.matchTemplate(sceneImage, template, result, Imgproc.TM_CCOEFF_NORMED);

            // Find the best match
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
            double maxVal = mmr.maxVal;

            Log.d(TAG, "Area " + areaId + " - Checking for '" + itemName + "'. Match score: " + maxVal);

            // If the match is strong enough, consider the item found
            if (maxVal >= TEMPLATE_MATCHING_THRESHOLD) {
                // To avoid counting the same item multiple times, a simple approach is to "erase" the found area
                // and re-run the matching. A more advanced method is non-maximum suppression.
                // For now, we'll just count it once if the score is high.
                // This simple version might over-count if multiple instances are close.
                itemsFoundInArea.put(itemName, itemsFoundInArea.getOrDefault(itemName, 0) + 1);
            }
        }

        // Report and store the results for this area
        if (itemsFoundInArea.isEmpty()) {
            Log.i(TAG, "In Area " + areaId + ", found no items.");
        } else {
            for (Map.Entry<String, Integer> entry : itemsFoundInArea.entrySet()) {
                Log.i(TAG, "In Area " + areaId + ", found " + entry.getValue() + " of " + entry.getKey());
                // The rules may ask to report only one type of item (e.g., the most prominent).
                // Here we report all found landmark items.
                api.setAreaInfo(areaId, entry.getKey(), entry.getValue());
            }
        }
        patrolResults.add(new PatrolResult(areaId, itemsFoundInArea));
    }


    /**
     * Wrapper for the moveTo API call to add a retry mechanism.
     * @param point The target point.
     * @param quaternion The target orientation.
     * @return True if the move was successful, false otherwise.
     */
    private boolean moveToWrapper(Point point, Quaternion quaternion) {
        int retry_count = 0;
        while (retry_count < LOOP_MAX) {
            // The last argument 'true' prints robot position to the log
            if (api.moveTo(point, quaternion, true).hasSucceeded()) {
                Log.i(TAG, "Move successful.");
                return true;
            }
            retry_count++;
            Log.w(TAG, "Move failed. Retrying... (" + retry_count + "/" + LOOP_MAX + ")");
        }
        Log.e(TAG, "Move failed after " + LOOP_MAX + " retries.");
        return false;
    }

    /**
     * Loads all template images from the assets folder into memory.
     */
    private void loadTemplates() {
        Log.i(TAG, "Loading template images into memory.");
        for (String fileName : TEMPLATE_NAMES) {
            try {
                InputStream inputStream = getAssets().open(fileName + ".png");
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                Mat mat = new Mat();
                Utils.bitmapToMat(bitmap, mat);
                // Convert template to grayscale, same as the scene will be
                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
                this.templates.add(mat);
                inputStream.close();
            } catch (IOException e) {
                Log.e(TAG, "Error loading template: " + fileName, e);
            }
        }
        Log.i(TAG, "Finished loading " + this.templates.size() + " templates.");
    }

    @Override
    protected void runPlan2() {
        // Not used in this mission
    }

    @Override
    protected void runPlan3() {
        // Not used in this mission
    }
}
