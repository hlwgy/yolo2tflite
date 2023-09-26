import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NonMaxSuppression {

    public static float iou(float[] box1, float[] box2) {
        float x1 = Math.max(box1[0], box2[0]);
        float y1 = Math.max(box1[1], box2[1]);
        float x2 = Math.min(box1[2], box2[2]);
        float y2 = Math.min(box1[3], box2[3]);

        float interArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);

        float box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
        float box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);

        return interArea / (box1Area + box2Area - interArea);
    }

    public static List<float[]> nonMaxSuppression(List<float[]> boxes, List<Float> scores, float threshold){
        List<float[]> result = new ArrayList<>();

        while (!boxes.isEmpty()) {
            // Find the index of the box with the highest score
            int bestScoreIdx = scores.indexOf(Collections.max(scores));
            float[] bestBox = boxes.get(bestScoreIdx);

            // Add the box with the highest score to the result
            result.add(bestBox);

            // Remove the box with the highest score from our lists
            boxes.remove(bestScoreIdx);
            scores.remove(bestScoreIdx);

            // Get rid of boxes with high IoU overlap
            List<float[]> newBoxes = new ArrayList<>();
            List<Float> newScores = new ArrayList<>();
            for (int i = 0; i < boxes.size(); i++) {
                if (iou(bestBox, boxes.get(i)) < threshold) {
                    newBoxes.add(boxes.get(i));
                    newScores.add(scores.get(i));
                }
            }

            boxes = newBoxes;
            scores = newScores;
        }

        return result;
    }
}
