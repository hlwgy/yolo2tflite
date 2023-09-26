import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DetectTool {

    static String TAG = "DetectTool";

    public static String detectImg(Interpreter interpreter, Bitmap bitmap){

        long startTime = System.currentTimeMillis();
        Bitmap size_bitmap = resizeBitmap(bitmap, 640);
        float[][][][] input_arr = bitmapToFloatArray(size_bitmap);
        float[][][] outArray = new float[1][6][8400];
        interpreter.run(input_arr, outArray);
        float[][] matrix_2d = outArray[0];
        float[][] outputMatrix = new float[8400][6];
        for (int i = 0; i < 8400; i++) {
            for (int j = 0; j < 6; j++) {
                outputMatrix[i][j] = matrix_2d[j][i];
            }
        }
        float threshold = 0.6f;
        float non_max = 0.8f;
        ArrayList<float[]> boxes = new ArrayList<>();
        ArrayList<Float> maxScores = new ArrayList();
        for (float[] detection : outputMatrix) {
            float[] score = Arrays.copyOfRange(detection, 4, 6);
            float maxValue = score[0];
            float maxIndex = 0;
            for(int i=1; i < score.length;i++){
                if(score[i] > maxValue){
                    maxValue = score[i];
                    maxIndex = i;
                }
            }
            if (maxValue >= threshold) {
                detection[4] = maxIndex;
                detection[5] = maxValue;
                boxes.add(detection);
                maxScores.add(maxValue);
            }
        }
        Log.d(TAG,"boxes.size(): "+boxes.size());
        List<float[]> results = NonMaxSuppression.nonMaxSuppression(boxes, maxScores, non_max);

        String strResNum = "";
        String[] names = new String[]{"1","2"};
        for ( int i=0; i<results.size(); i++){
            Log.d(TAG,"i:"+i+", result: "+Arrays.toString(results.get(i)));
            float id = results.get(i)[4];
            strResNum = strResNum + names[(int)id];
        }

        long endTime = System.currentTimeMillis();
        long timeElapsed = endTime - startTime;
        Log.d(TAG, "strResNum:"+strResNum+", Execution time: " + timeElapsed);

        return strResNum;
    }


    public static Bitmap resizeBitmap(Bitmap source, int maxSize) {
        int outWidth;
        int outHeight;
        int inWidth = source.getWidth();
        int inHeight = source.getHeight();
        if(inWidth > inHeight){
            outWidth = maxSize;
            outHeight = (inHeight * maxSize) / inWidth;
        } else {
            outHeight = maxSize;
            outWidth = (inWidth * maxSize) / inHeight;
        }

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(source, outWidth, outHeight, false);

        Bitmap outputImage = Bitmap.createBitmap(maxSize, maxSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(outputImage);
        canvas.drawColor(Color.WHITE);
        int left = (maxSize - outWidth) / 2;
        int top = (maxSize - outHeight) / 2;
        canvas.drawBitmap(resizedBitmap, left, top, null);

        return outputImage;
    }

    public static float[][][][] bitmapToFloatArray(Bitmap bitmap) {

        int height = bitmap.getHeight();
        int width = bitmap.getWidth();

        // 初始化一个float数组
        float[][][][] result = new float[1][height][width][3];

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                // 获取像素值
                int pixel = bitmap.getPixel(j, i);
                // 将RGB值分离并进行标准化（假设你需要将颜色值标准化到0-1之间）
                result[0][i][j][0] = ((pixel >> 16) & 0xFF) / 255.0f;
                result[0][i][j][1] = ((pixel >> 8) & 0xFF) / 255.0f;
                result[0][i][j][2] = (pixel & 0xFF) / 255.0f;
            }
        }
        return result;
    }

    private static MappedByteBuffer loadModelFile(Context context, String fileName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(fileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static Interpreter getInterpreter(Context context){
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        Interpreter interpreter = null;
        try {
            interpreter = new Interpreter(loadModelFile(context, "best_num_int8.tflite"), options);
        } catch (IOException e) {
            throw new RuntimeException("Error loading model file.", e);
        }
        return interpreter;
    }
}
