package com.ganesh.app.iof;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.tensorflow.lite.Interpreter;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class MainActivity extends AppCompatActivity {
    static {
        // to make sure that openCV lib are loaded
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV","Library not initialised");
        }else{
            Log.e("OpenCV","Library initialised");
        }
    }

    Interpreter tflite;
    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 1;
    static final int DIM_IMG_SIZE_X = 28;
    static final int DIM_IMG_SIZE_Y = 28;
    private static final String TAG = "TfLiteDemo";
    private static final String LABEL_PATH = "labels.txt";
    private static final int RESULTS_TO_SHOW = 1;
    private List<String> labelList;
    private ByteBuffer imgData = null;
    private float[][] labelProbArray = null;
    /* Preallocated buffers for storing image data in. */
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            ImageClassifier(this);
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("ImageClassifter", e.toString());
        }

        /* get image */
        ImageView imageView = (ImageView) findViewById(R.id.imageView);
        imageView.setImageResource(R.drawable.cow);


        Bitmap bitmap2 = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
        Bitmap bitmap = getResizedBitmap(bitmap2, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
        String textToShow = classifyFrame(bitmap);
        Log.e("textToShow", textToShow);
        bitmap.recycle();


    }

    /** Initializes an {@code ImageClassifier}. */
    private void ImageClassifier(Activity activity) throws IOException {
        try{
            tflite = new Interpreter(loadModelFile());
        }catch (Exception e){
            e.printStackTrace();
            Log.d(TAG, ""+e);
        }
        imgData =ByteBuffer.allocateDirect(
                4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        labelProbArray = new float[1][labelList.size()];
        //filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }

    public String classifyFrame(Bitmap bitmap){
        String textToShow="";
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            return "Uninitialized Classifier.";
        }
        convertBitmapToByteBuffer(bitmap);
        tflite.run(imgData, labelProbArray);


        // print the results
        float max=-1;  int idx=-1;
        for(int i=0;i<labelProbArray[0].length;i++){
            // Log.e("result", Float.toString(labelProbArray[0][i]));
            if(labelProbArray[0][i]>max) {
                max = labelProbArray[0][i];  idx=i+1;
            }
        }
        textToShow = ""+labelList.get(idx-1);
        return textToShow;
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
              // imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
               // imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
    }

    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        //bm.recycle();
        return resizedBitmap;
    }



    public void inferButton (View view) {
        Toast.makeText(getApplicationContext(), "inferring",Toast.LENGTH_SHORT).show();


        //Find the directory for the SD Card using the API
        File sdcard = Environment.getExternalStorageDirectory();  //*Don't* hardcode "/sdcard"
        File directory = new File (sdcard.getAbsolutePath() + "/IOF/original");  //Get the text file
        FileInputStream streamIn = null;
        try {
            int imgNo=1;
            for (final File fileEntry : directory.listFiles()) {
                try {
                    streamIn = new FileInputStream(fileEntry);
                    Bitmap bitmap2 = BitmapFactory.decodeStream(streamIn); //This gets the image

                    // make a copy of it to save fixed image to the disk
                    Mat imgCopy = new Mat();
                    Bitmap bmp32 = bitmap2.copy(Bitmap.Config.ARGB_8888, true);
                    Utils.bitmapToMat(bmp32, imgCopy);

                    Bitmap bitmap = getResizedBitmap(bitmap2, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                    String prediction = imgNo+"-"+classifyFrame(bitmap);

                    String path = sdcard.getAbsolutePath() + "/IOF/fixed/"+prediction+".jpg";
                    Imgcodecs.imwrite(path, imgCopy);  imgNo++;
                    Log.e("fixed",prediction);
                    bitmap.recycle();
                } catch (FileNotFoundException e) {
                    Log.e("sdcard","file not found: "+e.toString());
                }
            }

            streamIn.close();
        } catch (FileNotFoundException e) {
            Log.e("sdcard","file not found: "+e.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }

        Toast.makeText(this, "done predictions",Toast.LENGTH_SHORT).show();
    }


    /* Memory-map the model file in Assets */
    private MappedByteBuffer loadModelFile() throws IOException{
        /* load labels */
        try {
            labelList = new ArrayList<String>();
            BufferedReader reader =
                    new BufferedReader(new InputStreamReader(this.getAssets().open(LABEL_PATH)));
            String line;
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
            reader.close();
        }catch (Exception e){
            Log.e("labels", ""+e);
        }

        // Open the model using an input stream, and memory map it to load
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("graph.lite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    public void changeIcon(View view){
        try {
            Mat m =  Utils.loadResource(this, R.drawable.cow2, CvType.CV_8UC4);
            Bitmap bm = Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(m, bm);

            // find the imageview and draw it!
            ImageView iv = (ImageView) findViewById(R.id.sdcard);
            iv.setImageBitmap(bm);

            File sdcard = Environment.getExternalStorageDirectory();  //*Don't* hardcode "/sdcard"
            String path = sdcard.getAbsolutePath() + "/IOF/fixed/1.jpg";
            Imgcodecs.imwrite(path, m);

        }catch(Exception e){
            System.out.println(""+e);
        }
    }

}
