package com.ganesh.app.cv2;

import android.graphics.Bitmap;
import android.os.Bundle;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import android.app.Activity;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

public class MainActivity extends Activity{
    public static String msg = "";
    static {
        // to make sure that openCV lib are loaded
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
            msg="error, openCV lib not loaded";
        }else{
            msg="lib loaded";
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView tx = (TextView)findViewById(R.id.textView);
        tx.setText("msg: "+msg);

    }

    public void changeIcon(View view){
        try {
            Mat m =  Utils.loadResource(this, R.drawable.icon, CvType.CV_8UC4);            //Mat.zeros(100, 400, CvType.CV_8UC3);

            //Core.putText(m, "hello world", new Point(30,80), Core.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.2, new Scalar(200,200,0),2);

            Bitmap bm = Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(m, bm);

            // find the imageview and draw it!
            ImageView iv = (ImageView) findViewById(R.id.img1);
            iv.setImageBitmap(bm);

            TextView tx = (TextView)findViewById(R.id.textView);
            tx.setText("msg: tried");
        }catch(Exception e){
            TextView tx = (TextView)findViewById(R.id.textView);
            tx.setText("exception: "+e);

            ImageView iv = (ImageView) findViewById(R.id.img1);
            iv.setImageResource(R.drawable.icon);
        }


    }

}
