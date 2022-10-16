package com.example.facedetect;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Camera;
import android.graphics.Color;
import android.graphics.Point;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Display;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import com.example.facedetect.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'myapplication' library on application startup.
    static {
        System.loadLibrary("facedetect");
    }

    private ActivityMainBinding binding;
    private ImageAnalysis imageAnalysis = null;

    private Preview preview;
    private PreviewView previewView;
    private Button btnIn;
    private Button btnOut;
    private Button btnCancel;
    private ImageView imageView;

    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private final String TAG = "FaceDetect";
    private final String dataPath = "/storage/emulated/0/Pictures";
    private final String featureData = "Feature.yml";
    private final String nameData = "Feature_Name.dat";
    private final String yNFileName = "yunet.onnx";
    private final String sFFileName = "face_recognizer_fast.onnx";

    private AsyncTask<SendPostTaskParams, String, String> mAsyncTask ;

    private int featureListSize = 0;
    private Bitmap bitmap = null;
    private Mat mat = null;
    private Mat matOrg = null;
    private Mat tmpMat = null;
    private boolean btnDisplay = false;
    private boolean faceDetect = true;

    // 社員番号
    private String employeeNo = "";
    // 社員名
    private String employeeName = "";

    private final int REQUEST_CODE_FOR_PERMISSIONS = 1234;
    private final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE};

    private final Handler mHandler = new Handler();
    private Runnable timerEnd;

    public void TimerStart() {
        btnDisplay = false;
        timerEnd = () -> {
            mHandler.removeCallbacks(timerEnd);
            // タイマ満了でボタン非表示
            buttonDisplay(false);
            faceDetect = true;
        };
        mHandler.postDelayed(timerEnd, 2000);
    }

    private final Handler mRestartTimerHandler = new Handler();
    private Runnable restartTimerEnd;

    public void restartTimer() {
        restartTimerEnd = () -> {
            mRestartTimerHandler.removeCallbacks(restartTimerEnd);
            faceDetect = true;
        };
        mRestartTimerHandler.postDelayed(restartTimerEnd, 5000);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.viewFinder);
        imageView = findViewById(R.id.imageView);
        btnIn = findViewById(R.id.btnIn);
        btnOut = findViewById(R.id.btnOut);
        btnCancel = findViewById(R.id.btnCancel);

        // ボタン表示
        buttonDisplay(false);

        findViewById(R.id.btnIn).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 出勤ボタン押下
                btnInAction();
            }
        });
        findViewById(R.id.btnOut).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // 退勤ボタン押下
                btnOutAction();
            }
        });
        findViewById(R.id.btnCancel).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // キャンセルボタン押下
                btnCancelAction();
            }
        });

        featureListSize = featureOpen(
                openStorageFile(nameData).getAbsolutePath(),
                openStorageFile(featureData).getAbsolutePath()
        );

        if (checkPermissions()) {
            // startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_FOR_PERMISSIONS);
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);

        startCamera();
    }

    private boolean checkPermissions() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void startCamera() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        Context context = this;

        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    // Used to bind the lifecycle of cameras to the lifecycle owner
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                    // Preview
                    preview = new Preview.Builder()
                            //.setTargetAspectRatio(AspectRatio.RATIO_16_9)
                            //.setTargetResolution(new Size(previewView.getWidth(), previewView.getHeight()))
                            .setTargetRotation(Surface.ROTATION_0)
                            .build();
                    preview.setSurfaceProvider(previewView.getSurfaceProvider());

                    // Select back camera as a default
                    CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();

                    imageAnalysis = new ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            // .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                            .setTargetRotation(Surface.ROTATION_0)
                            //.setTargetResolution(new Size(previewView.getWidth(), previewView.getHeight()))
                            .build();

                    imageAnalysis.setAnalyzer(cameraExecutor, new MyImageAnalyzer());

                    // Unbind use cases before rebinding
                    cameraProvider.unbindAll();
                    //camera = (Camera) cameraProvider.bindToLifecycle((LifecycleOwner) context, cameraSelector, preview, imageAnalysis);
                    cameraProvider.bindToLifecycle((LifecycleOwner) context, cameraSelector, preview, imageAnalysis);
                } catch (Exception e) {
                    Log.e(TAG, "[startCamera] Use case binding failed", e);
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 0: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.i("permission", "permitted");
                } else {
                    Log.i("permission", "not permitted");
                }
                break;
            }
        }
    }

    private class MyImageAnalyzer implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(@NonNull ImageProxy image) {

            tmpMat = getMatFromImage(image);
            mat = fixMatRotation(tmpMat);
            //bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(),Bitmap.Config.ARGB_8888);
            bitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
            matOrg = new Mat();

            matOrg = mat;
            //Utils.bitmapToMat(bitmap, matOrg);

            //Log.i(TAG, "getRotationDegrees:"+String.valueOf(image.getImageInfo().getRotationDegrees()));
            //Log.i(TAG, "[analyze] width = " + image.getWidth() + ", height = " + image.getHeight() + "Rotation = " + previewView.getDisplay().getRotation());
            // Log.i(TAG, "[analyze] mat width = " + matOrg.cols() + ", mat height = " + matOrg.rows());

            detectFace();

            Bitmap dst = Bitmap.createBitmap(matOrg.cols(), matOrg.rows(), Bitmap.Config.ARGB_8888);
            if (matOrg.cols() > 0) {
                //Log.i(TAG, "matOrg.cols() > 0");
                Utils.matToBitmap(matOrg, dst);
            }
            /* Display the result onto ImageView */
            runOnUiThread(new Runnable() {
                @SuppressLint("SuspiciousIndentation")
                @Override
                public void run() {

                    if (View.VISIBLE == btnIn.getVisibility())
                        imageView.setImageBitmap((dst));
                        //imageView.setImageBitmap(transparent(dst));

                }
            });
            /* Close the image otherwise, this function is not called next time */
            image.close();
        }

        private Mat getMatFromImage(ImageProxy image) {
            /* https://stackoverflow.com/questions/30510928/convert-android-camera2-api-yuv-420-888-to-rgb */
            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
            ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);
            //Mat yuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
            Mat yuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CvType.CV_8UC1);
            yuv.put(0, 0, nv21);
            Mat mat = new Mat();
            Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2RGB_NV21, 3);
            return mat;
        }


        private Mat fixMatRotation(Mat matOrg) {
            Mat mat;
            switch (previewView.getDisplay().getRotation()) {
                default:
                case Surface.ROTATION_0:
                    mat = new Mat(matOrg.cols(), matOrg.rows(), matOrg.type());
                    Core.transpose(matOrg, mat);
                    Core.flip(mat, mat, -1);
                    break;
                case Surface.ROTATION_90:
                case Surface.ROTATION_270:
                    mat = matOrg;
                    Core.flip(mat, mat, 1);
                    break;
            }
            return mat;

        }
    }

    private File openAssetFile(String fileName)
    {
        File file = new File(getFilesDir().getPath() + File.separator + fileName);
        //if (!file.exists())
        {
            try (InputStream inputStream = getAssets().open(fileName);
                 FileOutputStream fileOutputStream = new FileOutputStream(file, false)) {
                byte[] buffer = new byte[1024];
                int read;
                while ((read = inputStream.read(buffer)) != -1) {
                    fileOutputStream.write(buffer, 0, read);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return file;
    }

    private File openStorageFile(String fileName)
    {
        File file = new File(getFilesDir().getPath() + File.separator + fileName);
        //if (!file.exists())
        {
            try (FileInputStream inputStream = new FileInputStream(new File(dataPath + "/" + fileName));
                 FileOutputStream fileOutputStream = new FileOutputStream(file, false)) {
                byte[] buffer = new byte[1024];
                int read;
                while ((read = inputStream.read(buffer)) != -1) {
                    fileOutputStream.write(buffer, 0, read);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return file;
    }

    private void detectFace() {

        if (0 == featureListSize) {
            featureListSize = featureOpen(
                    openStorageFile(nameData).getAbsolutePath(),
                    openStorageFile(featureData).getAbsolutePath()
            );
        }

        String a = featureAnalyzer(
                mat.getNativeObjAddr(),
                matOrg.getNativeObjAddr(),
                openAssetFile(yNFileName).getAbsolutePath(),
                openAssetFile(sFFileName).getAbsolutePath(),
                imageView.getHeight(),
                imageView.getWidth(),
                previewView.getDisplay().getRotation()
        );

        List<String> tokens = Arrays.asList(a.split("\\s*,\\s*"));

        //Log.i(TAG, "source:" + String.valueOf(a));

    if (2 == tokens.size() && true == faceDetect)
    {
        faceDetect = false;
        employeeName = tokens.get(1);
        employeeNo = tokens.get(0);

            if (false == btnDisplay)
            {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        // ボタン表示
                        buttonDisplay(true);

                    }
                });
            }
        }
        else if (0 == tokens.size())
        {
            if (View.VISIBLE == btnIn.getVisibility()) {
                // test用タイマ(仮)開始
                TimerStart();
            }
        }

    }

    private Bitmap transparent(Bitmap tmpBitmap) {
        int width = tmpBitmap.getWidth();
        int height = tmpBitmap.getHeight();
        int[] pixels = new int[width * height];
        int c = tmpBitmap.getPixel(0, 0);
        // 0,0 のピクセルと同じ色のピクセルを透明化する．
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        tmpBitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (pixels[x + y * width] == c) {
                    pixels[x + y * width] = 0;
                }
            }
        }
        bitmap.eraseColor(Color.argb(0, 0, 0, 0));
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);

        return tmpBitmap;
    }

    private void btnInAction() {
        // 出勤ボタン押下
        Toast.makeText(this, "出勤しました", Toast.LENGTH_SHORT).show();
        // Jsonファイル作成
        makeJsonFile(true);
        // ボタン非表示
        buttonDisplay(false);

        restartTimer();
    }
    // 退勤ボタン押下
    private void btnOutAction() {
        // 出勤ボタン押下
        Toast.makeText(this, "退勤しました", Toast.LENGTH_SHORT).show();
        // Jsonファイル作成
        makeJsonFile(false);
        // ボタン非表示
        buttonDisplay(false);

        restartTimer();
    }

    private void btnCancelAction() {
        // 出勤ボタン押下
        Toast.makeText(this, "キャンセルしました", Toast.LENGTH_SHORT).show();
        // ボタン非表示
        buttonDisplay(false);

        restartTimer();
    }

    // ボタン表示制御
    private void buttonDisplay(boolean dispBtn) {
        //if (true) {
        if (dispBtn) {
            btnIn.setVisibility(View.VISIBLE);
            btnOut.setVisibility(View.VISIBLE);
            btnCancel.setVisibility(View.VISIBLE);
        }
        else {
            btnIn.setVisibility(View.INVISIBLE);
            btnOut.setVisibility(View.INVISIBLE);
            btnCancel.setVisibility(View.INVISIBLE);

            if (null != imageView) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Bitmap dst = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888);
                        imageView.setImageBitmap(transparent(dst));
                    }
                });
            }

        }
    }

    private void makeJsonFile(boolean makeMode) {
        // 日時情報
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat dayFormat = new SimpleDateFormat("yyyy/MM/dd");
        SimpleDateFormat timeFormat = new SimpleDateFormat("HH:mm:ss");
        SimpleDateFormat fileFormat = new SimpleDateFormat("yyyyMMddHHmmss");

        String day = dayFormat.format(cal.getTime());
        String time = timeFormat.format(cal.getTime());
        String fileName = "attendance_" + fileFormat.format(cal.getTime()) + ".json";

        // Jsonファイル形式に置き換え
        // Stringの文字をダブルクォーテーションで囲む
        // JSONObjectのputは諦めた。。
        StringBuilder sBld = new StringBuilder();
        sBld.append("{");
        sBld.append("\"day\":");
        sBld.append("\"" + day + "\"");
        sBld.append(",");
        sBld.append("\"time\":");
        sBld.append("\"" + time + "\"");
        sBld.append(",");
        sBld.append("\"employeeNo\":");
        sBld.append("\"" + employeeNo + "\"");
        sBld.append(",");
        sBld.append("\"employeeName\":");
        sBld.append("\"" + employeeName + "\"");
        sBld.append(",");
        sBld.append("\"attendance\":");
        if (makeMode) {
            sBld.append("\"出勤\"");
        } else {
            sBld.append("\"退勤\"");
        }
        sBld.append("}");
        String jsonStr = sBld.toString();

        try {
            //FileOutputStream fileOutputstream = openFileOutput(fileName, MODE_PRIVATE);
            //fileOutputstream.write(jsonStr.getBytes());

            if (mAsyncTask != null) {
                mAsyncTask.cancel(true);
            }

            mAsyncTask = new SendPostAsyncTask(this){
                @Override
                protected void onPostExecute(String response) {
                    try {
                        JSONObject respData = new JSONObject(response);

                        Log.d( TAG, "レスポンスデータ : " + respData );
                    } catch ( JSONException e ) {
                        /// 解析エラー時の処理
                    }
                }
            }.execute( new SendPostTaskParams(
                    "https://192.168.1.45/test2.php",
                    new HashMap<String, Object>(){{ put( "post_1", jsonStr); }}
            ) );

            // TODO ファイルを送った後に削除がいる
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void saveBitmap(Bitmap saveImage, String name) throws IOException {

        final String SAVE_DIR = "/MyPhoto/";
        File file = new File(Environment.getExternalStorageDirectory().getPath() + SAVE_DIR);
        try{
            if(!file.exists()){
                file.mkdir();
            }
        }catch(SecurityException e){
            e.printStackTrace();
            throw e;
        }

        Date mDate = new Date();
        SimpleDateFormat fileNameDate = new SimpleDateFormat("yyyyMMdd_HHmmss");
        //String fileName = fileNameDate.format(mDate) + ".bmp";
        String fileName = name + "_" + fileNameDate.format(mDate) + ".bmp";
        String AttachName = file.getAbsolutePath() + "/" + fileName;

        try {
            FileOutputStream out = new FileOutputStream(AttachName);
            saveImage.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch(IOException e) {
            e.printStackTrace();
            throw e;
        }

        // save index
        ContentValues values = new ContentValues();
        ContentResolver contentResolver = getContentResolver();
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.TITLE, fileName);
        values.put("_data", AttachName);
        contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
    }
    /**
     * A native method that is implemented by the 'myapplication' native library,
     * which is packaged with this application.
     */
    public native String featureAnalyzer(long mat, long mat2, String path, String path2, int height, int width, int rotation);
    public native int featureOpen(String path, String path2);
}
