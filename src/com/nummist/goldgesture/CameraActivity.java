package com.nummist.goldgesture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

public final class CameraActivity extends Activity
        implements CvCameraViewListener2 {

    // A tag for log output.
    private static final String TAG = "CameraActivity";
    
    // Parameters for face detection.
    private static final double SCALE_FACTOR = 1.2;
    private static final int MIN_NEIGHBORS = 3;
    private static final int FLAGS = Objdetect.CASCADE_SCALE_IMAGE;
    private static final double MIN_SIZE_PROPORTIONAL = 0.25;
    private static final double MAX_SIZE_PROPORTIONAL = 1.0;
    
    // The portion of the face that is excluded from feature
    // selection on each side.
    // (We want to exclude boundary regions containing background.)
    private static final double MASK_PADDING_PROPORTIONAL = 0.15;
    
    // Parameters for face tracking.
    private static final int MIN_FEATURES = 10;
    private static final int MAX_FEATURES = 80;
    private static final double MIN_FEATURE_QUALITY = 0.05;
    private static final double MIN_FEATURE_DISTANCE = 4.0;
    private static final float MAX_FEATURE_ERROR = 200f;
    
    // Parameters for gesture detection
    private static final double MIN_SHAKE_DIST_PROPORTIONAL = 0.04;
    private static final double MIN_NOD_DIST_PROPORTIONAL = 0.005;
    private static final double MIN_BACK_AND_FORTH_COUNT = 2;
    
    // The camera view.
    private CameraBridgeViewBase mCameraView;
    
    // The dimensions of the image before orientation.
    private double mImageWidth;
    private double mImageHeight;
    
    // The current gray image before orientation.
    private Mat mGrayUnoriented;
    
    // The current and previous equalized gray images.
    private Mat mEqualizedGray;
    private Mat mLastEqualizedGray;
    private Mat ROI;
    // The mask, in which the face region is white and the
    // background is black.
    private Mat mMask;
    private Scalar mMaskForegroundColor;
    private Scalar mMaskBackgroundColor;
    
    // The face detector, more detection parameters, and
    // detected faces.
    private CascadeClassifier mFaceDetector;
    private CascadeClassifier mEyesDetector;
    private CascadeClassifier mMouthDetector;
    private Size mMinSize;
    private Size mMaxSize;
    private Size mMinSize_e;
    private Size mMaxSize_e;
    private Size mMinSize_m;
    private Size mMaxSize_m;
    private MatOfRect mFaces;
    private MatOfRect mEyes;
    private MatOfRect mMouth;
    // The initial features before tracking.
    private MatOfPoint mInitialFeatures;
    
    // The current and previous features being tracked.
    private MatOfPoint2f mFeatures;
    private MatOfPoint2f mLastFeatures;
    
    // The status codes and errors for the tracking.
    private MatOfByte mFeatureStatuses;
    private MatOfFloat mFeatureErrors;
    
    // Whether a face was being tracked last frame.
    private boolean mWasTrackingFace;
    
    // Colors for drawing.
    private Scalar mFaceRectColor;
    private Scalar mEyesRectColor;
    private Scalar mMouthRectColor;
    private Scalar mFeatureColor;
    
    // Gesture detectors.
    private BackAndForthGesture mNodHeadGesture;
    private BackAndForthGesture mShakeHeadGesture;
    
    // The audio tree for the 20 questions game.
    private YesNoAudioTree mAudioTree;
    
    // The OpenCV loader callback.
    private BaseLoaderCallback mLoaderCallback =
            new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(final int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.d(TAG, "OpenCV loaded successfully");
                    mCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };
    
    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        final Window window = getWindow();
        window.addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        setContentView(R.layout.activity_camera);
        mCameraView = (CameraBridgeViewBase)
                findViewById(R.id.camera_view);
        //mCameraView.enableFpsMeter();
        mCameraView.setCvCameraViewListener(this);
    }
    
    @Override
    public void onPause() {
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        if (mAudioTree != null) {
            mAudioTree.stop();
        }
        resetGestures();
        super.onPause();
    }
    
    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9,
                this, mLoaderCallback);
    }
    
    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        if (mAudioTree != null) {
            mAudioTree.stop();
        }
        resetGestures();
    }

    @Override
    public void onCameraViewStarted(final int width,
            final int height) {
        
        mImageWidth = width;
        mImageHeight = height;
        Point pt1 = null,pt2 = null;
        initFaceDetector();
        mFaces = new MatOfRect();
        mEyes = new MatOfRect();
        mMouth = new MatOfRect();
        final int smallerSide;
        if (height < width) {
            smallerSide = height;
        } else {
            smallerSide = width;
        }
        
        final double minSizeSide =
                MIN_SIZE_PROPORTIONAL * smallerSide;
        double minSizeSide_e = minSizeSide*0.5;
        double minSizeSide_m = minSizeSide*0.25;
        mMinSize = new Size(minSizeSide, minSizeSide);
        mMinSize_e = new Size(minSizeSide_e,minSizeSide_e);
        mMinSize_m = new Size(minSizeSide_m,minSizeSide_m);
        final double maxSizeSide =
                MAX_SIZE_PROPORTIONAL * smallerSide;
        double maxSizeSide_e = maxSizeSide*0.5;
        double maxSizeSide_m = maxSizeSide*0.8;
        mMaxSize = new Size(maxSizeSide, maxSizeSide);
        
        mMaxSize_m = new Size(maxSizeSide_m,maxSizeSide_m);
        mMaxSize_e = new Size(maxSizeSide_e,maxSizeSide_e);
        mInitialFeatures = new MatOfPoint();
        mFeatures = new MatOfPoint2f(new Point());
        mLastFeatures = new MatOfPoint2f(new Point());
        mFeatureStatuses = new MatOfByte();
        mFeatureErrors = new MatOfFloat();
        
        mFaceRectColor = new Scalar(0.0, 0.0, 255.0);
        mEyesRectColor = new Scalar(0.0,255.0,0.0);
        mMouthRectColor = new Scalar(255.0,0.0,0.0);
        mFeatureColor = new Scalar(0.0, 255.0, 0.0);
        
        final double minShakeDist =
                smallerSide * MIN_SHAKE_DIST_PROPORTIONAL;
        mShakeHeadGesture = new BackAndForthGesture(minShakeDist);
        
        final double minNodDist =
                smallerSide * MIN_NOD_DIST_PROPORTIONAL;
        mNodHeadGesture = new BackAndForthGesture(minNodDist);
        
        mAudioTree = new YesNoAudioTree(this);
        //mAudioTree.start();
        
        mGrayUnoriented = new Mat(height, width, CvType.CV_8UC1);
        
        // The rest of the matrices are transposed.
        
        mEqualizedGray = new Mat(width, height, CvType.CV_8UC1);
        mLastEqualizedGray = new Mat(width, height, CvType.CV_8UC1);
        
        mMask = new Mat(width, height, CvType.CV_8UC1);
        mMaskForegroundColor = new Scalar(255.0);
        mMaskBackgroundColor = new Scalar(0.0);
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(final CvCameraViewFrame inputFrame) {
        final Mat rgba = inputFrame.rgba();
        
        // For processing, orient the image to portrait and equalize
        // it.
        Imgproc.cvtColor(rgba, mGrayUnoriented,
                         Imgproc.COLOR_RGBA2GRAY);
        Core.transpose(mGrayUnoriented, mEqualizedGray);
        Core.flip(mEqualizedGray, mEqualizedGray, -1);
        Imgproc.equalizeHist(mEqualizedGray, mEqualizedGray);
        
        final List<Point> featuresList;
        
        mFaceDetector.detectMultiScale(
                mEqualizedGray, mFaces, SCALE_FACTOR, MIN_NEIGHBORS,
                FLAGS, mMinSize, mMaxSize);
       
        mEyesDetector.detectMultiScale(mEqualizedGray, mEyes, SCALE_FACTOR, MIN_NEIGHBORS, FLAGS, mMinSize_e, mMaxSize_e);
        mMouthDetector.detectMultiScale(mEqualizedGray, mMouth, SCALE_FACTOR, MIN_NEIGHBORS, FLAGS, mMinSize_m, mMaxSize_m);
        if (mFaces.rows() > 0) {
        	 Rect[] facesArray = mFaces.toArray();  
           for (int i = 0; i <facesArray.length; i++)  {
                 
        	   int minX = facesArray[i].x;
               int minY = facesArray[i].y;
               int width = facesArray[i].width;
               int height = facesArray[i].height;
               int maxX = minX + width;
               int maxY = minY + height;


               Core.rectangle(
                       rgba, new Point(mImageWidth-minY, mImageHeight-minX),
                       new Point(mImageWidth-maxY, mImageHeight-maxX),
                       mFaceRectColor);
               
               //eyes
           
            if(mEyes.rows() > 0)
            {
            	Rect[] eyesArray = mEyes.toArray();  
            	for(int j = 0;j<eyesArray.length;j++){
//            	final double[] eyes = mEyes.get(0, 0);
            	int minX_e = eyesArray[j].x;
                int minY_e = eyesArray[j].y;
                int width_e = eyesArray[j].width;
                int height_e = eyesArray[j].height;
                int maxX_e = minX_e + width_e;
                int maxY_e = minY_e + height_e;

               if(minY_e<(minY+0.3*width)){
                Core.rectangle(
                        rgba, new Point(mImageWidth-minY_e, mImageHeight-minX_e),
                        new Point(mImageWidth-maxY_e, mImageHeight-maxX_e),
                        mEyesRectColor);
               }
//                String x1 = Integer.toString((int) (mImageWidth-minY_e));
//                String y1 = Integer.toString((int) (mImageHeight-minX_e));
//                String x2 = Integer.toString((int) (mImageWidth-maxY_e));
//                String y2 = Integer.toString((int) (mImageHeight-maxX_e));
//                
//                Log.i("x1", x1);
//                Log.i("y1", y1);
//                Log.i("x2", x2);
//                Log.i("y2", y2);
                
                
            }
            }
            
            //mouth
           
            if(mMouth.rows() > 0)
            {
            	Rect[] mouthArray = mMouth.toArray();  
            	for(int m = 0;m<mouthArray.length;m++){

            	int minX_m = mouthArray[m].x;
                int minY_m = mouthArray[m].y;
                int width_m = mouthArray[m].width;
                int height_m= mouthArray[m].height;
                int maxX_m= minX_m + width_m;
                int maxY_m = minY_m + height_m;
                int center_x = minX_m+width_m/2;
                int center_y = minY_m+height_m/2;
                
               if(center_y>(minY+width/2)&&center_y<maxY&&center_x>(minX+width/3)&&center_x<(maxX-width/3)){
                Core.rectangle(
                        rgba, new Point(mImageWidth-minY_m, mImageHeight-minX_m),
                        new Point(mImageWidth-maxY_m, mImageHeight-maxX_m),
                        mMouthRectColor);
               }
            }
            }
            // Create a mask for the face region.
            double smallerSide;
            if (height < width) {
                smallerSide = height;
            } else {
                smallerSide = width;
            }
            double maskPadding =
                    smallerSide * MASK_PADDING_PROPORTIONAL;
            mMask.setTo(mMaskBackgroundColor);
            Core.rectangle(
                    mMask,
                    new Point(minX + maskPadding,
                              minY + maskPadding),
                    new Point(maxX - maskPadding,
                              maxY - maskPadding),
                    mMaskForegroundColor, -1);
            

            mWasTrackingFace = true;
            
        } 
        }
        // Draw the current features.
//        for (int i = 0; i< featuresList.size(); i++) {
//            final Point p = featuresList.get(i);
//            final Point pTrans = new Point(
//                    mImageWidth - p.y,
//                    mImageHeight - p.x);
//            Core.circle(rgba, pTrans, 8, mFeatureColor);
//        }
        
        // Swap the references to the current and previous images.
        final Mat swapEqualizedGray = mLastEqualizedGray;
        mLastEqualizedGray = mEqualizedGray;
        mEqualizedGray = swapEqualizedGray;
        
        // Swap the references to the current and previous features.
        final MatOfPoint2f swapFeatures = mLastFeatures;
        mLastFeatures = mFeatures;
        mFeatures = swapFeatures;
        
        // Mirror (horizontally flip) the preview.
        Core.flip(rgba, rgba, 1);
        
        return rgba;
    }
    
    private void startGestureDetection() {
        
        double[] featuresCenter = Core.mean(mFeatures).val;
        
        // Motion in x may indicate a shake of the head.
        mShakeHeadGesture.start(featuresCenter[0]);
        
        // Motion in y may indicate a nod of the head.
        mNodHeadGesture.start(featuresCenter[1]);
    }
    
    private void updateGestureDetection() {
        
        final double[] featuresCenter = Core.mean(mFeatures).val;
        
        // Motion in x may indicate a shake of the head.
        mShakeHeadGesture.update(featuresCenter[0]);
        final int shakeBackAndForthCount =
                mShakeHeadGesture.getBackAndForthCount();
        //Log.i(TAG, "shakeBackAndForthCount=" +
        //        shakeBackAndForthCount);
        final boolean shakingHead =
                (shakeBackAndForthCount >=
                MIN_BACK_AND_FORTH_COUNT);
        
        // Motion in y may indicate a nod of the head.
        mNodHeadGesture.update(featuresCenter[1]);
        final int nodBackAndForthCount =
                mNodHeadGesture.getBackAndForthCount();
        //Log.i(TAG, "nodBackAndForthCount=" +
        //        nodBackAndForthCount);
        final boolean noddingHead =
                (nodBackAndForthCount >=
                MIN_BACK_AND_FORTH_COUNT);
        
        if (shakingHead && noddingHead) {
            // The gesture is ambiguous. Ignore it.
            resetGestures();
        } else if (shakingHead) {
            mAudioTree.takeNoBranch();
            resetGestures();
        } else if (noddingHead) {
            mAudioTree.takeYesBranch();
            resetGestures();
        }
    }
    
    private void resetGestures() {
//        if (mNodHeadGesture != null) {
//            mNodHeadGesture.resetCounts();
//        }
//        if (mShakeHeadGesture != null) {
//            mShakeHeadGesture.resetCounts();
//        }
    }
    
    private void initFaceDetector() {
        try {
            // Load cascade file from application resources.
            
            InputStream is = getResources().openRawResource(
                    R.raw.haarcascade_frontalface_alt);
            File cascadeDir = getDir(
                    "cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(
                    cascadeDir, "haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            
            mFaceDetector = new CascadeClassifier(
                    cascadeFile.getAbsolutePath());
            if (mFaceDetector.empty()) {
                Log.e(TAG, "Failed to load cascade");
                finish();
            } else {
                Log.i(TAG, "Loaded cascade from " +
                      cascadeFile.getAbsolutePath());
            }

            cascadeDir.delete();
            
            
            
            //load eyes
            InputStream is_e = getResources().openRawResource(
                    R.raw.haarcascade_eye);
            File cascadeDir_e = getDir(
                    "cascade", Context.MODE_PRIVATE);
            File cascadeFile_e = new File(
                    cascadeDir_e, "haarcascade_eye.xml");
            FileOutputStream os_e = new FileOutputStream(cascadeFile_e);
            
            byte[] buffer_e = new byte[4096];
            int bytesRead_e;
            while ((bytesRead_e = is_e.read(buffer_e)) != -1) {
                os_e.write(buffer_e, 0, bytesRead_e);
            }
            is_e.close();
            os_e.close();
            
            mEyesDetector = new CascadeClassifier(
                    cascadeFile_e.getAbsolutePath());
            if (mEyesDetector.empty()) {
                Log.e(TAG, "Failed to load cascade e");
                finish();
            } else {
                Log.i(TAG, "Loaded cascade from " +
                      cascadeFile_e.getAbsolutePath());
            }

            cascadeDir_e.delete();
            
            //load mouth
            InputStream is_m = getResources().openRawResource(
                    R.raw.haarcascade_mcs_nose);
            File cascadeDir_m = getDir(
                    "cascade", Context.MODE_PRIVATE);
            File cascadeFile_m = new File(
                    cascadeDir_m, "haarcascade_mcs_nose.xml");
            FileOutputStream os_m = new FileOutputStream(cascadeFile_m);
            
            byte[] buffer_m = new byte[4096];
            int bytesRead_m;
            while ((bytesRead_m = is_m.read(buffer_m)) != -1) {
                os_m.write(buffer_m, 0, bytesRead_m);
            }
            is_m.close();
            os_m.close();
            
            mMouthDetector = new CascadeClassifier(
                    cascadeFile_m.getAbsolutePath());
            if (mMouthDetector.empty()) {
                Log.e(TAG, "Failed to load cascade m");
                finish();
            } else {
                Log.i(TAG, "Loaded cascade from " +
                      cascadeFile_m.getAbsolutePath());
            }

            cascadeDir_m.delete();
            
            //load left ear
            InputStream is_le = getResources().openRawResource(
                    R.raw.haarcascade_mcs_nose);
            File cascadeDir_le = getDir(
                    "cascade", Context.MODE_PRIVATE);
            File cascadeFile_le = new File(
                    cascadeDir_le, "haarcascade_mcs_nose.xml");
            FileOutputStream os_le = new FileOutputStream(cascadeFile_le);
            
            byte[] buffer_le = new byte[4096];
            int bytesRead_le;
            while ((bytesRead_le = is_le.read(buffer_le)) != -1) {
                os_le.write(buffer_le, 0, bytesRead_le);
            }
            is_le.close();
            os_le.close();
            
            mMouthDetector = new CascadeClassifier(
                    cascadeFile_le.getAbsolutePath());
            if (mMouthDetector.empty()) {
                Log.e(TAG, "Failed to load cascade m");
                finish();
            } else {
                Log.i(TAG, "Loaded cascade from " +
                      cascadeFile_le.getAbsolutePath());
            }

            cascadeDir_le.delete();
            
            
            
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: "
                  + e);
            finish();
        }
    }
}
