#include <jni.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <__locale>
#include <locale>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/gapi/imgproc.hpp>

#include <android/bitmap.h>

using namespace std;
using namespace cv;

struct Feature
{
    string      TagName;    // 特徴認識ID
    string      PName;      // 名前（ローマ字）
    cv::Mat     feature;    // 特徴
};

#define DETECT_MAGNIFICATION 0.25f
#define OUT_MAGNIFICATION 4
#define DEBUG_PRINT 0

vector<Feature> PFeature;   // 登録済特徴点読み込み用テーブル
Feature feature;            // 個人情報読み込み用ワーク

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_facedetect_MainActivity_featureAnalyzer(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMatSrc,
        jlong   objMatDst,
        jstring path,
        jstring path2,
        jint height,
        jint width,
        jint rotation){

    Mat* matSrc = (Mat*) objMatSrc;
    Mat* matDst = (Mat*) objMatDst;

    int count = 3;
    // 配列文字列を作成
    jclass c = env->FindClass("java/lang/String");
    jobjectArray feature = env->NewObjectArray(count, c, jstring());

    const char *c1 = env->GetStringUTFChars(path, 0);
    const char *c2 = env->GetStringUTFChars(path2, 0);

    cv::Ptr<cv::FaceDetectorYN> face_detector;

    face_detector = FaceDetectorYN::create(c1, "", Size(0, 0));
    if( face_detector.empty() )
    {
        cerr << "ERROR! FaceDetectorYN CREATE \n";
        //return -1.;
    }

    cv::Ptr<cv::FaceRecognizerSF> face_recognizer;
    face_recognizer = cv::FaceRecognizerSF::create(c2, "");
    if( face_recognizer.empty() )
    {
        cerr << "ERROR! FaceRecognizerSF CREATE \n";
        //return -1;
    }

    Mat frame,img, img2, img3;
    int ret;
    Mat dtfaces;
    Mat match_face;
    Mat aligned_face;
    Mat face_feature;
    double score;
    string no, name;

    frame = *matSrc;

    switch(frame.channels())
    {
        case 1:
            cvtColor(frame, img2, COLOR_GRAY2BGR);
            break;
        case 4:
            cvtColor(frame, img2, COLOR_BGRA2BGR);
            break;
        default:
            img2 = frame;
            break;
    }

    img = img2.clone();
    resize( img2, img2, Size(), DETECT_MAGNIFICATION, DETECT_MAGNIFICATION );

    float x_correction,y_correction;
    x_correction = (float) width / (float) img.cols;
    y_correction = (float) height / (float) img.rows;

    //iii = 1;

    face_detector->setInputSize( Size( img2.cols, img2.rows ) );

    //----------------------------------------
    // 顔の検出
    //----------------------------------------
    //vector<double> dtfaces;
    ret = face_detector->detect( img2, dtfaces );
    if( ret == 0 )
    {
        cerr << "ERROR! DETECT \n";
        //return -1;
    }

    if( dtfaces.empty() == 0 ) {
#if DEBUG_PRINT
        cout << "dtfaces.rows : " << dtfaces.rows << endl;
#endif
        for( int i=0; i<dtfaces.rows; i++)
        {
            face_recognizer->alignCrop(img2, dtfaces.row(i), aligned_face);

            face_recognizer->feature(aligned_face, face_feature);

#if DEBUG_PRINT
            cout << "対象者数 : " << PFeature.size() << endl;
#endif
            for (size_t j = 0; j < PFeature.size(); j++) {
                //score = face_recognizer->match(subject_feature[j], face_feature);
                score = face_recognizer->match(PFeature[j].feature, face_feature);

                cout << "SCORE : " << score << endl;
                cout << "INDEX : " << j << endl;
#if DEBUG_PRINT
                cout << "SCORE : " << score << endl;
                    cout << "INDEX : " << j << endl;
                    // 検出した顔を表示（112ｘ112）
                    imshow("FUTURE", aligned_face);
#endif
                //----------------------------------------------
                // 特徴点の表示
                //----------------------------------------------
                if (score >= 0.3 ){
#if DEBUG_PRINT
                    cout << "MATCH" << endl;
#endif
                    //*matPrevious = face_feature;
                    //no = (PFeature[j].TagName).erase((PFeature[j].TagName).find("Feature_"));
                    string tagName = PFeature[j].TagName;
                    //no.erase(remove(no.begin(), no.end(), "Feature_"), no.end());
                    no = tagName.substr(8, tagName.length()-8);

                    name = PFeature[j].PName;


                    // 矩形表示
                    rectangle(*matDst,
                              Rect2i(int(dtfaces.at<float>(i, 0))*(OUT_MAGNIFICATION*x_correction),
                                     int(dtfaces.at<float>(i, 1))*(OUT_MAGNIFICATION*y_correction),
                                     int(dtfaces.at<float>(i, 2))*(OUT_MAGNIFICATION*y_correction),
                                     int(dtfaces.at<float>(i, 3))*(OUT_MAGNIFICATION*y_correction)),
                              Scalar(255, 255, 255), 2*y_correction);
                    putText(*matDst, no,
                            Point(int(dtfaces.at<float>(i, 0))*(OUT_MAGNIFICATION*x_correction),int(dtfaces.at<float>(i, 1))*(OUT_MAGNIFICATION*y_correction)),
                            FONT_HERSHEY_SIMPLEX,
                            0.75*y_correction,
                            Scalar(255,255,255),
                            2*y_correction);

#if 0
                    // ランドマーク表示
                        circle( img, Point2i(int(dtfaces.at<float>(i, 4))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 5))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 6))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 7))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 8))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 9))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 10))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 11))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 12))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 13))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
#endif
                    //break;
                }
            }
        }
    }

    return env->NewStringUTF((no + "," + name).c_str());
}

extern "C" JNIEXPORT int JNICALL
Java_com_example_facedetect_MainActivity_featureOpen(
        JNIEnv* env,
        jobject, /* this */
        jstring path,
        jstring path2) {

    const char *c1 = env->GetStringUTFChars(path, 0);
    const char *c2 = env->GetStringUTFChars(path2, 0);


    if (PFeature.size() == 0)
    {
        ifstream ifs;

        ifs.open(c1, ios::binary|ios::in);
        while(ifs>>feature.TagName>>feature.PName)
        {
            PFeature.push_back( feature );

        }
        ifs.close();

        cv::FileStorage fs_read(c2, cv::FileStorage::READ);

        for(size_t i=0; i<PFeature.size(); i++ )
        {
            fs_read[PFeature[i].TagName] >> PFeature[i].feature;
        }

        fs_read.release();

    }

    return PFeature.size();
}
