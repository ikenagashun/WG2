#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <opencv2/core/persistence.hpp>

using namespace std;
using namespace cv;

#define magnification
#define DETECT_MAGNIFICATION 0.25f
#define OUT_MAGNIFICATION 4
#define DEBUG_PRINT 0

static const string PFolder         = "/home/pi/Detect_Personal/";
static const string FFolder         = "/home/pi/_Feature/";
static const string FFilename       = "Feature.yml";
static const string FFilename_Name  = "Feature_Name.dat";
static const string YNFilename      = "yunet.onnx";                               // YuNetの学習済みのモデル
static const string SFFilename      = "face_recognizer_fast.onnx";                // SFaceの学習済みのモデル

// 検出対象者情報の構造体
struct Feature
{
    string  TagName;    // 特徴認識ID
    string  PName;      // 名前（ローマ字）
    Mat     feature;    // 特徴
};

int main()
{
    vector<Feature> PFeature;   // 登録済特徴点読み込み用テーブル
    Feature feature;            // 個人情報読み込み用ワーク

    //-----------------------------------------------------
    // 対象者情報の取得(特徴認識ID、名前（英字））し、テーブル化する。
    // ＜注意＞
    //     ・対象者情報（Feature_Name.dat）は
    //       対象者登録時（Dictionary_Generate）に生成される。
    //     ・日本語での表示は別途対応が必要
    //-----------------------------------------------------
    ifstream ifs;

    ifs.open(FFolder+FFilename_Name, ios::binary|ios::in);
    while(ifs>>feature.TagName>>feature.PName)
    {
        PFeature.push_back( feature );
#if DEBUG_PRINT
        cout << "TagName : " << feature.TagName << endl;
        cout << "PName   : " << feature.PName << endl;
#endif
    }
    ifs.close();

    //-----------------------------------------------------
    // 対象者情報の取得
    //    特徴認識IDと特徴を取得し、特徴照合用テーブルに追加する。
    // ＜注意＞
    //     ・対象者情報（Feature_Name.dat）は
    //       対象者登録時（Dictionary_Generate）にて生成される。
    //     ・日本語での表示は別途対応が必要
    //-----------------------------------------------------
    cv::FileStorage fs_read(FFolder+FFilename, cv::FileStorage::READ);

#if DEBUG_PRINT
    cout << "PFeature Size : " << PFeature.size() << endl;
#endif

    for(size_t i=0; i<PFeature.size(); i++ )
    {
        fs_read[PFeature[i].TagName] >> PFeature[i].feature;
#if DEBUG_PRINT
        cout << PFeature[i].TagName << endl;
        cout << PFeature[i].PName << endl;
        cout << PFeature[i].feature << endl;
        cout << endl;
        cout << endl;
#endif
    }
    fs_read.release();

    //----------------------------------------------
    // 学習モデルの読み込み
    //   ・YuNetの学習済みのモデル
    //   ・SFaceの学習済みのモデル
    //
    //　＜注意＞
    //　　　モデルの使い回しは、match()のscore値が
    //　　　1固定になるので、再読込すること
    //----------------------------------------------
    Ptr<FaceDetectorYN> face_detector;

    face_detector = FaceDetectorYN::create(PFolder+YNFilename, "", Size(0, 0));
    if( face_detector.empty() )
    {
        cerr << "ERROR! FaceDetectorYN CREATE \n";
        return -1;
    }

    Ptr<FaceRecognizerSF> face_recognizer;
    face_recognizer = FaceRecognizerSF::create(PFolder+SFFilename, "");
    if( face_recognizer.empty() )
    {
        cerr << "ERROR! FaceRecognizerSF CREATE \n";
        return -1;
    }

    // カメラ画像の読み込み
    VideoCapture cap(0);
    if( !cap.isOpened())
    {
        cerr << "ERROR OPEN CAMERA\n";
        return -1;
    }

#if DEBUG_PRINT
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
#endif

// FPSの変更（Debug用）
//    cap.set(cv::CAP_PROP_FPS, 10 );
//    cout << "FPS : " << cap.get(cv::CAP_PROP_FPS) << endl;

    Mat frame,img, img2;
    Rect rct;
    int ret;
    Mat dtfaces;
    Mat match_face;
    Mat aligned_face;
    Mat face_feature;
    double score;

// 時間測定タイマー（Debug用）
//    TickMeter   tm;

    while( true )
    {
//tm.start();

        // １フレームを取得
        cap >> frame;

        if( frame.empty())
        {
            cerr << "ERROR! Blank frame grabbed \n";
            break;
        }

        //------------------------------------------
        // 画像チャンネルの変換（チャンネル３以外は３に変換）
        //------------------------------------------
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

        //----------------------------------------
        // 入力画像のサイズを変更（処理時間を考慮）
        //----------------------------------------
        img = img2.clone();
        resize( img2, img2, Size(), DETECT_MAGNIFICATION, DETECT_MAGNIFICATION );

#if DEBUG_PRINT
        cout << "ImageWidth:" << img2.cols << "  ImageHeight:" << img2.rows << endl;
#endif
        //----------------------------------------
        // 入力画像のサイズの指定
        //----------------------------------------
        face_detector->setInputSize( Size( img2.cols, img2.rows ) );

        //----------------------------------------
        // 顔の検出
        //----------------------------------------
        //vector<double> dtfaces;
        ret = face_detector->detect( img2, dtfaces );
        if( ret == 0 )
        {
            cerr << "ERROR! DETECT \n";
            return -1;
        }

        //----------------------------------------------
        // 特徴の抽出
        //----------------------------------------------
        if( dtfaces.empty() == 0 )
        {
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
                for( size_t j=0; j<PFeature.size(); j++)
                {
                    //score = face_recognizer->match(subject_feature[j], face_feature);
                    score = face_recognizer->match(PFeature[j].feature, face_feature);

#if DEBUG_PRINT
                    cout << "SCORE : " << score << endl;
                    cout << "INDEX : " << j << endl;
                    // 検出した顔を表示（112ｘ112）
                    imshow("FUTURE", aligned_face);
#endif
                    //----------------------------------------------
                    // 特徴点の表示
                    //----------------------------------------------
                    if( score >= 0.3)
                    {
#if DEBUG_PRINT
                        cout << "MATCH" << endl;
#endif
                        // 矩形表示
                        rectangle(img,
                                  Rect2i(int(dtfaces.at<float>(i, 0))*OUT_MAGNIFICATION,
                                         int(dtfaces.at<float>(i, 1))*OUT_MAGNIFICATION,
                                         int(dtfaces.at<float>(i, 2))*OUT_MAGNIFICATION,
                                         int(dtfaces.at<float>(i, 3))*OUT_MAGNIFICATION),
                                         Scalar(0, 255, 0), 2);
                        putText(img, PFeature[j].PName,
                                Point(int(dtfaces.at<float>(i, 0))*OUT_MAGNIFICATION,int(dtfaces.at<float>(i, 1))*OUT_MAGNIFICATION),
                                FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);

#if 0
                        // ランドマーク表示
                        circle( img, Point2i(int(dtfaces.at<float>(i, 4))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 5))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 6))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 7))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 8))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 9))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 10))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 11))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
                        circle( img, Point2i(int(dtfaces.at<float>(i, 12))*OUT_MAGNIFICATION, int(dtfaces.at<float>(i, 13))*OUT_MAGNIFICATION), 2, Scalar(255, 0, 0), 2);
#endif
                        break;
                    }
                }
            }

#if DEBUG_PRINT
imshow("FACE", aligned_face);       // 切り抜いた顔を表示
#endif
        }

        // 入力画像（カメラ画像）を拡大する場合
        //resize( img, img, Size(), 2.0, 2.0 );

        // 入力画像＋矩形を表示
        imshow( "Frame", img );

        waitKey( 5 );
    }
}
