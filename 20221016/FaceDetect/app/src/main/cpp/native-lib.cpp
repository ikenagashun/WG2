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

using namespace std;

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_facedetect_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_facedetect_MainActivity_processImage(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMatSrc,
        jlong   objMatDst) {

    cv::Mat* matSrc = (cv::Mat*) objMatSrc;
    cv::Mat* matDst = (cv::Mat*) objMatDst;

    static cv::Mat *matPrevious = NULL;
    if (matPrevious == NULL) {
        /* lazy initialization */
        matPrevious = new cv::Mat(matSrc->rows, matSrc->cols, matSrc->type());
    }
    cv::absdiff(*matSrc, *matPrevious, *matDst);
    *matPrevious = *matSrc;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_facedetect_MainActivity_readFeature(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMatSrc,
        jlong   objMatDst) {


    //-----------------------------------------------------
    // 対象者情報の取得
    //    特徴認識IDと特徴を取得し、特徴照合用テーブルに追加する。
    // ＜注意＞
    //     ・対象者情報（Feature_Name.dat）は
    //       対象者登録時（Dictionary_Generate）にて生成される。
    //     ・日本語での表示は別途対応が必要
    //-----------------------------------------------------
    //cv::FileStorage fs_read(FFolder+FFilename, cv::FileStorage::READ);

    //for(size_t i=0; i<PFeature.size(); i++ )
    {
    //    fs_read[PFeature[i].TagName] >> PFeature[i].feature;
#if DEBUG_PRINT
        cout << PFeature[i].TagName << endl;
        cout << PFeature[i].PName << endl;
        cout << PFeature[i].feature << endl;
        cout << endl;
        cout << endl;
#endif
    }
    //fs_read.release();

    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_facedetect_MainActivity_writeFeature(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMatSrc,
        jlong   objMatDst) {


    //-----------------------------------------------------
    // 対象者情報の取得
    //    特徴認識IDと特徴を取得し、特徴照合用テーブルに追加する。
    // ＜注意＞
    //     ・対象者情報（Feature_Name.dat）は
    //       対象者登録時（Dictionary_Generate）にて生成される。
    //     ・日本語での表示は別途対応が必要
    //-----------------------------------------------------
    //cv::FileStorage fs_read(FFolder+FFilename, cv::FileStorage::READ);

    //for(size_t i=0; i<PFeature.size(); i++ )
    //{
    //    fs_read[PFeature[i].TagName] >> PFeature[i].feature;
#if DEBUG_PRINT
        cout << PFeature[i].TagName << endl;
        cout << PFeature[i].PName << endl;
        cout << PFeature[i].feature << endl;
        cout << endl;
        cout << endl;
#endif
    //}
    //fs_read.release();

    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_facedetect_MainActivity_savemat(
        JNIEnv *env,
        jobject,
        jlong addrmat,
        jstring path)
{
    const char *nativepath = env->GetStringUTFChars(path, 0);
    cv::Mat* mat = (cv::Mat*) addrmat;
   // Mat& mat = *(Mat*)addrmat;


    cv::FileStorage storage(nativepath, cv::FileStorage::WRITE);
    //storage << "img" << mat;
    //storage << "img" << mat;
    //storage[nativepath] >> mat;
    storage.release();

    //const char *nativeString = env->GetStringUTFChars(location, NULL);
   // s//tring filepath=string(nativeString);
   // FileStorage fs(filepath, FileStorage::WRITE);
   // fs<<"frameCount"<<5;
   // fs.release();

    env->ReleaseStringUTFChars(path, nativepath);
}
extern "C" JNIEXPORT int JNICALL
Java_com_example_facedetect_MainActivity_processImage2(
        JNIEnv* env,
        jobject, /* this */
        jobject objMatSrc,
        jstring path) {

    const char *nativepath = env->GetStringUTFChars(path, 0);
    cv::Mat& mat = *(cv::Mat*)objMatSrc;

    cv::FileStorage storage(nativepath, cv::FileStorage::WRITE);
    //storage["kawata"] >> mat;
    storage.release();

    env->ReleaseStringUTFChars(path, nativepath);

    return 0;
}

struct Feature
{
    string TagName;    // 特徴認識ID
    string  PName;      // 名前（ローマ字）
    cv::Mat     feature;    // 特徴
};

extern "C" JNIEXPORT void  JNICALL
Java_com_example_facedetect_MainActivity_readmat(
        JNIEnv *env,
        jobject,
        //jlong addrmat,
        jlong addrmat)
{
    string dataPath = "/storage/emulated/0/Pictures/";
    string featureData = "Feature.yml";
    string nameData = "Feature_Name.dat";
    string testData = "Test.txt";

    //const char *nativepath = env->GetStringUTFChars(path, 0);
    cv::Mat* mat = (cv::Mat*) addrmat;
    //Mat& mat = *(Mat*)addrmat;
    cv::Mat mat2;

    cv::FileStorage storage(dataPath+ "/" + featureData , cv::FileStorage::READ);
    //storage["mat"] >> mat;
    //storage << "img" << mat;
    storage.release();

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

    ifs.open(dataPath + nameData, ios::binary|ios::in);
    while(ifs>>feature.TagName>>feature.PName)
    {
        PFeature.push_back( feature );
#if DEBUG_PRINT
        cout << "TagName : " << feature.TagName << endl;
        cout << "PName   : " << feature.PName << endl;
#endif
    }
    ifs.close();
    //env->ReleaseStringUTFChars(path, nativepath);

    cv::FileStorage fs_read(dataPath + featureData, cv::FileStorage::READ);

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

    jobjectArray ret;
    int i;

    ret= (jobjectArray)env->NewObjectArray(PFeature.size(),
                                           env->FindClass("java/lang/String"),
                                           env->NewStringUTF(""));
}
