#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QStandardPaths>
#include <QtGui>
#include <QtCore/QDebug>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/persistence.hpp>

#define SUBJECT_WIDTH   640.0f

using namespace std;
using namespace cv;

static const string PFolder    = "/home/pi/Entry_Feature/";                   // 検出対象の顔画像フォルダ（編集前）
static const string SFolder    = "/home/pi/_Subject/";                     // 検出対象の顔画像フォルダ（編集前）
static const string DFolder    = "/home/pi/_Detect/";                      // 検出対象の顔画像フォルダ（切り抜き後）
static const string FFolder    = "/home/pi/_Feature/";                     // 特徴点フォルダ（128次元）
static const string FFilename  = "Feature.yml";                            // 検出対象の顔画像（編集前）
static const string FFilename_Name  = "Feature_Name.dat";                  // 検出対象の個人情報
static const string YNFilename = "yunet.onnx";                             // YuNetの学習済みのモデル
static const string SFFilename = "face_recognizer_fast.onnx";              // SFaceの学習済みのモデル
static const string DFilename  = "Detect_";                                // 検出対象の顔画像（切り抜き後）
static const string Featurename    = "Feature_";                           // 個人特定用のタグ名（仮、複数人登録時は番号の編集が必要）

static QString  TagID;
static QString  TagName;
static QString  FileName;

static Mat      img;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_lineEdit_textEdited(const QString &arg1)
{
    TagID = arg1;
}

void MainWindow::on_lineEdit_2_textEdited(const QString &arg1)
{
    TagName = arg1;
}

void MainWindow::on_commandLinkButton_clicked()
{

    Mat         img_sub;
    String      SFileName;

    FileName = QFileDialog::getOpenFileName(this, tr("ファイルを開く"),
                                            QStandardPaths::writableLocation(QStandardPaths::DesktopLocation));
    ui->lineEdit_3->setText( FileName );

    SFileName = FileName.toStdString();

    img = imread(SFileName);
    if (img.empty())
    {
        cerr << "ERROR! READ \n";
        return;
    }

    //-----------------------------------------
    // 対象者画像の表示(編集前の画像）
    //-----------------------------------------
    // 640x480サイズに編集
    double  rt = (double)SUBJECT_WIDTH / img.cols;

    img_sub = img.clone();
    cv::resize( img_sub, img_sub, Size(), rt, rt );

    // 画像表示
    QImage  Qimg_sub(img_sub.data, img_sub.cols, img_sub.rows, QImage::Format_RGB888);
    Qimg_sub = Qimg_sub.rgbSwapped();

    ui->label_4->setPixmap( QPixmap::fromImage(Qimg_sub) );
    ui->label_4->show();

    waitKey(10);
}

void MainWindow::on_pushButton_clicked()
{
    //----------------------------------------------
    // 学習モデルの読み込み
    //   ・YuNetの学習済みのモデル
    //   ・SFaceの学習済みのモデル
    //----------------------------------------------
    Ptr<FaceDetectorYN> face_detector;

    face_detector = FaceDetectorYN::create(PFolder+YNFilename, "", Size(0, 0));
    if( face_detector.empty() )
    {
        cerr << "ERROR! FaceDetectorYN CREATE \n";
        return;
    }

    Ptr<FaceRecognizerSF> face_recognizer;
    face_recognizer = FaceRecognizerSF::create(PFolder+SFFilename, "");
    if( face_recognizer.empty() )
    {
        cerr << "ERROR! FaceRecognizerSF CREATE \n";
        return;
    }

    //----------------------------------------
    // 入力画像のサイズを変更（処理時間を考慮）
    // ＜注意＞
    //　　サイズを縮小すると検出率が下がる
    //----------------------------------------
    cv::resize( img, img, Size(), 0.5, 0.5 );

    //----------------------------------------
    // 入力画像のサイズの指定
    //----------------------------------------
    face_detector->setInputSize( Size( img.cols, img.rows ) );

    //----------------------------------------
    // 顔の検出
    //----------------------------------------
    int ret;
    Mat dtfaces;
    ret = face_detector->detect( img, dtfaces );
    if( ret == 0 )
    {
        cerr << "ERROR! DETECT \n";
        return;
    }
    if(dtfaces.empty())
    {
        cerr << "ERROR! DETECT EMPTY \n";
        return;
    }

    //----------------------------------------
    // 顔を切り抜く
    //----------------------------------------
    Mat aligned_face;
    bool retb;

    cout << "alignCrop" << endl;

    // 顔を切り抜く
    face_recognizer->alignCrop(img, dtfaces, aligned_face);

    // 顔画像を保存
    String DfileName = DFolder + DFilename + TagID.toStdString() + ".jpg";
    retb = imwrite(DfileName, aligned_face );
    if( retb == false )
    {
        cerr << "ERROR! IMWRITE \n";
    }

    // 画像表示
    QImage  Qimg_aligned(aligned_face.data, aligned_face.cols, aligned_face.rows, QImage::Format_RGB888);
    Qimg_aligned = Qimg_aligned.rgbSwapped();

    ui->label_7->setPixmap( QPixmap::fromImage(Qimg_aligned) );
    ui->label_7->show();

    //----------------------------------------------
    // 特徴の抽出
    //----------------------------------------------
    Mat face_feature;
    face_recognizer->feature(aligned_face, face_feature);

    cout << face_feature << endl;

    //----------------------------------------------
    // 特徴の保存（Mat構造を保存するためYAMLとする）
    //----------------------------------------------
    String ft = Featurename + TagID.toStdString();

    cv::FileStorage fs_write( FFolder+FFilename, cv::FileStorage::APPEND);

    fs_write << ft << face_feature;
    fs_write.release();

    //----------------------------------------------
    // 対象者情報の保存(binary)
    //----------------------------------------------
    fstream fs;

    ofstream ofs;
    ofs.open(FFolder+FFilename_Name, ios::binary|ios::app);
//    ofs<< TagID.toStdString() <<endl;
    ofs<< ft <<endl;
    ofs<< TagName.toStdString() <<endl;
    ofs.close();

    waitKey(10);
}

void MainWindow::on_pushButton_2_clicked()
{
    ui->lineEdit->clear();
    ui->lineEdit_2->clear();
    ui->lineEdit_3->clear();
    ui->label_4->clear();
    ui->label_7->clear();
}
