//
//  OpenCV.m
//  StructureFomMotion
//
//  Created by kouki furukawa on 2020/04/27.
//  Copyright © 2020 kouki FURUKAWA. All rights reserved.
//

// Objective-C++の環境構築 https://qiita.com/KyH/items/74008b793e277ee5cbc9
#import <opencv2/opencv.hpp>
#import <opencv2/highgui.hpp>
#import <opencv2/imgcodecs/ios.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <algorithm>
#import "OpenCV.h" //ライブラリによってはNOマクロがバッティングするので，これは最後にimport


using namespace cv;
using namespace std;

#define PI 3.14159265359

@implementation OpenCV
Mat DifferenceOfGaussian(Mat img, int size, double sigma, double k, int n);
Mat SearchExtremum(Mat img1, Mat img2, Mat img3, int k);
Mat gaussian(Mat img, int size, int sigma);
Mat concatHorizontally(Mat img1, Mat img2);

- (UIImage *) SIFT:(UIImage *)img{
    // *************** UIImage -> cv::Mat変換 ***************
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(img.CGImage);
    CGFloat cols = img.size.width;
    CGFloat rows = img.size.height;

    Mat mat(rows, cols, CV_8UC4);

    CGContextRef contextRef = CGBitmapContextCreate(mat.data,
                                                    cols,
                                                    rows,
                                                    8,
                                                    mat.step[0],
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault);

    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), img.CGImage);
    CGContextRelease(contextRef);

    // *************** 処理 ***************
    
    // Gray scale 変換
    Mat grayImg;
    cv::cvtColor(mat, grayImg, cv::COLOR_RGB2GRAY);
    

    /* Oriented FAST and Rotated BRIEF : ORB */
    std::vector<cv::KeyPoint> points;
    // アルゴリズムの選択
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 2);
    // キーポイント検出
    orb->detect(mat, points);
    // キーポイントの描画
    Mat result;
    cv::drawKeypoints(mat, points, result,   cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    /* 単なるShi-Tomasiのコーナー検出
    vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(grayImg, points, 100, 0.01, 100, cv::Mat(), 3, 3, 0, 0.04);
    // void goodFeaturesToTrack(const Mat& image, vector<Point2f>& corners, int maxCorners, double qualityLevel, double minDistance, const Mat& mask=Mat(), int blockSize=3, bool useHarrisDetector=false, double k=0.04)
    
    // 特徴点を丸で描く(RGBAだからA=255にしている)
    for (int i = 0; i < points.size(); i++) {
        cv::circle(mat, points[i], 50, cv::Scalar(255, 0, 0, 255), -1, cv::LINE_AA);
    }
     */
    
    /* SIFT(作りかけ)
    // DoG
    Mat DoG1 = DifferenceOfGaussian(grayImg, 3, 1.6, sqrt(2.0), 1);
    Mat DoG2 = DifferenceOfGaussian(grayImg, 3, 1.6, sqrt(2.0), 2);
    Mat DoG3 = DifferenceOfGaussian(grayImg, 3, 1.6, sqrt(2.0), 3);
    Mat DoG4 = DifferenceOfGaussian(grayImg, 3, 1.6, sqrt(2.0), 4);

    // 極値探索
    Mat Extremum2 = SearchExtremum(DoG1, DoG2, DoG3, 2);
    Mat Extremum3 = SearchExtremum(DoG2, DoG3, DoG4, 3);

    Mat result = Extremum2;// + Extremum3;

    equalizeHist(result, result);
     */
    
    
    /* AKAZE
    // AKAZEを使用する http://cvwww.ee.ous.ac.jp/opencv_practice5/
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);

    // 検出したキーポイント（特徴点）を格納する配列
    std::vector<cv::KeyPoint> keyAkaze;
    
    // キーポイント検出
    // CAUTION!!! iPhone8だとメモリが足らずクラッシュした
    akaze->detect(grayImg, keyAkaze);

    // 画像上にキーポイントの場所を描く
    Mat result;
    cv::drawKeypoints(grayImg, keyAkaze, result, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    */
    
    // *************** cv::Mat → UIImage ***************
    UIImage *resultImg = MatToUIImage(result);
    
    return resultImg;
}


- (UIImage *) SfM:(UIImage *)img1: (UIImage *)img2 {
    // *************** UIImage -> cv::Mat変換 ***************
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(img1.CGImage);
    CGFloat cols = img1.size.width;
    CGFloat rows = img1.size.height;

    Mat mat1(rows, cols, CV_8UC4);
    Mat mat2(rows, cols, CV_8UC4);

    // mat1
    CGContextRef contextRef = CGBitmapContextCreate(mat1.data,
                                                    cols,
                                                    rows,
                                                    8,
                                                    mat1.step[0],
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault);

    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), img1.CGImage);
    
    // mat2
    contextRef = CGBitmapContextCreate(mat2.data,
                                                    cols,
                                                    rows,
                                                    8,
                                                    mat2.step[0],
                                                    colorSpace,
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault);
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), img2.CGImage);
    
    
    // Release
    CGContextRelease(contextRef);

    // *************** 処理 ***************
    
    /* Shi-Tomashi corner detection
    // Gray scale 変換
    Mat grayImg1, grayImg2;
    cv::cvtColor(mat1, grayImg1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(mat2, grayImg2, cv::COLOR_RGB2GRAY);

    // 特徴点検出
    vector<cv::Point2f> points1, points2;
    cv::goodFeaturesToTrack(grayImg1, points1, 100, 0.01, 100, cv::Mat(), 3, 3, 0, 0.04);
    cv::goodFeaturesToTrack(grayImg2, points2, 100, 0.01, 100, cv::Mat(), 3, 3, 0, 0.04);
    // void goodFeaturesToTrack(const Mat& image, vector<Point2f>& corners, int maxCorners, double qualityLevel, double minDistance, const Mat& mask=Mat(), int blockSize=3, bool useHarrisDetector=false, double k=0.04)
    
    // 特徴点を丸で描く(RGBAだからA=255にしている)
    for (int i = 0; i < points1.size(); i++) {
        cv::circle(mat1, points1[i], 50, cv::Scalar(255, 0, 0, 255), -1, cv::LINE_AA);
    }
    for (int i = 0; i < points2.size(); i++) {
        cv::circle(mat2, points2[i], 50, cv::Scalar(0, 255, 0, 255), -1, cv::LINE_AA);
    }
     */
    
    
    /* Oriented FAST and Rotated BRIEF : ORB */
    // http://cvwww.ee.ous.ac.jp/opencv_practice5/
    std::vector<cv::KeyPoint> KeyPoints1, KeyPoints2;
    // アルゴリズムの選択
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 2);
    // キーポイント検出
    orb->detect(mat1, KeyPoints1);
    orb->detect(mat2, KeyPoints2);
    // 特徴点マッチングアルゴリズムの選択
    cv::Ptr<cv::DescriptorMatcher> hamming = cv::DescriptorMatcher::create("BruteForce-Hamming");
    // 特徴量記述の計算
    cv::Mat des1, des2;
    orb->compute(mat1, KeyPoints1, des1);
    orb->compute(mat2, KeyPoints2, des2);
    // 特徴点マッチング(特徴量記述des1とdes2のマッチングを行い、結果をmatchへ書き込む)
    std::vector<cv::DMatch> match;
    hamming->match(des1, des2, match);
    // 特徴量距離の小さい順にソートする（選択ソート）
    for (int i = 0; i < match.size() - 1; i++) {
       double min = match[i].distance;
       int n = i;
       for (int j = i + 1; j < match.size(); j++) {
           if (min > match[j].distance) {
               n = j;
               min = match[j].distance;
           }
       }
       std::swap(match[i], match[n]);
    }
    // 上位50点を残して、残りのマッチング結果を削除する。
    std::cout << match.size() << std::endl;
    match.erase(match.begin() + 50, match.end());
    std::cout << match.size() << std::endl;
    
    /* マッチング結果を太線で描画
    cv::Mat tmp_result = concatHorizontally(mat1, mat2);
    for (int i = 0; i < match.size() - 1; i++) {
        cv::Point img1_point = KeyPoints1[match[i].queryIdx].pt;
        cv::Point img2_point = KeyPoints2[match[i].trainIdx].pt;
        img2_point.x += cols; // img2はimg1の右側に連結されるから，横幅分足す
        
        int color = int(255 * i / match.size());
        cv::line(tmp_result, img1_point, img2_point, cv::Scalar(color, 0, 255 - color, 255), 10);
    }
    return MatToUIImage(tmp_result);
     */
    

    // 参考：https://blog.negativemind.com/2017/08/04/opencv-triangulate-points-function/
    

    // ステレオペアの算出
    std::vector<cv::Point2f>pts1, pts2;
    for (int j = 0; j < (int)match.size(); j++) {
       pts1.push_back(KeyPoints1[match[j].queryIdx].pt);
       pts2.push_back(KeyPoints2[match[j].trainIdx].pt);
    }

    
    /*
    // カメラの内部パラメータを求める
    // ピクセル単位の焦点距離fx,fyは，「mm単位の焦点距離/画素ピッチ」で表される．
    // 画素ピッチ = イメージセンサの横or縦の長さ/x方向の受光素子数
    // 簡単のため，fx=fyとする．
    // iPhone8の焦点距離は f=3.99[mm]，画素ピッチは1.22*10^(-3)[mm/pixel] (c.f. ソニー製の裏面照射型CMOS)
    // 以上より，fx = fy = 3.99/(1.22*10^(-3)) = 3270[pixel]
    const float fx = 3270;
    const float fy = 3270;
    // 光学中心は画像中心とする
    const float cx = cols / 2;
    const float cy = rows / 2;
    */
    
    // 【追記】基本行列の算出においては，焦点距離はどうでも良い（=1.0で良い）みたい．
    // → https://mem-archive.com/2018/09/17/post-615/
    // 主点を-1している理由 → https://ja.coder.work/so/c%2B%2B/860763
    const float cx = cols / 2 - 1;
    const float cy = rows / 2 - 1;
    const float focal = 1.0;
    cv::Mat essential_matrix = cv::findEssentialMat(pts1, pts2, focal, cv::Point2f(cx, cy));
    
    
    // 以下，相互標定R,t（カメラ1→カメラ2への変換）の推定 http://pukulab.blog.fc2.com/blog-entry-44.html
    
    // カメラ2の姿勢の推定
    cv::Matx33d R; // 1→2への回転行列
    cv::Matx31d t; // 1→2への平行移動
    cv::recoverPose(essential_matrix, pts1, pts2, R, t, focal, cv::Point2f(cx, cy));
    
    // R_1,t_1は初期カメラ姿勢(今回だと2枚だけなので，img1の姿勢)
    // R_12,t_12は，img1からimg2に射影変換した姿勢
    cv::Matx33d R_1, R_12;
    cv::Matx31d t_1, t_12;
    R_1 << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;
    t_1 << 0.0, 0.0, 0.0;
    // カメラ移動
    R_12 = R_1 * R.t();
    t_12 = t_1 - R_12 * t;
    cv::Matx43d EstimatedPose;
    EstimatedPose << R_12(0,0), R_12(0,1), R_12(0,2), t_12(0,0),
          R_12(1,0), R_12(1,1), R_12(1,2), t_12(1,0),
          R_12(2,0), R_12(2,1), R_12(2,2), t_12(2,0);
    
    std::cout << EstimatedPose << std::endl;

    // キーポイントの座標を取得したい時
    // KeyPoints.at(i).pt
    
    
    
    
    
    /*
     [Caution!!!] これより先は Multi-View Stereo の領域
     
     画像1の深度情報を効率よく＆確度よく調べるために，基礎行列Fを求める．
     これによって画像1の各点が画像2の各エピポーラ線に写像できる．
     
     Fが求まったら，画像1の各点に対応する画像2のエピポーラ線上でPhoto-consistency関数を調べ，ある一定の閾値を超えたら対応点とする．
     対応点の画像1側の深度をバンドル調整（三角測量）によって算出する．（先に求めた相互標定R,tを使う）
     
     なお，ある画素のPhoto-consistency関数が全て閾値以下であれば，その画素はオブジェクトでないただの背景と考える．
    */
     
    
    
    
    
     // Fundamental matrixの算出
     Mat fundamental_matrix = findFundamentalMat(pts1, pts2, FM_8POINT);
     std::cout << fundamental_matrix << std::endl;
     
     // F行列と特徴点から，エピポーラ線の方程式を計算
     // Vec3fについて：OpenCVでは，opencv2/core/core.hpp の中で typedef Vec<float, 3> Vec3f; の記述あり
     // →
     // 出力 epilines[0] = ai, epilines[1] = bi, epilines[2] = ci    aix + biy + ci = 0
     std::vector<Vec3f> epilines1, epilines2;
     computeCorrespondEpilines(pts1, 1, fundamental_matrix, epilines1); // 1:1枚目の点→2枚目の直線に(F行列をそのまま使用)
     computeCorrespondEpilines(pts2, 2, fundamental_matrix, epilines2); // 2:2枚目の点→1枚目の直線に(F行列を転置して使用)
    // std::cout << epilines1[5]; // デバッグ


     // エピポーラ線描画（画像1の特徴点 → 画像2のエピポーラ線）
     for (int i = 0; i < (int)epilines1.size(); i++) {
         float a = epilines1[i][0];
         float b = epilines1[i][1];
         float c = epilines1[i][2];
         int x1 = 0;
         int y1 = int(-(a * x1 + c)/b);
         int x2 = cols;
         int y2 = int(-(a * x2 + c)/b);
         
         int color = int(255 * i / epilines1.size());
         cv::circle(mat1, pts1[i], 50, cv::Scalar(color, 0, 255 - color, 255), -1, cv::LINE_AA);
         cv::line(mat2, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(color, 0, 255 - color, 255), 25);
     }
    
    
    
    // 特徴点マッチングの結果を画像化する
    Mat result;
    cv::drawMatches(mat1, KeyPoints1, mat2, KeyPoints2, match, result);

    
//    // 正規化相互相関によるマッチング http://zin.co.jp/blog/opencv-matchtemplateの種類まとめ/
//    MatchTemplate(mat1, current_template, result, CV_TM_CCORR_NORMED)
    
    
    // *************** cv::Mat → UIImage ***************
    return MatToUIImage(result);
    
}


Mat DifferenceOfGaussian(Mat img, int size, double sigma, double k, int n){
    // 元画像に2つのσでガウシアンフィルタをかけて差分を出力．
    // SIFTで使うため，シグマは σ*k^(n-1) と σ*k^n とする．
    Mat result;
    Mat blur1, blur2;
    GaussianBlur(img, blur1, cv::Size(size, size), n * pow(k, n - 1) );
    GaussianBlur(img, blur2, cv::Size(size, size), n * pow(k, n) );
//    result = blur1 - blur2;
    absdiff(blur1, blur2, result);
    return result;
}

/* 注目画素を26近傍と比較，最大なら(x,y)にkを格納． */
Mat SearchExtremum(Mat img1, Mat img2, Mat img3, int k){
    int cols = img1.cols;
    int rows = img1.rows;
    
    // 結果を格納するMat
    Mat result = Mat::zeros(cv::Size(rows, cols), CV_8UC1);
    
    /* 3*3の窓を考えるため，画像の端の1pixelは無視する． */
    for(int y = 1 ; y < rows - 1 ; y++){
        for(int x = 1 ; x < cols - 1 ; x++){
            // 注目画素
            unsigned char *p = img2.ptr<unsigned char>(y);
            int interest_pixel = p[x];
            
            // 関心領域の画素を全て配列pixel_arrayに格納
            int max_pixel = -100;
            int min_pixel = 256;
            for(int i = -1 ; i <= 1 ; i++){
                unsigned char *p1 = img1.ptr<unsigned char>(y + i);
                unsigned char *p2 = img2.ptr<unsigned char>(y + i);
                unsigned char *p3 = img3.ptr<unsigned char>(y + i);
                
                // 27画素の最大値，最小値を求める．
                for(int j = -1 ; j <= 1 ; j++){
                    if(max_pixel < p1[x + j]){
                        max_pixel = p1[x + j];
                    }
                    if(max_pixel < p2[x + j]){
                        max_pixel = p2[x + j];
                    }
                    if(max_pixel < p3[x + j]){
                        max_pixel = p3[x + j];
                    }

                    if(min_pixel > p1[x + j]){
                        min_pixel = p1[x + j];
                    }
                    if(min_pixel > p2[x + j]){
                        min_pixel = p2[x + j];
                    }
                    if(min_pixel > p3[x + j]){
                        min_pixel = p3[x + j];
                    }

                }
            }
            
            // 注目画素が最大or最小なら(x,y,k)を保存
            if(max_pixel == interest_pixel || min_pixel == interest_pixel){
                result.data[x * rows + y] = k;
            }

        }
    }
    
    return result;
    
    
}


// ガウシアンフィルタ(size=3,sigma=0.85 でよく使われるカーネルの再現を確認済み)
Mat gaussian(Mat img, int size, double sigma) {
    /* input img must be gray-scale(CV_8UC1) */
    int cols = img.cols;
    int rows = img.rows;
    Mat result = Mat::zeros(cv::Size(rows, cols), CV_8UC1);
    
    
    // 重みのsumを取る（正規化のため）
    double weight_sum = 0.0;
    for(int i = - size / 2 ; i <= size / 2 ; i ++){
        for(int j = - size / 2 ; j <= size / 2 ; j ++){
            // 二次元ガウシアン関数
            double f = exp(- ( i * i + j * j ) / ( 2 * sigma * sigma ) ) / ( 2 * PI * sigma * sigma ) ;
            
            // sum取る
            weight_sum += f;
        }
    }
    

    // 原理通りのガウシアンフィルタ
    // 端は計算しない
    for(int y = size / 2 ; y < rows - size / 2 ; y++){
        // result画像の座標(0, y)のポインタ取得
        unsigned char *p = result.ptr<unsigned char>(y);
        for(int x = size / 2 ; x < cols - size / 2 ; x++){
            double tmp = 0.0;

            // カーネルの畳み込み計算
            for(int i = - size / 2 ; i <= size / 2 ; i ++){
                unsigned char *p2 = img.ptr<unsigned char>(y + i);
                for(int j = - size / 2 ; j <= size / 2 ; j ++){
                    // 重みを計算
                    double f = exp(- ( i * i + j * j ) / ( 2 * sigma * sigma ) ) / ( 2 * PI * sigma * sigma ) ;

                    // 重み付けして加算
                    tmp += f * double(p2[x + j]);
                }
            }

            // 結果代入(正規化する)
            p[x] = int(tmp / weight_sum);
        }
    }

    
    // 作りかけ
//    // 高速ガウシアンフィルタ W(x,y) = W1(x) * W2(y)
//    // 中間画像を生成
//    Mat intermediate = Mat::zeros(cv::Size(rows, cols), CV_8UC1);
//
//    // 端は計算しない
//    for(int y = size / 2 ; y < rows - size / 2 ; y++){
//
//        for(int i = - size / 2 ; i <= size / 2 ; i ++){
//
//            for(int x = size / 2 ; x < cols - size / 2 ; x++){
//                // 中間画像を生成
//                double tmp = 0.0;
//
//                unsigned char *p1 = intermediate.ptr<unsigned char>(y);
//                unsigned char *p2 = img.ptr<unsigned char>(y + i);
//                for(int j = - size / 2 ; j <= size / 2 ; j ++){
//                    // 重みを計算
//                    double f = exp(- ( j * j ) / ( 2 * sigma * sigma ) ) / ( sqrt(2 * PI) * sigma ) ;
//
//                    // 中間画像を生成
//                    tmp += f * double(p2[x + j]);
//                }
//            }
//
//            // result画像の座標(0, y)のポインタ取得
//            unsigned char *p = result.ptr<unsigned char>(y);
//
//        }
//    }
//
//    result /= weight_sum;
    
    return result;
}


// 入力画像を横に連結して返す
Mat concatHorizontally(Mat img1, Mat img2){
    int rows = img1.rows;
    int cols = img1.cols;
    
    // 2つの画像を横に並べた大きさのMat型を作成する
    Mat base(rows, 2 * cols, CV_8UC4);
    // img1を張り付ける領域をあらわすMat型を作成する
    Mat roi1(base, cv::Rect(0 , 0 , cols , rows));
    // img1をroi1にコピーする
    img1.copyTo(roi1);
    // img2を張り付ける領域をあらわすMat型を作成する
    Mat roi2(base, cv::Rect(cols, 0, cols , rows));
    // img2をroi2にコピーする
    img2.copyTo(roi2);
    // 結果代入
    Mat result = base;
    
    return result;
}


@end


