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
#import "OpenCV.h" //ライブラリによってはNOマクロがバッティングするので，これは最後にimport


using namespace cv;
using namespace std;

#define PI 3.14159265359

@implementation OpenCV
Mat DifferenceOfGaussian(Mat img, int size, double sigma, double k, int scale);
Mat gaussian(Mat img, int size, int sigma);

- (UIImage *) SIFT:(UIImage *)img{
    // 今はテストでグレースケールの変換だけ行っている
    
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
    Mat grayImg;
    cv::cvtColor(mat, grayImg, cv::COLOR_RGB2GRAY); //グレースケール変換
    
    // Distinctive Image Features from Scale-Invariant Keypoints でSIFTについて触れられている．
    // http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
    // octave 4
    // scale 5
    // sigma 1.6
    // k √2
    Mat difference1,difference2,difference3,difference4,equ;
    double k = sqrt(2.0);
    double sigma = 1.6;
    difference1 = DifferenceOfGaussian(grayImg, 3, sigma, k, 1);
    difference2 = DifferenceOfGaussian(grayImg, 3, sigma, k, 2);
    difference3 = DifferenceOfGaussian(grayImg, 3, sigma, k, 3);
    difference4 = DifferenceOfGaussian(grayImg, 3, sigma, k, 4);
    
    equalizeHist(difference1, equ);
    
    
    // *************** cv::Mat → UIImage ***************
    UIImage *resultImg = MatToUIImage(equ);
    
    return resultImg;
}

Mat DifferenceOfGaussian(Mat img, int size, double sigma, double k, int scale){
    // 元画像に2つのσでガウシアンフィルタをかけて差分を出力．
    // SIFTで使うため，シグマは σ*k^(n-1) と σ*k^n とする．(n=scale)
    Mat result;
    Mat blur1, blur2;
    GaussianBlur(img, blur1, cv::Size(size, size), sigma * pow(k, scale - 1 ) );
    GaussianBlur(img, blur2, cv::Size(size, size), sigma * pow(k, scale) );
    result = blur1 - blur2;
    return result;
}




// ガウシアンフィルタ(size=3,sigma=0.85 でよく使われるカーネルの再現を確認済み)
Mat gaussian(Mat img, int size, double sigma) {
    /* input img must be gray-scale(CV_8UC1) */
    int cols = img.cols;
    int rows = img.rows;
    Mat result = img.clone();
    
    
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
    
    
    // 端は計算しない
    for(int y = size / 2 ; y < rows - size / 2 ; y++){
        // result画像の座標(0, y)のポインタ取得
        unsigned char *p = result.ptr<unsigned char>(y);
        for(int x = size / 2 ; x < cols - size / 2 ; x++){
            // result画像の座標(x,y)のピクセル値を初期化（元画像をコピーしてるため）
            p[x] = 0;
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

    
    return result;
}

@end
