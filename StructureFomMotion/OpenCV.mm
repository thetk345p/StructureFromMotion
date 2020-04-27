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

@implementation OpenCV
- (UIImage *) SIFT:(UIImage *)img{
    // 今はテストでグレースケールの変換だけ行っている
    
    // *************** UIImage -> cv::Mat変換 ***************
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(img.CGImage);
    CGFloat cols = img.size.width;
    CGFloat rows = img.size.height;

    cv::Mat mat(rows, cols, CV_8UC4);

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
    cv::Mat grayImg;
    cv::cvtColor(mat, grayImg, cv::COLOR_RGB2GRAY); //グレースケール変換

    // *************** cv::Mat → UIImage ***************
    UIImage *resultImg = MatToUIImage(grayImg);
    return resultImg;
}
@end
