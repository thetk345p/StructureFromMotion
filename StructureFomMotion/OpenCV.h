//
//  OpenCV.h
//  StructureFomMotion
//
//  Created by kouki furukawa on 2020/04/27.
//  Copyright © 2020 kouki FURUKAWA. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenCV : NSObject
//+ or - (返り値 *)関数名:(引数の型 *)引数名;
//+ : クラスメソッド
//- : インスタンスメソッド

- (UIImage *)SIFT:(UIImage *)img;
- (UIImage *)SfM:(UIImage *)img1: (UIImage *)img2;
@end

NS_ASSUME_NONNULL_END
