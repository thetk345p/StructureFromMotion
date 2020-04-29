//
//  Camera.swift
//  StructureFomMotion
//
//  Created by kouki furukawa on 2020/04/27.
//  Copyright © 2020 kouki FURUKAWA. All rights reserved.
//

import Foundation
import AVFoundation

// 撮った画像を回転させないためのextension
extension UIImage {
    func fixedOrientation() -> UIImage {
        if self.imageOrientation == .up { return self }
        
        UIGraphicsBeginImageContextWithOptions(self.size, false, self.scale)
        self.draw(in: CGRect(origin: CGPoint.zero, size: self.size))
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image!
    }
}

class VC_Camera: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // AppDelegateのインスタンスを取得
    let appDelegate:AppDelegate = UIApplication.shared.delegate as! AppDelegate
    
    // ドキュメントディレクトリの「ファイルURL」（URL型）定義
    let documentDirectoryFileURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

    // Outlet宣言
    @IBOutlet weak var Nlabel: UILabel!
    @IBOutlet weak var LastPic: UIImageView!
    
    // OpenCVクラスをインスタンス化
    var openCV = OpenCV()
    
    // 撮影回数
    var count: Int!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        // 撮影回数を初期化
        count = 0;
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
                
        // ラベル
        Nlabel.text = "This is the " + String(count) + "-th photo"
        
        // 最後に撮った写真
        let path = documentDirectoryFileURL.appendingPathComponent(appDelegate.DirectoryName)
        let fileURL = URL(fileURLWithPath: path.absoluteString).appendingPathComponent(String(count) + "_Processed.png")
        // 画像が回転するのを防ぐ(UIImage に extension を設定している)
        // https://blog.morizotter.com/2015/06/20/uiimage-orientation-problem/
        let unfixedImage = UIImage(contentsOfFile: fileURL.path)
        let fixedImage = unfixedImage?.fixedOrientation()
        LastPic.image = fixedImage //UIImage(contentsOfFile: fileURL.path)?.fixedOrientation()
        print(fileURL.path)
    }
    
    @IBAction func TakePhoto(_ sender: UIButton) {
        // カメラを起動
        let sourceType:UIImagePickerController.SourceType = UIImagePickerController.SourceType.camera
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera){
             // インスタンスの作成
             let cameraPicker = UIImagePickerController()
             cameraPicker.sourceType = sourceType
             cameraPicker.delegate = self
             self.present(cameraPicker, animated: true, completion: nil)
         }
         else{
             print("Can't open camera")
         }
    }
    


    // 写真を撮った後に呼び出される
    /// - Parameters:
    ///   - picker: ピッカー
    ///   - info: 写真情報
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        count += 1
        
        //UserDefaults のインスタンス生成
        let userDefaults = UserDefaults.standard
        
        
        // 撮影した画像の取得および命名
        let unfixed_image = info[.originalImage] as! UIImage
        let image = unfixed_image.fixedOrientation()
        var fileName = String(count) + ".png"
        

        // ディレクトリのパスにファイル名をつなげてファイルのフルパスを作る
        var path = documentDirectoryFileURL.appendingPathComponent(appDelegate.DirectoryName)
        path = path.appendingPathComponent(fileName)

        //pngで保存する
        var pngImageData = image.pngData()
        do {
            try pngImageData!.write(to: path)
            userDefaults.set(path, forKey: "userImage")
        } catch {
            //エラー処理
            print("error 1")
        }
        

        
        
        
        // *************** 画像処理 ***************
        var resultImg: UIImage //結果を格納する
        resultImg = self.openCV.sift(image) //変換
        // *****************************************

        fileName = String(count) + "_Processed.png"
        
        // ディレクトリのパスにファイル名をつなげてファイルのフルパスを作る
        path = documentDirectoryFileURL.appendingPathComponent(appDelegate.DirectoryName)
        path = path.appendingPathComponent(fileName)

        // 変換後の画像をpngで保存する
        pngImageData = resultImg.pngData()
        do {
            try pngImageData!.write(to: path)
            userDefaults.set(path, forKey: "userImage")
        } catch {
            //エラー処理
            print("error 2")
        }
        
        // アルバムにも保存する
        UIImageWriteToSavedPhotosAlbum(resultImg, nil, nil, nil)
        
        // 前の画面に戻る
         self.dismiss(animated: true, completion: nil)
    }
    
    
    
    @IBAction func Back(_ sender: UIButton) {
        // 前の画面に戻る
         self.dismiss(animated: true, completion: nil)
    }
    
}
