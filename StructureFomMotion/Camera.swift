//
//  Camera.swift
//  StructureFomMotion
//
//  Created by kouki furukawa on 2020/04/27.
//  Copyright © 2020 kouki FURUKAWA. All rights reserved.
//

import Foundation
import AVFoundation

class VC_Camera: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // AppDelegateのインスタンスを取得
    let appDelegate:AppDelegate = UIApplication.shared.delegate as! AppDelegate
    
    // OpenCVクラスをインスタンス化
    var openCV = OpenCV()
    
    // 撮影回数
    var count: Int!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        // 撮影回数を初期化
        count = 0;
        
        // カメラを起動する Launch camera
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = self
        present(picker, animated: true, completion: nil)
    }


    /// シャッターボタンを押下した際、確認メニューに切り替わる
    /// - Parameters:
    ///   - picker: ピッカー
    ///   - info: 写真情報
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        // ドキュメントディレクトリの「ファイルURL」（URL型）定義
        var documentDirectoryFileURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        
        //UserDefaults のインスタンス生成
        let userDefaults = UserDefaults.standard
        
        
        // 撮影した画像の取得および命名
        let image = info[.originalImage] as! UIImage
        var fileName = String(count) + ".png"
        

        // ディレクトリのパスにファイル名をつなげてファイルのフルパスを作る
        var path = documentDirectoryFileURL.appendingPathComponent(appDelegate.DirectoryName + fileName)
        documentDirectoryFileURL = path

        
        //pngで保存する
        var pngImageData = image.pngData()
        do {
            try pngImageData!.write(to: documentDirectoryFileURL)
            userDefaults.set(documentDirectoryFileURL, forKey: "userImage")
        } catch {
            //エラー処理
            print("エラー")
        }
        

        
        
        
        // *************** 画像処理 ***************
        var resultImg: UIImage //結果を格納する
        resultImg = self.openCV.sift(image) //変換
        // *****************************************
        
        
        fileName = String(count) + "_Processed.png"
        
        // ディレクトリのパスにファイル名をつなげてファイルのフルパスを作る
        path = documentDirectoryFileURL.appendingPathComponent(appDelegate.DirectoryName + fileName)
        documentDirectoryFileURL = path
        
        // 変換後の画像をpngで保存する
        pngImageData = image.pngData()
        do {
            try pngImageData!.write(to: documentDirectoryFileURL)
            userDefaults.set(documentDirectoryFileURL, forKey: "userImage")
        } catch {
            //エラー処理
            print("エラー")
        }
        
        
        // 前の画面に戻る
        self.dismiss(animated: true, completion: nil)
    }
}
