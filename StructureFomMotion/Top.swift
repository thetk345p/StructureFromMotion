//
//  Top.swift
//  StructureFomMotion
//
//  Created by kouki furukawa on 2020/04/27.
//  Copyright © 2020 kouki FURUKAWA. All rights reserved.
//

import Foundation

class VC_Top: UIViewController {
    @IBOutlet weak var ProjectName: UITextField!
        
    //AppDelegateのインスタンスを取得
    let appDelegate:AppDelegate = UIApplication.shared.delegate as! AppDelegate
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }

    @IBAction func Start(_ sender: UIButton) {
        // プロジェクト名の取得 Get project name
        var Name = ProjectName.text!
        
        // ディレクトリパス作成 Create directory path
        let documentPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let path_file_name = documentPath.appendingPathComponent( Name + "/" )

        // file:///var/mobile/Containers/Data/Application/F1B3414C-D25B-4B19-928F-243DDEE34581/Documents/プロジェクト名 となってしまうため，"file://" を消す
        var path_file_name_string = path_file_name.absoluteString
        if let range = path_file_name_string.range(of: "file://") {
            path_file_name_string.replaceSubrange(range, with: "")
        }
        
        // ディレクトリ作成 Create directory
        if(createDir(path_file_name_string) == true){
            // ディレクトリ名を AppDelegete 上の変数に保存
            appDelegate.DirectoryName = Name
            
            // 画面遷移 Screen transition
            let next = storyboard!.instantiateViewController(withIdentifier: "camera")
            next.modalTransitionStyle = .crossDissolve
            next.modalPresentationStyle = .fullScreen
            self.present(next, animated: true, completion: nil)
        }else{
            // 失敗した場合はアラート表示
            let alert: UIAlertController = UIAlertController(title: "エラー", message: "including invalid letter OR same name directory is already existing", preferredStyle:  UIAlertController.Style.alert)

            let defaultAction: UIAlertAction = UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler:{
                (action: UIAlertAction!) -> Void in
                print("error")
            })
            alert.addAction(defaultAction)
            present(alert, animated: true, completion: nil)
        }
    }
    
    
    
    fileprivate func createDir(_ dirPath: String) -> Bool {
        // ディレクトリを作成します。
        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(atPath: dirPath, withIntermediateDirectories: true, attributes: nil)
        } catch {
            // エラーの場合
            return false
        }
        return true
    }
}
