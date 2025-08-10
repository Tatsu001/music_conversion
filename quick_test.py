#!/usr/bin/env python3
"""クイックテストスクリプト"""

def quick_test():
    print("🧪 クイックテスト実行中...")
    
    # 基本ライブラリテスト
    try:
        import numpy as np
        import torch
        import librosa
        import demucs
        import madmom
        import basic_pitch
        import pretty_midi
        
        print("✓ 全ライブラリインポート成功")
        
        # GPU確認
        if torch.cuda.is_available():
            print(f"✓ GPU利用可能: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CPU実行モード")
        
        # madmom機能テスト
        from madmom.features.beats import RNNBeatProcessor
        processor = RNNBeatProcessor()
        print("✓ madmom機能確認完了")
        
        print("\n🎉 全システム正常動作中！")
        return True
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False

if __name__ == "__main__":
    quick_test()
