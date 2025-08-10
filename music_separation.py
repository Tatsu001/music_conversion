#!/usr/bin/env python3
"""
音楽変換パイプライン Step 1: 音源分離
Demucsを使用してMP3ファイルをボーカル・ドラム・ベース・その他に分離

Requirements:
- Python 3.11
- PyTorch 2.7.1+cu118 (CUDA対応)
- Demucs 4.0.1
- librosa 0.11.0
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import torch
import librosa
import soundfile as sf
from typing import Dict, List, Tuple

class MusicSeparator:
    """Demucsを使用した音源分離クラス"""
    
    def __init__(self, model_name: str = "htdemucs"):
        """
        初期化
        
        Args:
            model_name: 使用するDemucsモデル名
                      - "htdemucs": 高品質（デフォルト）
                      - "htdemucs_ft": ファインチューニング版
                      - "mdx_extra": 軽量版
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎵 MusicSeparator初期化")
        print(f"   モデル: {self.model_name}")
        print(f"   デバイス: {self.device}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    def separate_audio(self, input_file: str, output_dir: str = None) -> Dict[str, str]:
        """
        音源分離を実行
        
        Args:
            input_file: 入力音声ファイル（MP3等）
            output_dir: 出力ディレクトリ（None時は入力ファイル名ベース）
            
        Returns:
            分離された各トラックのファイルパス辞書
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
        
        # 出力ディレクトリ設定
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_separated"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 音源分離開始")
        print(f"   入力: {input_file}")
        print(f"   出力: {output_dir}")
        
        # 処理時間測定開始
        start_time = time.time()
        
        try:
            # Demucsコマンド実行（Demucs 4.0.1対応）
            cmd = [
                "python", "-m", "demucs.separate",
                "-n", self.model_name,  # --model から -n に変更
                "-o", str(output_path),  # --out から -o に変更
                "-d", self.device,      # --device から -d に変更
                str(input_path)
            ]
            
            print(f"   実行コマンド: {' '.join(cmd)}")
            
            # サブプロセスで実行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                print(f"❌ Demucs実行エラー:")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                raise RuntimeError(f"Demucs実行失敗: {result.stderr}")
            
        except Exception as e:
            print(f"❌ 音源分離エラー: {e}")
            raise
        
        # 処理時間測定終了
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 分離ファイル検索
        separated_files = self._find_separated_files(output_path, input_path.stem)
        
        print(f"✅ 音源分離完了")
        print(f"   処理時間: {processing_time:.2f}秒")
        print(f"   分離ファイル数: {len(separated_files)}")
        
        return separated_files
    
    def _find_separated_files(self, output_dir: Path, stem: str) -> Dict[str, str]:
        """分離されたファイルを検索"""
        separated_files = {}
        
        # Demucsの出力構造: output_dir/model_name/stem/track.wav
        model_output_dir = output_dir / self.model_name / stem
        
        if not model_output_dir.exists():
            print(f"⚠️ 分離ファイルディレクトリが見つかりません: {model_output_dir}")
            return separated_files
        
        # 標準的な分離トラック名
        track_names = ["vocals", "drums", "bass", "other"]
        
        for track in track_names:
            track_file = model_output_dir / f"{track}.wav"
            if track_file.exists():
                separated_files[track] = str(track_file)
                print(f"   {track}: {track_file}")
            else:
                print(f"   ⚠️ {track}: ファイルなし")
        
        return separated_files
    
    def analyze_separation_quality(self, separated_files: Dict[str, str]) -> Dict[str, Dict]:
        """分離品質の簡易分析"""
        print(f"\n📊 分離品質分析")
        
        analysis = {}
        
        for track_name, file_path in separated_files.items():
            if not os.path.exists(file_path):
                continue
                
            try:
                # 音声ファイル読み込み
                y, sr = librosa.load(file_path, sr=None)
                
                # 基本統計
                duration = len(y) / sr
                rms_energy = librosa.feature.rms(y=y)[0].mean()
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0].mean()
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
                
                analysis[track_name] = {
                    "duration": duration,
                    "rms_energy": float(rms_energy),
                    "zero_crossing_rate": float(zero_crossing_rate),
                    "spectral_centroid": float(spectral_centroid),
                    "file_size": os.path.getsize(file_path) / (1024*1024)  # MB
                }
                
                print(f"   {track_name}:")
                print(f"     時間: {duration:.2f}秒")
                print(f"     エネルギー: {rms_energy:.4f}")
                print(f"     ファイルサイズ: {analysis[track_name]['file_size']:.2f}MB")
                
            except Exception as e:
                print(f"   ❌ {track_name} 分析エラー: {e}")
                analysis[track_name] = {"error": str(e)}
        
        return analysis


def main():
    """メイン実行関数"""
    print("🎵 音楽変換パイプライン Step 1: 音源分離")
    print("=" * 50)
    
    # 作業ディレクトリ設定
    work_dir = Path.home() / "ドキュメント" / "conversion_music"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 作業ディレクトリ: {work_dir}")
    
    # GPU環境確認
    if torch.cuda.is_available():
        print(f"🚀 CUDA利用可能: {torch.version.cuda}")
        print(f"   GPU数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ CUDA利用不可、CPUで実行します")
    
    # テスト用音声ファイル確認
    test_files = []
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']
    
    for ext in audio_extensions:
        test_files.extend(work_dir.glob(f"*{ext}"))
        # 大文字小文字の違いも考慮
        test_files.extend(work_dir.glob(f"*{ext.upper()}"))
    
    # 重複除去
    test_files = list(set(test_files))
    
    if not test_files:
        print("\n📝 テスト用音声ファイルが見つかりません")
        print("   以下の場所に音声ファイルを配置してください:")
        print(f"   {work_dir}/")
        print("   対応形式: MP3, WAV, M4A, FLAC, AAC, OGG")
        print("\n   サンプルファイル生成コマンド（オプション）:")
        print("   # 10秒のテスト音声生成")
        print("   ffmpeg -f lavfi -i 'sine=frequency=440:duration=10' test_audio.wav")
        return
    
    print(f"\n🎵 見つかった音声ファイル: {len(test_files)}個")
    for i, file in enumerate(test_files[:5]):  # 最初の5個表示
        print(f"   {i+1}. {file.name}")
    
    # 音源分離実行
    separator = MusicSeparator(model_name="htdemucs")
    
    # 最初のファイルで実行
    test_file = test_files[0]
    print(f"\n🔄 テスト実行: {test_file.name}")
    
    try:
        separated_files = separator.separate_audio(str(test_file))
        
        if separated_files:
            # 分離品質分析
            analysis = separator.analyze_separation_quality(separated_files)
            
            print(f"\n✅ Step 1 完了!")
            print(f"   次のステップ: MIDI変換 (Step 2)")
            print(f"   分離されたファイル:")
            for track, path in separated_files.items():
                print(f"     {track}: {Path(path).name}")
        else:
            print(f"❌ 分離ファイルが見つかりませんでした")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(f"\n🔧 トラブルシューティング:")
        print(f"   1. 仮想環境の有効化: source music_conversion_env/bin/activate")
        print(f"   2. Demucsインストール確認: pip list | grep demucs")
        print(f"   3. CUDA環境確認: python -c 'import torch; print(torch.cuda.is_available())'")
        print(f"   4. Demucsヘルプ確認: python -m demucs.separate --help")


if __name__ == "__main__":
    main()