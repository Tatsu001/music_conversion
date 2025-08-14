#!/usr/bin/env python3
"""
音楽変換パイプライン Step 1: 高品質音源分離
Demucsの高品質モデルを使用してMP3ファイルをボーカル・ドラム・ベース・その他に分離

Features:
- 複数Demucsモデル対応 (htdemucs_ft, mdx_extra_q, htdemucs_6s)
- 高品質分離設定 (float32, overlap調整)
- 前処理パイプライン
- A/Bテスト機能
- 品質評価システム

Requirements:
- Python 3.11
- PyTorch 2.7.1+cu118 (CUDA対応)
- Demucs 4.0.1
- librosa 0.11.0
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class EnhancedMusicSeparator:
    """高品質音源分離クラス"""
    
    # 利用可能モデル定義
    AVAILABLE_MODELS = {
        "htdemucs": {
            "name": "htdemucs",
            "quality": "standard",
            "speed": "fast",
            "description": "標準モデル（デフォルト）"
        },
        "htdemucs_ft": {
            "name": "htdemucs_ft", 
            "quality": "high",
            "speed": "medium",
            "description": "ファインチューニング版（推奨）"
        },
        "mdx_extra_q": {
            "name": "mdx_extra_q",
            "quality": "highest", 
            "speed": "slow",
            "description": "最高品質版"
        },
        "htdemucs_6s": {
            "name": "htdemucs_6s",
            "quality": "high",
            "speed": "medium", 
            "description": "6ステム分離版"
        }
    }
    
    def __init__(self, 
                 model_name: str = "htdemucs_ft",
                 quality_preset: str = "high",
                 use_preprocessing: bool = True):
        """
        初期化
        
        Args:
            model_name: 使用するDemucsモデル名
            quality_preset: 品質プリセット（standard/high/highest）
            use_preprocessing: 前処理パイプラインの使用
        """
        self.model_name = model_name
        self.quality_preset = quality_preset
        self.use_preprocessing = use_preprocessing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎵 EnhancedMusicSeparator初期化")
        print(f"   モデル: {self.model_name}")
        print(f"   品質プリセット: {self.quality_preset}")
        print(f"   前処理: {'有効' if self.use_preprocessing else '無効'}")
        print(f"   デバイス: {self.device}")
        
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # モデル情報表示
        if model_name in self.AVAILABLE_MODELS:
            model_info = self.AVAILABLE_MODELS[model_name]
            print(f"   説明: {model_info['description']}")
        else:
            print(f"   ⚠️ 未知のモデルです: {model_name}")
    
    def get_quality_settings(self) -> Dict[str, any]:
        """品質プリセットに基づく設定を取得"""
        settings = {
            "standard": {
                "overlap": 0.5,
                "use_float32": False,
                "mp3_preset": 2,
                "segment": 4
            },
            "high": {
                "overlap": 0.75,
                "use_float32": True,
                "mp3_preset": 2,
                "segment": 7
            },
            "highest": {
                "overlap": 0.85,
                "use_float32": True,
                "mp3_preset": 2,
                "segment": 12
            }
        }
        return settings.get(self.quality_preset, settings["high"])
    
    def preprocess_audio(self, input_file: str, output_file: str = None) -> str:
        """音声前処理パイプライン"""
        if not self.use_preprocessing:
            return input_file
        
        print(f"🔧 音声前処理開始")
        
        # 前処理用一時ファイル
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_preprocessed.wav")
        
        try:
            # 音声読み込み
            y, sr = librosa.load(input_file, sr=44100)
            
            # 1. ピーク正規化 (-1dBFS)
            peak_norm_factor = 0.891  # -1dBFS
            y_normalized = y * (peak_norm_factor / np.max(np.abs(y)))
            
            # 2. RMS正規化 (-23 LUFS相当)
            target_rms = 0.1  # -23 LUFS相当
            current_rms = np.sqrt(np.mean(y_normalized**2))
            if current_rms > 0:
                rms_factor = target_rms / current_rms
                y_normalized = y_normalized * min(rms_factor, 1.0)  # クリッピング防止
            
            # 3. DC成分除去
            y_processed = y_normalized - np.mean(y_normalized)
            
            # 保存
            sf.write(output_file, y_processed, sr)
            
            print(f"   前処理完了: {Path(output_file).name}")
            print(f"   ピーク: {np.max(np.abs(y_processed)):.3f}")
            print(f"   RMS: {np.sqrt(np.mean(y_processed**2)):.3f}")
            
            return output_file
            
        except Exception as e:
            print(f"   ⚠️ 前処理エラー、オリジナルファイルを使用: {e}")
            return input_file
    
    def separate_audio(self, input_file: str, output_dir: str = None) -> Dict[str, str]:
        """
        高品質音源分離を実行
        
        Args:
            input_file: 入力音声ファイル
            output_dir: 出力ディレクトリ
            
        Returns:
            分離された各トラックのファイルパス辞書
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
        
        # 出力ディレクトリ設定
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_separated_enhanced"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 前処理実行
        processed_file = self.preprocess_audio(input_file)
        
        print(f"\n🔄 高品質音源分離開始")
        print(f"   入力: {input_file}")
        print(f"   処理済み: {Path(processed_file).name}")
        print(f"   出力: {output_dir}")
        
        # 処理時間測定開始
        start_time = time.time()
        
        try:
            # 品質設定取得
            quality_settings = self.get_quality_settings()
            
            # Demucsコマンド構築
            cmd = [
                "python", "-m", "demucs.separate",
                "-n", self.model_name,
                "-o", str(output_path),
                "-d", self.device,
                "--overlap", str(quality_settings["overlap"])
            ]
            
            # 高品質設定追加
            if quality_settings["use_float32"]:
                cmd.append("--float32")
            
            if quality_settings["segment"] > 4:
                cmd.extend(["--segment", str(quality_settings["segment"])])
            
            # 入力ファイル追加
            cmd.append(processed_file)
            
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
        finally:
            # 前処理一時ファイル削除
            if processed_file != input_file and os.path.exists(processed_file):
                os.remove(processed_file)
        
        # 処理時間測定終了
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 分離ファイル検索
        separated_files = self._find_separated_files(output_path, Path(processed_file).stem)
        
        print(f"✅ 高品質音源分離完了")
        print(f"   処理時間: {processing_time:.2f}秒")
        print(f"   分離ファイル数: {len(separated_files)}")
        
        # 分離結果のメタデータ保存
        self._save_separation_metadata(output_path, {
            "model": self.model_name,
            "quality_preset": self.quality_preset,
            "processing_time": processing_time,
            "settings": quality_settings,
            "separated_files": separated_files,
            "timestamp": datetime.now().isoformat()
        })
        
        return separated_files
    
    def _find_separated_files(self, output_dir: Path, stem: str) -> Dict[str, str]:
        """分離されたファイルを検索"""
        separated_files = {}
        
        # Demucsの出力構造を確認
        model_output_dir = output_dir / self.model_name / stem
        
        if not model_output_dir.exists():
            print(f"⚠️ 分離ファイルディレクトリが見つかりません: {model_output_dir}")
            return separated_files
        
        # htdemucs_6s は6ステム出力
        if self.model_name == "htdemucs_6s":
            track_names = ["vocals", "drums", "bass", "piano", "guitar", "other"]
        else:
            track_names = ["vocals", "drums", "bass", "other"]
        
        for track in track_names:
            track_file = model_output_dir / f"{track}.wav"
            if track_file.exists():
                separated_files[track] = str(track_file)
                print(f"   {track}: {track_file}")
            else:
                print(f"   ⚠️ {track}: ファイルなし")
        
        return separated_files
    
    def _save_separation_metadata(self, output_dir: Path, metadata: Dict):
        """分離メタデータを保存"""
        metadata_file = output_dir / "separation_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   メタデータ保存: {metadata_file.name}")
    
    def analyze_separation_quality(self, separated_files: Dict[str, str]) -> Dict[str, Dict]:
        """詳細な分離品質分析"""
        print(f"\n📊 詳細分離品質分析")
        
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
                
                # 高度な分析
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfccs, axis=1)
                
                # 動的範囲
                dynamic_range = np.max(np.abs(y)) - np.min(np.abs(y))
                
                analysis[track_name] = {
                    "duration": duration,
                    "rms_energy": float(rms_energy),
                    "zero_crossing_rate": float(zero_crossing_rate),
                    "spectral_centroid": float(spectral_centroid),
                    "spectral_bandwidth": float(spectral_bandwidth),
                    "spectral_rolloff": float(spectral_rolloff),
                    "dynamic_range": float(dynamic_range),
                    "mfcc_features": mfcc_mean.tolist(),
                    "file_size": os.path.getsize(file_path) / (1024*1024)  # MB
                }
                
                print(f"   {track_name}:")
                print(f"     時間: {duration:.2f}秒")
                print(f"     エネルギー: {rms_energy:.4f}")
                print(f"     スペクトル中心: {spectral_centroid:.1f}Hz")
                print(f"     動的範囲: {dynamic_range:.4f}")
                print(f"     ファイルサイズ: {analysis[track_name]['file_size']:.2f}MB")
                
            except Exception as e:
                print(f"   ❌ {track_name} 分析エラー: {e}")
                analysis[track_name] = {"error": str(e)}
        
        return analysis


def create_separation_comparison(input_file: str, models: List[str] = None) -> Dict[str, Dict]:
    """複数モデルでA/Bテスト実行"""
    if models is None:
        models = ["htdemucs", "htdemucs_ft", "mdx_extra_q"]
    
    print(f"\n🔄 A/Bテスト開始: {len(models)}モデル比較")
    
    results = {}
    input_path = Path(input_file)
    base_output_dir = input_path.parent / f"{input_path.stem}_comparison"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        print(f"\n--- {model} モデル実行 ---")
        
        try:
            separator = EnhancedMusicSeparator(
                model_name=model,
                quality_preset="high",
                use_preprocessing=True
            )
            
            model_output_dir = base_output_dir / model
            separated_files = separator.separate_audio(input_file, str(model_output_dir))
            
            if separated_files:
                analysis = separator.analyze_separation_quality(separated_files)
                
                results[model] = {
                    "separated_files": separated_files,
                    "analysis": analysis,
                    "model_info": EnhancedMusicSeparator.AVAILABLE_MODELS.get(model, {})
                }
                
                print(f"   ✅ {model} 完了")
            else:
                print(f"   ❌ {model} 失敗")
                
        except Exception as e:
            print(f"   ❌ {model} エラー: {e}")
            results[model] = {"error": str(e)}
    
    # 比較結果保存
    comparison_file = base_output_dir / "comparison_results.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 比較結果保存: {comparison_file}")
    return results


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="高品質音源分離システム")
    parser.add_argument("--input", "-i", type=str, help="入力音声ファイル")
    parser.add_argument("--output", "-o", type=str, help="出力ディレクトリ")
    parser.add_argument("--model", "-m", type=str, default="htdemucs_ft",
                       choices=list(EnhancedMusicSeparator.AVAILABLE_MODELS.keys()),
                       help="使用モデル")
    parser.add_argument("--quality", "-q", type=str, default="high",
                       choices=["standard", "high", "highest"],
                       help="品質プリセット")
    parser.add_argument("--no-preprocessing", action="store_true",
                       help="前処理を無効化")
    parser.add_argument("--compare", action="store_true",
                       help="複数モデル比較実行")
    
    args = parser.parse_args()
    
    print("🎵 高品質音楽変換パイプライン Step 1: 音源分離")
    print("=" * 60)
    
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
    
    # 入力ファイル確認
    if args.input:
        input_file = args.input
    else:
        # テスト用音声ファイル検索
        test_files = []
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']
        
        for ext in audio_extensions:
            test_files.extend(work_dir.glob(f"*{ext}"))
            test_files.extend(work_dir.glob(f"*{ext.upper()}"))
        
        test_files = list(set(test_files))
        
        if not test_files:
            print("\n📝 入力音声ファイルが見つかりません")
            print("   --input オプションでファイルを指定するか、")
            print(f"   {work_dir}/ に音声ファイルを配置してください")
            return
        
        input_file = str(test_files[0])
        print(f"\n🎵 自動選択: {Path(input_file).name}")
    
    # 利用可能モデル表示
    print(f"\n🔧 利用可能モデル:")
    for model_name, model_info in EnhancedMusicSeparator.AVAILABLE_MODELS.items():
        marker = "👈" if model_name == args.model else "  "
        print(f"   {marker} {model_name}: {model_info['description']}")
    
    try:
        if args.compare:
            # 複数モデル比較
            results = create_separation_comparison(input_file)
            print(f"\n✅ 比較テスト完了! 結果を確認してください。")
        else:
            # 単一モデル実行
            separator = EnhancedMusicSeparator(
                model_name=args.model,
                quality_preset=args.quality,
                use_preprocessing=not args.no_preprocessing
            )
            
            separated_files = separator.separate_audio(input_file, args.output)
            
            if separated_files:
                # 分離品質分析
                analysis = separator.analyze_separation_quality(separated_files)
                
                print(f"\n✅ 高品質分離完了!")
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
        print(f"   4. モデル確認: python -c 'import demucs.api; print(demucs.api.list_models())'")


if __name__ == "__main__":
    main()