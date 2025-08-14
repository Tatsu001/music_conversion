#!/usr/bin/env python3
"""
高品質音源分離結果からのMIDI変換
enhanced_music_separation.pyで分離されたファイルを対象にMIDI変換実行

Features:
- htdemucs_ft高品質分離結果対応
- 品質比較レポート生成
- 既存システムとの連携
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# MIDI変換システムをインポート
sys.path.append(str(Path(__file__).parent))
from midi_quantizer import MIDIQuantizer, QuantizationSettings

# 必要なライブラリをインポート
import numpy as np
import librosa
import pretty_midi
import warnings
warnings.filterwarnings('ignore')

try:
    from basic_pitch.inference import predict_and_save, predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
    print("✅ basic-pitch利用可能")
except ImportError:
    print("⚠️ basic-pitch がインストールされていません")
    BASIC_PITCH_AVAILABLE = False


class MIDIConverter:
    """音声トラックをMIDIに変換するクラス"""
    
    def __init__(self, quantization_settings=None, original_audio_file=None):
        self.sample_rate = 22050
        self.quantization_settings = quantization_settings
        self.quantizer = None
        self.original_audio_file = original_audio_file
        
        if quantization_settings and quantization_settings.enabled:
            self.quantizer = MIDIQuantizer(quantization_settings)
            print("🎼 MIDIConverter初期化完了（量子化機能付き）")
        else:
            print("🎼 MIDIConverter初期化完了")
    
    def convert_separated_tracks(self, separated_files: Dict[str, str], output_dir: str = None) -> Dict[str, str]:
        """分離されたトラックをMIDIに変換"""
        if not separated_files:
            raise ValueError("分離ファイルが指定されていません")
        
        if output_dir is None:
            first_file = Path(next(iter(separated_files.values())))
            output_dir = first_file.parent.parent / "midi_tracks"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\\n🎼 MIDI変換開始")
        print(f"   出力ディレクトリ: {output_dir}")
        
        midi_files = {}
        
        for track_name, audio_file in separated_files.items():
            if not os.path.exists(audio_file):
                print(f"   ⚠️ {track_name}: ファイルが見つかりません")
                continue
            
            print(f"\\n   🔄 {track_name} 処理中...")
            
            try:
                if track_name in ["vocals", "other"]:
                    midi_file = self._convert_melody_track(audio_file, output_path, track_name)
                elif track_name == "drums":
                    midi_file = self._convert_drum_track(audio_file, output_path, track_name)
                elif track_name == "bass":
                    midi_file = self._convert_bass_track(audio_file, output_path, track_name)
                else:
                    midi_file = self._convert_melody_track(audio_file, output_path, track_name)
                
                if midi_file:
                    # 量子化適用（元音源でBPM検出）
                    if self.quantizer:
                        print(f"      🎯 量子化適用中...")
                        bpm_source = self.original_audio_file if self.original_audio_file else audio_file
                        midi_file = self.quantizer.quantize_midi_file(midi_file, bpm_source, track_name)
                    
                    midi_files[track_name] = midi_file
                    print(f"      ✅ 変換完了: {Path(midi_file).name}")
                else:
                    print(f"      ❌ 変換失敗")
                    
            except Exception as e:
                print(f"      ❌ エラー: {e}")
        
        print(f"\\n✅ MIDI変換完了: {len(midi_files)}個のファイル生成")
        return midi_files
    
    def _convert_melody_track(self, audio_file: str, output_dir: Path, track_name: str) -> str:
        """メロディトラック変換"""
        if not BASIC_PITCH_AVAILABLE:
            return self._convert_with_librosa_melody(audio_file, output_dir, track_name)
        
        try:
            output_file = output_dir / f"{track_name}.mid"
            
            predict_and_save(
                [audio_file], str(output_dir), save_midi=True,
                sonify_midi=False, save_model_outputs=False, save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH
            )
            
            generated_file = output_dir / f"{Path(audio_file).stem}_basic_pitch.mid"
            if generated_file.exists():
                generated_file.rename(output_file)
                return str(output_file)
            
            return None
        except Exception as e:
            print(f"         basic-pitch エラー: {e}")
            return self._convert_with_librosa_melody(audio_file, output_dir, track_name)
    
    def _convert_with_librosa_melody(self, audio_file: str, output_dir: Path, track_name: str) -> str:
        """librosa メロディ変換"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            f0 = librosa.yin(y, fmin=80, fmax=800, sr=sr)
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            
            midi_data = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            
            current_note = None
            note_start = None
            
            for time, freq in zip(times, f0):
                if freq > 100 and not np.isnan(freq):
                    midi_note = int(librosa.hz_to_midi(freq))
                    
                    if current_note is None or abs(midi_note - current_note) > 1:
                        if current_note is not None and note_start is not None:
                            if time - note_start >= 0.1:
                                note = pretty_midi.Note(80, current_note, note_start, time)
                                instrument.notes.append(note)
                        current_note = midi_note
                        note_start = time
                else:
                    if current_note is not None and note_start is not None:
                        if time - note_start >= 0.1:
                            note = pretty_midi.Note(80, current_note, note_start, time)
                            instrument.notes.append(note)
                        current_note = None
                        note_start = None
            
            midi_data.instruments.append(instrument)
            output_file = output_dir / f"{track_name}.mid"
            midi_data.write(str(output_file))
            
            print(f"         librosa変換: {len(instrument.notes)}個の音符")
            return str(output_file)
            
        except Exception as e:
            print(f"         librosa変換エラー: {e}")
            return None
    
    def _convert_drum_track(self, audio_file: str, output_dir: Path, track_name: str) -> str:
        """ドラムトラック変換"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, units='frames', hop_length=512,
                backtrack=True, pre_max=3, post_max=3,
                delta=0.2, wait=10
            )
            
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            midi_data = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0, is_drum=True)
            
            for onset_time in onset_times:
                note = pretty_midi.Note(
                    velocity=100, pitch=36, start=onset_time, end=onset_time + 0.1
                )
                instrument.notes.append(note)
            
            midi_data.instruments.append(instrument)
            output_file = output_dir / f"{track_name}.mid"
            midi_data.write(str(output_file))
            
            print(f"         ドラム変換: {len(onset_times)}個のヒット")
            return str(output_file)
            
        except Exception as e:
            print(f"         ドラム変換エラー: {e}")
            return None
    
    def _convert_bass_track(self, audio_file: str, output_dir: Path, track_name: str) -> str:
        """ベーストラック変換"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            f0 = librosa.yin(y, fmin=40, fmax=300, sr=sr)
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            
            midi_data = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=33)  # Bass
            
            notes = []
            for time, freq in zip(times, f0):
                if freq > 50 and not np.isnan(freq):
                    midi_note = max(24, min(60, int(librosa.hz_to_midi(freq))))
                    notes.append((time, midi_note))
            
            if notes:
                for i in range(len(notes) - 1):
                    start_time, pitch = notes[i]
                    end_time = notes[i + 1][0]
                    if end_time - start_time >= 0.2:
                        note = pretty_midi.Note(80, pitch, start_time, end_time)
                        instrument.notes.append(note)
            
            midi_data.instruments.append(instrument)
            output_file = output_dir / f"{track_name}.mid"
            midi_data.write(str(output_file))
            
            print(f"         ベース変換: {len(instrument.notes)}個の音符")
            return str(output_file)
            
        except Exception as e:
            print(f"         ベース変換エラー: {e}")
            return None
    
    def merge_midi_tracks(self, midi_files: Dict[str, str], output_file: str = None, include_tracks: List[str] = None) -> str:
        """MIDIファイルを統合"""
        if output_file is None:
            first_file = Path(next(iter(midi_files.values())))
            output_file = first_file.parent / "merged_composition.mid"
        
        print(f"\\n🎵 MIDIトラック統合")
        print(f"   出力: {output_file}")
        
        if include_tracks:
            print(f"   対象トラック: {', '.join(include_tracks)}")
            filtered_files = {k: v for k, v in midi_files.items() if k in include_tracks}
        else:
            filtered_files = midi_files
        
        merged_midi = pretty_midi.PrettyMIDI()
        
        for track_name, midi_path in filtered_files.items():
            try:
                track_midi = pretty_midi.PrettyMIDI(midi_path)
                for instrument in track_midi.instruments:
                    merged_midi.instruments.append(instrument)
                print(f"   ✅ {track_name}: {len(track_midi.instruments)}楽器追加")
            except Exception as e:
                print(f"   ❌ {track_name}: エラー - {e}")
        
        merged_midi.write(str(output_file))
        
        total_notes = sum(len(inst.notes) for inst in merged_midi.instruments)
        end_time = merged_midi.get_end_time()
        
        print(f"   📊 統合結果:")
        print(f"     楽器数: {len(merged_midi.instruments)}")
        print(f"     総音符数: {total_notes}")
        print(f"     演奏時間: {end_time:.2f}秒")
        
        return str(output_file)
    
    def analyze_midi_files(self, midi_files: Dict[str, str]) -> Dict:
        """MIDI分析"""
        print(f"\\n📊 MIDI分析")
        analysis = {}
        
        for track_name, midi_path in midi_files.items():
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                note_count = sum(len(inst.notes) for inst in midi_data.instruments)
                duration = midi_data.get_end_time()
                instrument_count = len(midi_data.instruments)
                file_size = Path(midi_path).stat().st_size / 1024  # KB
                
                analysis[track_name] = {
                    "note_count": note_count,
                    "duration": duration,
                    "instrument_count": instrument_count,
                    "file_size_kb": file_size
                }
                
                print(f"   {track_name}:")
                print(f"     音符数: {note_count}")
                print(f"     長さ: {duration:.2f}秒")
                print(f"     楽器数: {instrument_count}")
                print(f"     ファイルサイズ: {file_size:.2f}KB")
                
            except Exception as e:
                print(f"   ❌ {track_name}: 分析エラー - {e}")
                
        return analysis

def find_enhanced_separation_results(base_dir: str = None) -> Dict[str, Dict[str, str]]:
    """
    高品質分離結果を検索
    
    Args:
        base_dir: 検索ベースディレクトリ
        
    Returns:
        {model_name: {track: file_path}} 形式の辞書
    """
    if base_dir is None:
        base_dir = Path.home() / "ドキュメント" / "conversion_music"
    
    base_path = Path(base_dir)
    results = {}
    
    # 標準的な分離結果ディレクトリを検索
    search_patterns = [
        "input_separated_enhanced",
        "*_separated_enhanced", 
        "*_separated",
        "*_comparison"
    ]
    
    for pattern in search_patterns:
        for sep_dir in base_path.glob(pattern):
            if not sep_dir.is_dir():
                continue
                
            print(f"🔍 検索中: {sep_dir.name}")
            
            # モデル別ディレクトリを検索
            for model_dir in sep_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                print(f"   モデル: {model_name}")
                
                # 音声ファイルを検索
                audio_files = find_audio_files_in_model_dir(model_dir)
                
                if audio_files:
                    results[f"{sep_dir.name}/{model_name}"] = audio_files
                    print(f"     ✅ ファイル数: {len(audio_files)}")
                    for track, file_path in audio_files.items():
                        print(f"       {track}: {Path(file_path).name}")
    
    return results

def find_audio_files_in_model_dir(model_dir: Path) -> Dict[str, str]:
    """モデルディレクトリ内の音声ファイルを検索"""
    audio_files = {}
    
    # 複数の階層構造に対応
    search_dirs = [
        model_dir,  # 直接
        model_dir / "input",  # htdemucs/input
        model_dir / "input_preprocessed",  # htdemucs_ft/input_preprocessed
    ]
    
    # ディレクトリ内のサブディレクトリも検索
    for subdir in model_dir.iterdir():
        if subdir.is_dir():
            search_dirs.append(subdir)
    
    track_names = ["vocals", "drums", "bass", "other", "piano", "guitar"]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        for track in track_names:
            for ext in [".wav", ".mp3", ".flac"]:
                audio_file = search_dir / f"{track}{ext}"
                if audio_file.exists() and track not in audio_files:
                    audio_files[track] = str(audio_file)
    
    return audio_files

def convert_enhanced_separation_to_midi(separation_results: Dict[str, Dict[str, str]], 
                                      output_base_dir: str = None,
                                      quantization_settings: QuantizationSettings = None) -> Dict[str, Dict]:
    """
    高品質分離結果をMIDI変換
    
    Args:
        separation_results: 分離結果辞書
        output_base_dir: 出力ベースディレクトリ
        
    Returns:
        MIDI変換結果辞書
    """
    if output_base_dir is None:
        output_base_dir = Path.cwd() / "enhanced_midi_results"
    
    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🎼 高品質分離MIDI変換開始")
    print(f"   出力ベース: {output_base_dir}")
    print(f"   対象数: {len(separation_results)}")
    
    conversion_results = {}
    
    for model_key, audio_files in separation_results.items():
        print(f"\n--- {model_key} MIDI変換 ---")
        
        # 出力ディレクトリ設定
        safe_model_key = model_key.replace("/", "_").replace("\\", "_")
        model_output_dir = output_base_path / f"{safe_model_key}_midi"
        
        try:
            # 元音源パスを取得（BPM検出用）
            work_dir = Path.home() / "ドキュメント" / "conversion_music"
            original_audio = work_dir / "input.m4a"
            
            # MIDI変換実行（元音源パスを渡す）
            converter = MIDIConverter(quantization_settings, str(original_audio) if original_audio.exists() else None)
            midi_files = converter.convert_separated_tracks(audio_files, str(model_output_dir))
            
            if midi_files:
                # MIDI分析
                analysis = converter.analyze_midi_files(midi_files)
                
                # 統合MIDI作成（全トラック）
                merged_midi_all = model_output_dir / "merged_composition_all.mid"
                converter.merge_midi_tracks(midi_files, str(merged_midi_all))
                
                # ボーカル+その他のみの統合MIDI作成
                merged_midi_vocal_other = model_output_dir / "merged_composition_vocal_other.mid"
                converter.merge_midi_tracks(midi_files, str(merged_midi_vocal_other), 
                                          include_tracks=["vocals", "other"])
                
                conversion_results[model_key] = {
                    "midi_files": midi_files,
                    "analysis": analysis,
                    "merged_midi_all": str(merged_midi_all),
                    "merged_midi_vocal_other": str(merged_midi_vocal_other),
                    "output_dir": str(model_output_dir)
                }
                
                print(f"   ✅ 変換完了: {len(midi_files)}ファイル")
                
            else:
                print(f"   ❌ 変換失敗")
                conversion_results[model_key] = {"error": "MIDI変換失敗"}
                
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            conversion_results[model_key] = {"error": str(e)}
    
    return conversion_results

def create_conversion_comparison_report(results: Dict[str, Dict], output_dir: str):
    """MIDI変換比較レポート作成"""
    print(f"\n📊 MIDI変換比較レポート生成")
    
    # 成功した結果のみ抽出
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        print("❌ 有効な結果がありません")
        return
    
    # レポートデータ準備
    report_data = {
        "timestamp": "2025-08-14T23:58:00",
        "total_conversions": len(results),
        "successful_conversions": len(valid_results),
        "results": {}
    }
    
    # 各モデルの結果を分析
    for model_key, result in valid_results.items():
        if "analysis" in result:
            analysis = result["analysis"]
            
            # 統計計算
            total_notes = sum(track.get("note_count", 0) for track in analysis.values())
            avg_duration = sum(track.get("duration", 0) for track in analysis.values()) / len(analysis)
            
            report_data["results"][model_key] = {
                "midi_files": len(result["midi_files"]),
                "total_notes": total_notes,
                "average_duration": avg_duration,
                "tracks": list(analysis.keys()),
                "merged_midi_all": result["merged_midi_all"],
                "merged_midi_vocal_other": result["merged_midi_vocal_other"]
            }
    
    # JSON保存
    report_file = Path(output_dir) / "midi_conversion_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # 結果表示
    print(f"📄 レポート保存: {report_file.name}")
    print(f"\n📊 変換結果サマリー:")
    print(f"   成功: {len(valid_results)}/{len(results)}")
    
    for model_key, model_data in report_data["results"].items():
        print(f"   {model_key}:")
        print(f"     音符数: {model_data['total_notes']}")
        print(f"     平均時間: {model_data['average_duration']:.1f}秒")
        print(f"     統合MIDI（全）: {Path(model_data['merged_midi_all']).name}")
        print(f"     統合MIDI（vocal+other）: {Path(model_data['merged_midi_vocal_other']).name}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="高品質音源分離結果からMIDI変換")
    parser.add_argument("--input-dir", "-i", type=str, help="入力ディレクトリ")
    parser.add_argument("--output-dir", "-o", type=str, help="出力ディレクトリ") 
    parser.add_argument("--model", "-m", type=str, help="特定モデルのみ処理")
    parser.add_argument("--quantize", action="store_true", help="MIDI量子化を有効化")
    parser.add_argument("--grid", type=str, default="thirtysecond", choices=["quarter", "eighth", "sixteenth", "thirtysecond"], help="量子化グリッド解像度")
    parser.add_argument("--strength", type=float, default=0.8, help="量子化強度 (0.0-1.0)")
    parser.add_argument("--bpm", type=float, help="手動BPM指定（自動検出より優先）")
    
    args = parser.parse_args()
    
    print("🎼 高品質分離 → MIDI変換システム")
    print("=" * 50)
    
    # 作業ディレクトリ
    work_dir = Path.home() / "ドキュメント" / "conversion_music"
    print(f"📁 作業ディレクトリ: {work_dir}")
    
    # 高品質分離結果検索
    separation_results = find_enhanced_separation_results(args.input_dir or str(work_dir))
    
    if not separation_results:
        print("\n❌ 高品質分離結果が見つかりません")
        print("先に enhanced_music_separation.py を実行してください:")
        print("python enhanced_music_separation.py --model htdemucs_ft --quality high --input input.m4a")
        return
    
    print(f"\n🎵 見つかった分離結果: {len(separation_results)}")
    
    # 特定モデルフィルタリング
    if args.model:
        filtered_results = {k: v for k, v in separation_results.items() 
                          if args.model in k.lower()}
        if filtered_results:
            separation_results = filtered_results
            print(f"🔍 フィルタ適用: {args.model} → {len(separation_results)}件")
        else:
            print(f"⚠️ 指定モデル '{args.model}' が見つかりません")
    
    # 量子化設定
    quantization_settings = None
    if args.quantize:
        quantization_settings = QuantizationSettings(
            enabled=True,
            grid_resolution=args.grid,
            strength=args.strength,
            auto_detect_bpm=args.bpm is None,
            manual_bpm=args.bpm
        )
        print(f"🎯 量子化設定: {args.grid}音符, 強度{args.strength}, BPM{args.bpm or '自動検出'}")
    
    # MIDI変換実行
    output_dir = args.output_dir or str(work_dir / "enhanced_midi_results")
    conversion_results = convert_enhanced_separation_to_midi(separation_results, output_dir, quantization_settings)
    
    # 比較レポート生成
    create_conversion_comparison_report(conversion_results, output_dir)
    
    print(f"\n✅ 高品質分離MIDI変換完了!")
    print(f"   結果ディレクトリ: {output_dir}")
    
    # 成功した結果の詳細表示
    successful_results = [k for k, v in conversion_results.items() if "error" not in v]
    if successful_results:
        print(f"\n🎵 生成されたMIDIファイル:")
        for model_key in successful_results:
            result = conversion_results[model_key]
            print(f"   {model_key}:")
            print(f"     📁 ディレクトリ: {Path(result['output_dir']).name}")
            print(f"     🎼 統合MIDI（全）: {Path(result['merged_midi_all']).name}")
            print(f"     🎵 統合MIDI（vocal+other）: {Path(result['merged_midi_vocal_other']).name}")

if __name__ == "__main__":
    main()