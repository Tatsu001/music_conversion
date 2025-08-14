#!/usr/bin/env python3
"""
MIDI精度チューニング実行 & 最適化されたMIDI変換スクリプト
チューニング結果を元に改良版のMIDI変換を実行
"""

import os
import sys
import time
import json
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from basic_pitch.inference import predict_and_save, predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False


class OptimizedMIDIConverter:
    """最適化されたMIDI変換器"""
    
    def __init__(self, tuning_results: Dict = None):
        self.sample_rate = 22050
        self.tuning_results = tuning_results or {}
        
        # デフォルト最適パラメータ（チューニング結果で上書き可能）
        self.optimal_params = {
            "vocals": {
                "fmin": 80, "fmax": 1000, "threshold": 80, 
                "min_duration": 0.08, "note_tolerance": 1
            },
            "drums": {
                "delta": 0.15, "wait": 8, "pre_max": 2, "post_max": 2,
                "sensitivity": "high"
            },
            "bass": {
                "fmin": 40, "fmax": 300, "threshold": 50, 
                "min_duration": 0.2, "note_tolerance": 2
            },
            "other": {
                "fmin": 100, "fmax": 1200, "threshold": 100, 
                "min_duration": 0.05, "note_tolerance": 1
            }
        }
        
        print("🎯 OptimizedMIDIConverter初期化完了")
        if tuning_results:
            print("   チューニング結果適用済み")
    
    def load_tuning_results(self, results_file: str):
        """チューニング結果読み込み"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.tuning_results = json.load(f)
            
            # 最適パラメータ更新
            for track_type in ["vocals", "drums", "bass", "other"]:
                if track_type in self.tuning_results:
                    best_config = self.tuning_results[track_type].get("best_config", {})
                    if best_config.get("name") != "none":
                        # チューニング結果から最適パラメータを抽出
                        all_results = self.tuning_results[track_type].get("all_results", {})
                        config_name = best_config["name"]
                        if config_name in all_results:
                            params = all_results[config_name].get("params", {})
                            if params:
                                self.optimal_params[track_type].update(params)
            
            print(f"✅ チューニング結果読み込み完了: {results_file}")
            
        except Exception as e:
            print(f"⚠️ チューニング結果読み込み失敗: {e}")
            print("   デフォルト最適パラメータを使用します")
    
    def convert_with_optimal_params(self, separated_files: Dict[str, str], output_dir: str = None) -> Dict[str, str]:
        """最適パラメータでMIDI変換"""
        if not separated_files:
            raise ValueError("分離ファイルが指定されていません")
        
        if output_dir is None:
            first_file = Path(next(iter(separated_files.values())))
            output_dir = first_file.parent.parent / "optimized_midi_tracks"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎼 最適化MIDI変換開始")
        print(f"   出力ディレクトリ: {output_dir}")
        
        midi_files = {}
        conversion_stats = {}
        
        for track_name, audio_file in separated_files.items():
            if not os.path.exists(audio_file):
                print(f"   ⚠️ {track_name}: ファイルが見つかりません")
                continue
            
            print(f"\n   🔄 {track_name} 最適化変換中...")
            
            try:
                start_time = time.time()
                
                if track_name in ["vocals", "other"]:
                    # メロディトラック: 最適化+basic-pitch併用
                    midi_file, stats = self._convert_melody_optimized(
                        audio_file, output_path, track_name
                    )
                elif track_name == "drums":
                    # ドラムトラック: 最適化
                    midi_file, stats = self._convert_drums_optimized(
                        audio_file, output_path, track_name
                    )
                elif track_name == "bass":
                    # ベーストラック: 最適化
                    midi_file, stats = self._convert_bass_optimized(
                        audio_file, output_path, track_name
                    )
                else:
                    # その他: 標準最適化
                    midi_file, stats = self._convert_melody_optimized(
                        audio_file, output_path, track_name
                    )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                if midi_file:
                    midi_files[track_name] = midi_file
                    conversion_stats[track_name] = {
                        **stats,
                        "processing_time": processing_time
                    }
                    print(f"      ✅ 変換完了: {Path(midi_file).name}")
                    print(f"         音符数: {stats.get('note_count', 0)}")
                    print(f"         処理時間: {processing_time:.2f}秒")
                else:
                    print(f"      ❌ 変換失敗")
                    
            except Exception as e:
                print(f"      ❌ エラー: {e}")
        
        # 変換統計表示
        self._display_conversion_summary(conversion_stats)
        
        return midi_files
    
    def _convert_melody_optimized(self, audio_file: str, output_dir: Path, track_name: str) -> Tuple[Optional[str], Dict]:
        """最適化メロディ変換（basic-pitch優先）"""
        stats = {"method": "unknown", "note_count": 0}
        
        # 1. basic-pitch試行（利用可能なら）
        if BASIC_PITCH_AVAILABLE:
            try:
                print(f"         basic-pitch試行...")
                midi_file, bp_stats = self._convert_with_basic_pitch(
                    audio_file, output_dir, track_name
                )
                if midi_file and bp_stats.get("note_count", 0) > 10:
                    stats = {**bp_stats, "method": "basic-pitch"}
                    return midi_file, stats
                else:
                    print(f"         basic-pitch結果不十分、librosa最適化に切替")
            except Exception as e:
                print(f"         basic-pitch失敗: {e}")
        
        # 2. librosa最適化
        try:
            print(f"         librosa最適化実行...")
            midi_file, lib_stats = self._convert_with_librosa_optimized(
                audio_file, output_dir, track_name
            )
            stats = {**lib_stats, "method": "librosa_optimized"}
            return midi_file, stats
            
        except Exception as e:
            print(f"         librosa最適化失敗: {e}")
            return None, {"method": "failed", "error": str(e)}
    
    def _convert_with_basic_pitch(self, audio_file: str, output_dir: Path, track_name: str) -> Tuple[Optional[str], Dict]:
        """basic-pitch変換（修正済みAPI）"""
        try:
            output_file = output_dir / f"{track_name}_bp.mid"
            
            # basic-pitchで予測
            model_output, midi_data, note_events = predict(
                str(audio_file),
                ICASSP_2022_MODEL_PATH
            )
            
            # MIDIファイル作成
            pm_midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)  # Piano
            
            for note_event in note_events:
                pitch, start_time, end_time, confidence = note_event
                
                # 信頼度フィルタ（0.5以上）
                if confidence >= 0.5:
                    note = pretty_midi.Note(
                        velocity=int(80 + confidence * 40),  # 信頼度に応じたベロシティ
                        pitch=int(pitch),
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
            
            pm_midi.instruments.append(instrument)
            pm_midi.write(str(output_file))
            
            stats = {
                "note_count": len(instrument.notes),
                "confidence_filtered": sum(1 for ne in note_events if ne[3] >= 0.5),
                "total_detected": len(note_events),
                "avg_confidence": np.mean([ne[3] for ne in note_events]) if note_events else 0
            }
            
            return str(output_file), stats
            
        except Exception as e:
            raise Exception(f"basic-pitch変換エラー: {e}")
    
    def _convert_with_librosa_optimized(self, audio_file: str, output_dir: Path, track_name: str) -> Tuple[Optional[str], Dict]:
        """librosa最適化変換"""
        # 最適パラメータ取得
        params = self.optimal_params.get(track_name, self.optimal_params["vocals"])
        
        # 音声読み込み
        y, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # 前処理（ノイズ除去）
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # ピッチ検出（最適パラメータ適用）
        f0 = librosa.yin(y, 
                        fmin=params["fmin"], 
                        fmax=params["fmax"], 
                        sr=sr)
        
        # 時間軸
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
        
        # スムージング（新機能）
        f0_smooth = self._smooth_pitch(f0, window_size=5)
        
        # MIDIファイル作成
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        # 音符生成（改良版）
        notes = self._generate_notes_optimized(
            f0_smooth, times, params
        )
        
        for note_info in notes:
            note = pretty_midi.Note(
                velocity=note_info["velocity"],
                pitch=note_info["pitch"],
                start=note_info["start"],
                end=note_info["end"]
            )
            instrument.notes.append(note)
        
        midi_data.instruments.append(instrument)
        
        # ファイル保存
        output_file = output_dir / f"{track_name}_opt.mid"
        midi_data.write(str(output_file))
        
        stats = {
            "note_count": len(notes),
            "avg_duration": np.mean([n["end"] - n["start"] for n in notes]) if notes else 0,
            "pitch_range": max([n["pitch"] for n in notes]) - min([n["pitch"] for n in notes]) if notes else 0
        }
        
        return str(output_file), stats
    
    def _smooth_pitch(self, f0: np.ndarray, window_size: int = 5) -> np.ndarray:
        """ピッチスムージング（新機能）"""
        # 移動平均でスムージング
        kernel = np.ones(window_size) / window_size
        f0_padded = np.pad(f0, (window_size//2, window_size//2), mode='edge')
        f0_smooth = np.convolve(f0_padded, kernel, mode='valid')
        
        # NaN値は元の値を保持
        nan_mask = np.isnan(f0)
        f0_smooth[nan_mask] = f0[nan_mask]
        
        return f0_smooth
    
    def _generate_notes_optimized(self, f0: np.ndarray, times: np.ndarray, params: Dict) -> List[Dict]:
        """最適化音符生成"""
        notes = []
        current_note = None
        note_start = None
        note_velocity = 80
        
        for i, (time, freq) in enumerate(zip(times, f0)):
            if freq > params["threshold"] and not np.isnan(freq):
                midi_note = int(librosa.hz_to_midi(freq))
                
                # 音域制限
                if params.get("min_pitch"):
                    midi_note = max(params["min_pitch"], midi_note)
                if params.get("max_pitch"):
                    midi_note = min(params["max_pitch"], midi_note)
                
                tolerance = params.get("note_tolerance", 1)
                
                if current_note is None or abs(midi_note - current_note) > tolerance:
                    # 前の音符を確定
                    if current_note is not None and note_start is not None:
                        duration = time - note_start
                        if duration >= params["min_duration"]:
                            notes.append({
                                "pitch": current_note,
                                "start": note_start,
                                "end": time,
                                "velocity": note_velocity
                            })
                    
                    # 新しい音符開始
                    current_note = midi_note
                    note_start = time
                    # ベロシティを周波数に基づいて調整
                    note_velocity = min(127, max(40, int(80 + freq / 10)))
            else:
                # 音符終了
                if current_note is not None and note_start is not None:
                    duration = time - note_start
                    if duration >= params["min_duration"]:
                        notes.append({
                            "pitch": current_note,
                            "start": note_start,
                            "end": time,
                            "velocity": note_velocity
                        })
                    current_note = None
                    note_start = None
        
        # 最後の音符処理
        if current_note is not None and note_start is not None:
            notes.append({
                "pitch": current_note,
                "start": note_start,
                "end": times[-1],
                "velocity": note_velocity
            })
        
        return notes
    
    def _convert_drums_optimized(self, audio_file: str, output_dir: Path, track_name: str) -> Tuple[Optional[str], Dict]:
        """最適化ドラム変換"""
        params = self.optimal_params.get("drums", {})
        
        # 音声読み込み
        y, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # オンセット検出（最適パラメータ）
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, units='frames',
            hop_length=512,
            backtrack=True,
            pre_max=params.get("pre_max", 2),
            post_max=params.get("post_max", 2),
            pre_avg=3,
            post_avg=5,
            delta=params.get("delta", 0.15),
            wait=params.get("wait", 8)
        )
        
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # MIDIファイル作成
        midi_data = pretty_midi.PrettyMIDI()
        drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
        
        # 改良ドラム分類
        hits = self._classify_drums_improved(y, sr, onset_times)
        
        for hit in hits:
            note = pretty_midi.Note(
                velocity=hit["velocity"],
                pitch=hit["midi_note"],
                start=hit["time"],
                end=hit["time"] + 0.1
            )
            drum_instrument.notes.append(note)
        
        midi_data.instruments.append(drum_instrument)
        
        # ファイル保存
        output_file = output_dir / f"{track_name}_opt.mid"
        midi_data.write(str(output_file))
        
        stats = {
            "note_count": len(hits),
            "kick_count": sum(1 for h in hits if h["type"] == "kick"),
            "snare_count": sum(1 for h in hits if h["type"] == "snare"),
            "hihat_count": sum(1 for h in hits if h["type"] == "hihat")
        }
        
        return str(output_file), stats
    
    def _classify_drums_improved(self, y: np.ndarray, sr: int, onset_times: np.ndarray) -> List[Dict]:
        """改良ドラム分類"""
        hits = []
        
        for onset_time in onset_times:
            frame_idx = librosa.time_to_frames(onset_time, sr=sr, hop_length=512)
            
            # より大きな分析窓
            window_size = 4096
            start_sample = max(0, frame_idx * 512 - window_size // 2)
            end_sample = min(len(y), start_sample + window_size)
            window = y[start_sample:end_sample]
            
            if len(window) >= 1024:
                # スペクトル分析
                spectrum = np.abs(np.fft.fft(window, n=4096))
                freqs = np.fft.fftfreq(4096, 1/sr)
                
                # 周波数帯域分割（改良版）
                low_band = spectrum[0:80]       # 0-800Hz  (キック)
                mid_low = spectrum[80:200]      # 800-2000Hz (スネア低域)
                mid_high = spectrum[200:400]    # 2000-4000Hz (スネア高域)
                high_band = spectrum[400:800]   # 4000-8000Hz (ハイハット)
                
                # エネルギー計算
                low_energy = np.sum(low_band)
                mid_energy = np.sum(mid_low) + np.sum(mid_high)
                high_energy = np.sum(high_band)
                total_energy = low_energy + mid_energy + high_energy
                
                if total_energy > 0:
                    # エネルギー比率
                    low_ratio = low_energy / total_energy
                    mid_ratio = mid_energy / total_energy
                    high_ratio = high_energy / total_energy
                    
                    # 改良分類基準
                    if low_ratio > 0.55 and low_energy > high_energy:
                        drum_type = "kick"
                        midi_note = 36
                        velocity = min(127, int(90 + low_ratio * 37))
                    elif high_ratio > 0.45 and high_energy > low_energy:
                        drum_type = "hihat"
                        midi_note = 42
                        velocity = min(127, int(70 + high_ratio * 57))
                    else:
                        drum_type = "snare"
                        midi_note = 38
                        velocity = min(127, int(80 + mid_ratio * 47))
                    
                    hits.append({
                        "time": onset_time,
                        "type": drum_type,
                        "midi_note": midi_note,
                        "velocity": velocity,
                        "energy_ratios": {
                            "low": low_ratio,
                            "mid": mid_ratio,
                            "high": high_ratio
                        }
                    })
        
        return hits
    
    def _convert_bass_optimized(self, audio_file: str, output_dir: Path, track_name: str) -> Tuple[Optional[str], Dict]:
        """最適化ベース変換"""
        params = self.optimal_params.get("bass", {})
        
        # 音声読み込み
        y, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # 低域フィルタ強化
        y_filtered = librosa.effects.preemphasis(y, coef=0.0)
        
        # 低域強調フィルタ追加
        from scipy import signal
        nyquist = sr / 2
        low_cutoff = 300 / nyquist  # 300Hz以下を強調
        b, a = signal.butter(4, low_cutoff, btype='low')
        y_bass_enhanced = signal.filtfilt(b, a, y_filtered)
        
        # ピッチ検出（低域特化）
        f0 = librosa.yin(y_bass_enhanced, 
                        fmin=params.get("fmin", 40), 
                        fmax=params.get("fmax", 300), 
                        sr=sr)
        
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
        
        # MIDIファイル作成
        midi_data = pretty_midi.PrettyMIDI()
        bass_instrument = pretty_midi.Instrument(program=33)  # Electric Bass
        
        # ベース専用音符生成
        notes = self._generate_bass_notes_optimized(f0, times, params)
        
        for note_info in notes:
            note = pretty_midi.Note(
                velocity=note_info["velocity"],
                pitch=note_info["pitch"],
                start=note_info["start"],
                end=note_info["end"]
            )
            bass_instrument.notes.append(note)
        
        midi_data.instruments.append(bass_instrument)
        
        # ファイル保存
        output_file = output_dir / f"{track_name}_opt.mid"
        midi_data.write(str(output_file))
        
        stats = {
            "note_count": len(notes),
            "avg_pitch": np.mean([n["pitch"] for n in notes]) if notes else 0,
            "avg_duration": np.mean([n["end"] - n["start"] for n in notes]) if notes else 0
        }
        
        return str(output_file), stats
    
    def _generate_bass_notes_optimized(self, f0: np.ndarray, times: np.ndarray, params: Dict) -> List[Dict]:
        """ベース専用最適化音符生成"""
        notes = []
        current_note = None
        note_start = None
        
        # ベース専用パラメータ
        threshold = params.get("threshold", 50)
        min_duration = params.get("min_duration", 0.2)
        note_tolerance = params.get("note_tolerance", 2)
        
        for i, (time, freq) in enumerate(zip(times, f0)):
            if freq > threshold and not np.isnan(freq):
                midi_note = int(librosa.hz_to_midi(freq))
                
                # ベース音域制限強化 (E1-E4: 28-64)
                midi_note = max(28, min(64, midi_note))
                
                if current_note is None or abs(midi_note - current_note) > note_tolerance:
                    # 前の音符確定
                    if current_note is not None and note_start is not None:
                        duration = time - note_start
                        if duration >= min_duration:
                            # ベロシティ調整（低音ほど強く）
                            velocity = max(70, min(120, int(120 - (current_note - 28) * 1.5)))
                            notes.append({
                                "pitch": current_note,
                                "start": note_start,
                                "end": time,
                                "velocity": velocity
                            })
                    
                    current_note = midi_note
                    note_start = time
            else:
                # 音符終了
                if current_note is not None and note_start is not None:
                    duration = time - note_start
                    if duration >= min_duration:
                        velocity = max(70, min(120, int(120 - (current_note - 28) * 1.5)))
                        notes.append({
                            "pitch": current_note,
                            "start": note_start,
                            "end": time,
                            "velocity": velocity
                        })
                    current_note = None
                    note_start = None
        
        # 最後の音符処理
        if current_note is not None and note_start is not None:
            velocity = max(70, min(120, int(120 - (current_note - 28) * 1.5)))
            notes.append({
                "pitch": current_note,
                "start": note_start,
                "end": times[-1],
                "velocity": velocity
            })
        
        return notes
    
    def _display_conversion_summary(self, stats: Dict):
        """変換統計表示"""
        print(f"\n📊 最適化変換統計")
        
        total_notes = 0
        total_time = 0
        
        for track, track_stats in stats.items():
            notes = track_stats.get("note_count", 0)
            method = track_stats.get("method", "unknown")
            proc_time = track_stats.get("processing_time", 0)
            
            total_notes += notes
            total_time += proc_time
            
            print(f"   {track}:")
            print(f"     音符数: {notes}")
            print(f"     手法: {method}")
            print(f"     処理時間: {proc_time:.2f}秒")
            
            # トラック固有統計
            if "avg_confidence" in track_stats:
                print(f"     平均信頼度: {track_stats['avg_confidence']:.3f}")
            if "kick_count" in track_stats:
                print(f"     キック: {track_stats['kick_count']}")
                print(f"     スネア: {track_stats['snare_count']}")
                print(f"     ハイハット: {track_stats['hihat_count']}")
            if "avg_pitch" in track_stats:
                print(f"     平均ピッチ: {track_stats['avg_pitch']:.1f}")
        
        print(f"\n   📈 総計:")
        print(f"     総音符数: {total_notes}")
        print(f"     総処理時間: {total_time:.2f}秒")
        print(f"     平均処理速度: {total_notes/max(total_time, 0.1):.1f}音符/秒")
    
    def merge_optimized_tracks(self, midi_files: Dict[str, str], output_file: str = None) -> Optional[str]:
        """最適化トラック統合"""
        if not midi_files:
            return None
        
        if output_file is None:
            first_file = Path(next(iter(midi_files.values())))
            output_file = first_file.parent / "optimized_merged_composition.mid"
        
        print(f"\n🎵 最適化トラック統合")
        print(f"   出力: {output_file}")
        
        try:
            merged_midi = pretty_midi.PrettyMIDI()
            
            # トラック順序最適化
            track_order = ["bass", "drums", "other", "vocals"]
            
            for track_name in track_order:
                if track_name in midi_files:
                    midi_file = midi_files[track_name]
                    if os.path.exists(midi_file):
                        try:
                            track_midi = pretty_midi.PrettyMIDI(midi_file)
                            
                            for instrument in track_midi.instruments:
                                # トラック名設定
                                instrument.name = f"Optimized_{track_name}"
                                merged_midi.instruments.append(instrument)
                            
                            print(f"   ✅ {track_name}: 統合完了")
                            
                        except Exception as e:
                            print(f"   ❌ {track_name}: 統合エラー - {e}")
            
            # 統合ファイル保存
            merged_midi.write(str(output_file))
            
            # 統計
            total_instruments = len(merged_midi.instruments)
            total_notes = sum(len(inst.notes) for inst in merged_midi.instruments)
            duration = merged_midi.get_end_time()
            
            print(f"   📊 統合結果:")
            print(f"     楽器数: {total_instruments}")
            print(f"     総音符数: {total_notes}")
            print(f"     演奏時間: {duration:.2f}秒")
            print(f"     音符密度: {total_notes/max(duration, 1):.1f}音符/秒")
            
            return str(output_file)
            
        except Exception as e:
            print(f"   ❌ 統合エラー: {e}")
            return None


def run_complete_optimization_pipeline():
    """完全最適化パイプライン実行"""
    print("🚀 完全最適化パイプライン開始")
    print("=" * 60)
    
    # 作業ディレクトリ確認
    work_dir = Path.home() / "ドキュメント" / "conversion_music"
    
    if not work_dir.exists():
        print(f"❌ 作業ディレクトリが見つかりません: {work_dir}")
        return
    
    # 分離済みファイル検索
    separated_dirs = list(work_dir.glob("*_separated"))
    
    if not separated_dirs:
        print("❌ 分離済みファイルが見つかりません")
        return
    
    separated_dir = separated_dirs[0]
    track_dir = separated_dir / "htdemucs" / "input"
    
    if not track_dir.exists():
        print(f"❌ トラックディレクトリが見つかりません: {track_dir}")
        return
    
    print(f"📁 処理対象: {track_dir}")
    
    # 分離ファイル収集
    separated_files = {}
    for track_name in ["vocals", "drums", "bass", "other"]:
        track_file = track_dir / f"{track_name}.wav"
        if track_file.exists():
            separated_files[track_name] = str(track_file)
            print(f"   ✅ {track_name}: {track_file.name}")
        else:
            print(f"   ⚠️ {track_name}: ファイルなし")
    
    if not separated_files:
        print("❌ 有効な分離ファイルが見つかりません")
        return
    
    # Step 1: チューニング結果確認
    tuning_results_file = work_dir / "tuning_results" / "tuning_results.json"
    
    converter = OptimizedMIDIConverter()
    
    if tuning_results_file.exists():
        print(f"\n📊 チューニング結果読み込み")
        converter.load_tuning_results(str(tuning_results_file))
    else:
        print(f"\n⚠️ チューニング結果なし、デフォルト最適パラメータ使用")
    
    # Step 2: 最適化MIDI変換実行
    print(f"\n🎼 最適化MIDI変換実行")
    start_time = time.time()
    
    try:
        midi_files = converter.convert_with_optimal_params(separated_files)
        
        if midi_files:
            # Step 3: 最適化トラック統合
            merged_file = converter.merge_optimized_tracks(midi_files)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n✅ 最適化パイプライン完了!")
            print(f"   総処理時間: {total_time:.2f}秒")
            print(f"   生成ファイル数: {len(midi_files)}")
            
            if merged_file:
                print(f"   統合ファイル: {Path(merged_file).name}")
                print(f"\n🎯 次のステップ: 音色変換 (Step 3)")
                print(f"   8bit変換またはオルゴール変換を実行してください")
            
            # 品質評価実行
            quality_report = evaluate_conversion_quality(midi_files, separated_files)
            print(f"\n📈 品質評価完了")
            
        else:
            print(f"❌ MIDIファイルが生成されませんでした")
            
    except Exception as e:
        print(f"❌ パイプラインエラー: {e}")
        import traceback
        traceback.print_exc()


def evaluate_conversion_quality(midi_files: Dict[str, str], original_files: Dict[str, str]) -> Dict:
    """変換品質評価"""
    print(f"\n📊 変換品質評価")
    
    quality_metrics = {}
    
    for track_name, midi_file in midi_files.items():
        if not os.path.exists(midi_file):
            continue
        
        original_file = original_files.get(track_name)
        if not original_file or not os.path.exists(original_file):
            continue
        
        print(f"   {track_name} 評価中...")
        
        try:
            # MIDI分析
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
            midi_duration = midi_data.get_end_time()
            
            # 原音声分析
            y, sr = librosa.load(original_file, sr=22050)
            audio_duration = len(y) / sr
            
            # 基本メトリクス
            metrics = {
                "midi_notes": total_notes,
                "midi_duration": midi_duration,
                "audio_duration": audio_duration,
                "duration_match": abs(midi_duration - audio_duration) / audio_duration,
                "note_density": total_notes / max(midi_duration, 1),
                "file_size_kb": os.path.getsize(midi_file) / 1024
            }
            
            # トラック固有評価
            if track_name in ["vocals", "other"]:
                # メロディ評価
                if midi_data.instruments:
                    pitches = [note.pitch for inst in midi_data.instruments for note in inst.notes]
                    if pitches:
                        metrics["pitch_range"] = max(pitches) - min(pitches)
                        metrics["avg_pitch"] = np.mean(pitches)
                        
                        # 音程変化分析
                        pitch_changes = [abs(pitches[i] - pitches[i-1]) for i in range(1, len(pitches))]
                        metrics["avg_pitch_change"] = np.mean(pitch_changes) if pitch_changes else 0
                        
            elif track_name == "drums":
                # ドラム評価
                if midi_data.instruments and midi_data.instruments[0].is_drum:
                    drum_notes = midi_data.instruments[0].notes
                    kick_notes = [n for n in drum_notes if n.pitch == 36]
                    snare_notes = [n for n in drum_notes if n.pitch == 38]
                    hihat_notes = [n for n in drum_notes if n.pitch == 42]
                    
                    metrics["kick_ratio"] = len(kick_notes) / max(len(drum_notes), 1)
                    metrics["snare_ratio"] = len(snare_notes) / max(len(drum_notes), 1)
                    metrics["hihat_ratio"] = len(hihat_notes) / max(len(drum_notes), 1)
                    metrics["rhythm_regularity"] = calculate_rhythm_regularity(drum_notes)
                    
            elif track_name == "bass":
                # ベース評価
                if midi_data.instruments:
                    bass_notes = [note for inst in midi_data.instruments for note in inst.notes]
                    if bass_notes:
                        pitches = [note.pitch for note in bass_notes]
                        durations = [note.end - note.start for note in bass_notes]
                        
                        metrics["avg_bass_pitch"] = np.mean(pitches)
                        metrics["bass_range"] = max(pitches) - min(pitches)
                        metrics["avg_note_duration"] = np.mean(durations)
                        metrics["bass_consistency"] = 1 - (np.std(pitches) / max(np.mean(pitches), 1))
            
            # 品質スコア計算
            quality_score = calculate_quality_score(metrics, track_name)
            metrics["quality_score"] = quality_score
            
            quality_metrics[track_name] = metrics
            
            print(f"     音符数: {total_notes}")
            print(f"     時間精度: {(1-metrics['duration_match'])*100:.1f}%")
            print(f"     品質スコア: {quality_score:.1f}/100")
            
        except Exception as e:
            print(f"     ❌ 評価エラー: {e}")
            quality_metrics[track_name] = {"error": str(e)}
    
    # 総合評価
    valid_scores = [m["quality_score"] for m in quality_metrics.values() 
                   if "quality_score" in m]
    
    if valid_scores:
        overall_score = np.mean(valid_scores)
        print(f"\n🎯 総合品質スコア: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            print("   🌟 優秀な変換品質!")
        elif overall_score >= 60:
            print("   ✅ 良好な変換品質")
        elif overall_score >= 40:
            print("   ⚠️ 改善の余地あり")
        else:
            print("   🔧 パラメータ再調整を推奨")
    
    return quality_metrics


def calculate_rhythm_regularity(drum_notes: List) -> float:
    """リズム規則性計算"""
    if len(drum_notes) < 3:
        return 0.0
    
    # 音符間隔計算
    intervals = []
    sorted_notes = sorted(drum_notes, key=lambda n: n.start)
    
    for i in range(1, len(sorted_notes)):
        interval = sorted_notes[i].start - sorted_notes[i-1].start
        intervals.append(interval)
    
    if not intervals:
        return 0.0
    
    # 間隔の一貫性（標準偏差の逆数）
    std_interval = np.std(intervals)
    mean_interval = np.mean(intervals)
    
    if mean_interval == 0:
        return 0.0
    
    # 規則性スコア（0-1）
    regularity = max(0, 1 - (std_interval / mean_interval))
    return regularity


def calculate_quality_score(metrics: Dict, track_type: str) -> float:
    """品質スコア計算"""
    score = 0.0
    
    # 基本スコア（時間精度）
    duration_match = metrics.get("duration_match", 1.0)
    time_score = max(0, (1 - duration_match) * 30)
    score += time_score
    
    # 音符密度スコア
    note_density = metrics.get("note_density", 0)
    if track_type in ["vocals", "other"]:
        # メロディ: 1-8音符/秒が理想
        if 1 <= note_density <= 8:
            density_score = 25
        else:
            density_score = max(0, 25 - abs(note_density - 4) * 3)
    elif track_type == "drums":
        # ドラム: 0.5-5ヒット/秒が理想
        if 0.5 <= note_density <= 5:
            density_score = 25
        else:
            density_score = max(0, 25 - abs(note_density - 2.5) * 5)
    elif track_type == "bass":
        # ベース: 0.3-3音符/秒が理想
        if 0.3 <= note_density <= 3:
            density_score = 25
        else:
            density_score = max(0, 25 - abs(note_density - 1.5) * 8)
    else:
        density_score = 15
    
    score += density_score
    
    # トラック固有スコア
    if track_type in ["vocals", "other"]:
        # 音域スコア
        pitch_range = metrics.get("pitch_range", 0)
        if pitch_range >= 24:  # 2オクターブ以上
            range_score = 20
        elif pitch_range >= 12:  # 1オクターブ以上
            range_score = 15
        else:
            range_score = pitch_range * 15 / 12
        score += range_score
        
        # 音程変化スコア
        avg_change = metrics.get("avg_pitch_change", 0)
        if 1 <= avg_change <= 5:  # 適度な音程変化
            change_score = 15
        else:
            change_score = max(0, 15 - abs(avg_change - 3) * 3)
        score += change_score
        
    elif track_type == "drums":
        # ドラム分類バランススコア
        kick_ratio = metrics.get("kick_ratio", 0)
        snare_ratio = metrics.get("snare_ratio", 0)
        hihat_ratio = metrics.get("hihat_ratio", 0)
        
        balance_score = 0
        if 0.15 <= kick_ratio <= 0.45:
            balance_score += 8
        if 0.10 <= snare_ratio <= 0.40:
            balance_score += 8
        if 0.25 <= hihat_ratio <= 0.70:
            balance_score += 9
        
        score += balance_score
        
        # リズム規則性スコア
        regularity = metrics.get("rhythm_regularity", 0)
        score += regularity * 10
        
    elif track_type == "bass":
        # ベース一貫性スコア
        consistency = metrics.get("bass_consistency", 0)
        score += consistency * 20
        
        # 音符長スコア
        avg_duration = metrics.get("avg_note_duration", 0)
        if 0.2 <= avg_duration <= 1.0:
            duration_score = 15
        else:
            duration_score = max(0, 15 - abs(avg_duration - 0.6) * 10)
        score += duration_score
    
    return min(100, max(0, score))


def main():
    """メイン実行関数"""
    print("🎯 MIDI変換最適化 & 実行")
    print("=" * 50)
    
    # オプション選択
    print("\n実行オプション:")
    print("1. チューニング実行 (パラメータ最適化)")
    print("2. 最適化変換実行 (チューニング結果適用)")
    print("3. 完全パイプライン (チューニング + 変換)")
    
    try:
        choice = input("\n選択してください (1-3): ").strip()
        
        if choice == "1":
            # チューニング実行
            print("\n🔬 チューニング実行...")
            os.system("python midi_tuning.py")
            
        elif choice == "2":
            # 最適化変換のみ実行
            run_complete_optimization_pipeline()
            
        elif choice == "3":
            # 完全パイプライン実行
            print("\n🔬 Step 1: チューニング実行...")
            os.system("python midi_tuning.py")
            
            print("\n🎼 Step 2: 最適化変換実行...")
            run_complete_optimization_pipeline()
            
        else:
            print("❌ 無効な選択です")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 処理を中断しました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()