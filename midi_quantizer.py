#!/usr/bin/env python3
"""
MIDI量子化エンジン
音符のタイミングをリズムグリッドに合わせる機能

Features:
- BPM自動検出
- 4分音符・8分音符・16分音符・32分音符グリッド量子化
- 量子化強度調整（0.0-1.0）
- トラック別設定対応
- スイング量子化対応
"""

import numpy as np
import librosa
import pretty_midi
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class QuantizationSettings:
    """量子化設定データクラス"""
    enabled: bool = True
    grid_resolution: str = "thirtysecond"  # quarter, eighth, sixteenth, thirtysecond
    strength: float = 0.8           # 0.0-1.0
    auto_detect_bpm: bool = True
    manual_bpm: Optional[float] = None
    swing_ratio: float = 0.0        # 0.0=ストレート, 0.67=強スイング
    track_settings: Dict[str, Dict] = None

class MIDIQuantizer:
    """MIDI量子化エンジンクラス"""
    
    # グリッド解像度定義（4分音符を1.0とした比率）
    GRID_RESOLUTIONS = {
        "quarter": 1.0,      # 4分音符
        "eighth": 0.5,       # 8分音符  
        "sixteenth": 0.25,   # 16分音符
        "thirtysecond": 0.125, # 32分音符
        "triplet": 1.0/3,    # 3連符
    }
    
    def __init__(self, settings: QuantizationSettings = None):
        """
        初期化
        
        Args:
            settings: 量子化設定
        """
        self.settings = settings or QuantizationSettings()
        self.detected_bpm = None
        self.beat_times = None
        
    def detect_bpm_and_beats(self, audio_file: str) -> Tuple[float, np.ndarray]:
        """
        音声ファイルからBPMとビート位置を検出
        
        Args:
            audio_file: 音声ファイルパス
            
        Returns:
            (BPM, ビート時間配列)
        """
        try:
            # 音声読み込み
            y, sr = librosa.load(audio_file, sr=22050)
            
            # テンポとビート検出
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, 
                hop_length=512,
                start_bpm=60, 
                tightness=100
            )
            
            # フレームを時間に変換
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
            
            # BPM設定決定
            if self.settings.auto_detect_bpm:
                detected_bpm = float(tempo)
            else:
                detected_bpm = self.settings.manual_bpm or 120.0
            
            self.detected_bpm = detected_bpm
            self.beat_times = beat_times
            
            print(f"   BPM検出: {detected_bpm:.1f}")
            print(f"   ビート数: {len(beat_times)}")
            
            return detected_bpm, beat_times
            
        except Exception as e:
            print(f"   ⚠️ BPM検出エラー: {e}")
            # フォールバック値
            fallback_bpm = self.settings.manual_bpm or 120.0
            self.detected_bpm = fallback_bpm
            return fallback_bpm, np.array([])
    
    def create_quantization_grid(self, duration: float, bpm: float, 
                                resolution: str = "eighth") -> np.ndarray:
        """
        量子化グリッドを生成
        
        Args:
            duration: 楽曲の総演奏時間（秒）
            bpm: BPM
            resolution: グリッド解像度
            
        Returns:
            グリッド時間配列
        """
        if resolution not in self.GRID_RESOLUTIONS:
            resolution = "eighth"
        
        # グリッド間隔計算（秒）
        quarter_note_duration = 60.0 / bpm  # 4分音符の長さ
        grid_interval = quarter_note_duration * self.GRID_RESOLUTIONS[resolution]
        
        # スイング調整
        if self.settings.swing_ratio > 0 and resolution in ["eighth", "sixteenth", "thirtysecond"]:
            # 偶数・奇数グリッドで異なる間隔
            grid_times = []
            current_time = 0.0
            grid_count = 0
            
            while current_time <= duration:
                grid_times.append(current_time)
                
                # スイング適用（奇数拍を遅らせる）
                if grid_count % 2 == 0:
                    # 偶数拍（オンビート）
                    next_interval = grid_interval * (1 + self.settings.swing_ratio)
                else:
                    # 奇数拍（オフビート）
                    next_interval = grid_interval * (1 - self.settings.swing_ratio)
                
                current_time += next_interval
                grid_count += 1
            
            return np.array(grid_times)
        
        else:
            # 通常のストレートグリッド
            return np.arange(0, duration + grid_interval, grid_interval)
    
    def quantize_timing(self, original_time: float, grid_times: np.ndarray, 
                       strength: float = None) -> float:
        """
        単一タイミングを量子化
        
        Args:
            original_time: 元のタイミング
            grid_times: グリッド時間配列
            strength: 量子化強度（Noneの場合は設定値を使用）
            
        Returns:
            量子化後のタイミング
        """
        if len(grid_times) == 0:
            return original_time
        
        strength = strength if strength is not None else self.settings.strength
        
        # 最寄りのグリッドポイントを検索
        distances = np.abs(grid_times - original_time)
        closest_grid_idx = np.argmin(distances)
        closest_grid_time = grid_times[closest_grid_idx]
        
        # 量子化強度に基づく重み付き平均
        quantized_time = (
            original_time * (1 - strength) + 
            closest_grid_time * strength
        )
        
        return quantized_time
    
    def quantize_midi_file(self, midi_file: str, audio_file: str = None, 
                          track_name: str = None, output_file: str = None) -> str:
        """
        MIDIファイル全体を量子化
        
        Args:
            midi_file: 入力MIDIファイル
            audio_file: BPM検出用音声ファイル
            track_name: トラック名（設定適用用）
            output_file: 出力MIDIファイル（Noneの場合は元ファイルを上書き）
            
        Returns:
            量子化後MIDIファイルパス
        """
        if not self.settings.enabled:
            return midi_file
        
        print(f"🎼 MIDI量子化実行: {Path(midi_file).name}")
        
        try:
            # MIDIファイル読み込み
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            
            # BPM検出（音声ファイルが指定されている場合）
            if audio_file and Path(audio_file).exists():
                bpm, _ = self.detect_bpm_and_beats(audio_file)
            else:
                bpm = self.detected_bpm or self.settings.manual_bpm or 120.0
            
            # 楽曲時間取得
            duration = midi_data.get_end_time()
            
            # トラック別設定取得
            track_settings = self._get_track_settings(track_name)
            resolution = track_settings.get("grid_resolution", self.settings.grid_resolution)
            strength = track_settings.get("strength", self.settings.strength)
            
            print(f"   BPM: {bpm:.1f}, 解像度: {resolution}, 強度: {strength}")
            
            # 量子化グリッド生成
            grid_times = self.create_quantization_grid(duration, bpm, resolution)
            
            # 各楽器の音符を量子化
            total_notes_processed = 0
            quantization_stats = {
                "total_notes": 0,
                "avg_shift_ms": 0,
                "max_shift_ms": 0
            }
            
            shifts = []
            
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    # 開始タイミング量子化
                    original_start = note.start
                    quantized_start = self.quantize_timing(original_start, grid_times, strength)
                    shift_ms = abs(quantized_start - original_start) * 1000
                    shifts.append(shift_ms)
                    
                    # 終了タイミングも相対的に調整
                    note_duration = note.end - note.start
                    quantized_end = quantized_start + note_duration
                    
                    # 音符更新
                    note.start = max(0, quantized_start)
                    note.end = max(note.start + 0.1, quantized_end)  # 最小音符長確保
                    
                    total_notes_processed += 1
            
            # 統計計算
            if shifts:
                quantization_stats = {
                    "total_notes": total_notes_processed,
                    "avg_shift_ms": np.mean(shifts),
                    "max_shift_ms": np.max(shifts)
                }
            
            # 保存
            if output_file is None:
                output_file = midi_file
            
            midi_data.write(output_file)
            
            print(f"   ✅ 量子化完了: {total_notes_processed}音符処理")
            print(f"   平均シフト: {quantization_stats['avg_shift_ms']:.1f}ms")
            print(f"   最大シフト: {quantization_stats['max_shift_ms']:.1f}ms")
            
            return output_file
            
        except Exception as e:
            print(f"   ❌ 量子化エラー: {e}")
            return midi_file
    
    def _get_track_settings(self, track_name: str) -> Dict:
        """トラック別設定取得"""
        if not track_name or not self.settings.track_settings:
            return {}
        
        return self.settings.track_settings.get(track_name, {})
    
    def quantize_note_list(self, notes: List[Tuple[float, float, int]], 
                          duration: float, bpm: float = None,
                          track_name: str = None) -> List[Tuple[float, float, int]]:
        """
        音符リストを量子化（MIDIファイル生成前の段階で使用）
        
        Args:
            notes: [(start_time, end_time, pitch), ...] 形式の音符リスト
            duration: 楽曲の総演奏時間
            bpm: BPM（Noneの場合は検出済みまたはデフォルト値使用）
            track_name: トラック名
            
        Returns:
            量子化済み音符リスト
        """
        if not self.settings.enabled or not notes:
            return notes
        
        # BPM決定
        bpm = bpm or self.detected_bpm or self.settings.manual_bpm or 120.0
        
        # トラック設定
        track_settings = self._get_track_settings(track_name)
        resolution = track_settings.get("grid_resolution", self.settings.grid_resolution)
        strength = track_settings.get("strength", self.settings.strength)
        
        # グリッド生成
        grid_times = self.create_quantization_grid(duration, bpm, resolution)
        
        # 音符量子化
        quantized_notes = []
        for start_time, end_time, pitch in notes:
            quantized_start = self.quantize_timing(start_time, grid_times, strength)
            note_duration = end_time - start_time
            quantized_end = quantized_start + note_duration
            
            quantized_notes.append((quantized_start, quantized_end, pitch))
        
        return quantized_notes
    
    @classmethod
    def load_settings_from_file(cls, config_file: str) -> QuantizationSettings:
        """設定ファイルから量子化設定を読み込み"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            quantization_config = config.get("quantization", {})
            
            return QuantizationSettings(
                enabled=quantization_config.get("enabled", True),
                grid_resolution=quantization_config.get("grid_resolution", "thirtysecond"),
                strength=quantization_config.get("strength", 0.8),
                auto_detect_bpm=quantization_config.get("auto_detect_bpm", True),
                manual_bpm=quantization_config.get("manual_bpm"),
                swing_ratio=quantization_config.get("swing_ratio", 0.0),
                track_settings=quantization_config.get("track_settings", {})
            )
            
        except Exception as e:
            print(f"⚠️ 設定ファイル読み込みエラー: {e}")
            return QuantizationSettings()
    
    def save_settings_to_file(self, config_file: str):
        """量子化設定をファイルに保存"""
        config = {
            "quantization": {
                "enabled": self.settings.enabled,
                "grid_resolution": self.settings.grid_resolution,
                "strength": self.settings.strength,
                "auto_detect_bpm": self.settings.auto_detect_bpm,
                "manual_bpm": self.settings.manual_bpm,
                "swing_ratio": self.settings.swing_ratio,
                "track_settings": self.settings.track_settings or {}
            }
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"📝 量子化設定保存: {config_file}")
        except Exception as e:
            print(f"❌ 設定保存エラー: {e}")


def create_default_quantization_config():
    """デフォルト量子化設定ファイル作成"""
    default_config = {
        "quantization": {
            "enabled": True,
            "grid_resolution": "thirtysecond",
            "strength": 0.8,
            "auto_detect_bpm": True,
            "manual_bpm": None,
            "swing_ratio": 0.0,
            "track_settings": {
                "drums": {
                    "strength": 1.0,
                    "grid_resolution": "thirtysecond"
                },
                "bass": {
                    "strength": 0.9,
                    "grid_resolution": "thirtysecond"
                },
                "vocals": {
                    "strength": 0.6,
                    "grid_resolution": "thirtysecond"
                },
                "other": {
                    "strength": 0.8,
                    "grid_resolution": "thirtysecond"
                }
            }
        }
    }
    
    config_file = Path.cwd() / "quantization_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"📝 デフォルト量子化設定作成: {config_file}")
    return str(config_file)


def main():
    """テスト実行"""
    print("🎼 MIDI量子化エンジン テスト")
    print("=" * 40)
    
    # デフォルト設定作成
    config_file = create_default_quantization_config()
    
    # 設定読み込みテスト
    settings = MIDIQuantizer.load_settings_from_file(config_file)
    quantizer = MIDIQuantizer(settings)
    
    print(f"✅ 量子化エンジン初期化完了")
    print(f"   グリッド解像度: {settings.grid_resolution}")
    print(f"   量子化強度: {settings.strength}")
    print(f"   BPM自動検出: {settings.auto_detect_bpm}")
    
    # グリッド生成テスト
    test_duration = 10.0
    test_bpm = 120.0
    grid = quantizer.create_quantization_grid(test_duration, test_bpm, "thirtysecond")
    
    print(f"\n📊 グリッド生成テスト (10秒, 120BPM, 32分音符):")
    print(f"   グリッドポイント数: {len(grid)}")
    print(f"   最初の5ポイント: {grid[:5]}")
    
    # 量子化テスト
    test_times = [0.12, 0.53, 1.02, 1.48, 2.01]
    print(f"\n🎯 量子化テスト:")
    for original in test_times:
        quantized = quantizer.quantize_timing(original, grid, 0.8)
        shift_ms = abs(quantized - original) * 1000
        print(f"   {original:.2f}s → {quantized:.2f}s (shift: {shift_ms:.1f}ms)")


if __name__ == "__main__":
    main()