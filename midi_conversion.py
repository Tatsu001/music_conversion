#!/usr/bin/env python3
"""
音楽変換パイプライン Step 2: MIDI変換
分離された音声トラックをMIDIファイルに変換

Requirements:
- basic-pitch (Spotify製、メロディ検出)
- librosa (ドラム・ベース検出)
- pretty_midi (MIDIファイル操作)
"""

import os
import sys
import time
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
    print("✅ basic-pitch利用可能")
except ImportError:
    print("⚠️ basic-pitch がインストールされていません")
    print("   インストール: pip install basic-pitch")
    BASIC_PITCH_AVAILABLE = False


class MIDIConverter:
    """音声トラックをMIDIに変換するクラス"""
    
    def __init__(self):
        self.sample_rate = 22050  # librosaのデフォルト
        print("🎼 MIDIConverter初期化完了")
        
    def convert_separated_tracks(self, separated_files: Dict[str, str], output_dir: str = None) -> Dict[str, str]:
        """
        分離されたトラックをMIDIに変換
        
        Args:
            separated_files: 分離音声ファイルのパス辞書
            output_dir: MIDI出力ディレクトリ
            
        Returns:
            生成されたMIDIファイルのパス辞書
        """
        if not separated_files:
            raise ValueError("分離ファイルが指定されていません")
        
        # 出力ディレクトリ設定
        if output_dir is None:
            first_file = Path(next(iter(separated_files.values())))
            output_dir = first_file.parent.parent / "midi_tracks"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎼 MIDI変換開始")
        print(f"   出力ディレクトリ: {output_dir}")
        
        midi_files = {}
        
        for track_name, audio_file in separated_files.items():
            if not os.path.exists(audio_file):
                print(f"   ⚠️ {track_name}: ファイルが見つかりません")
                continue
            
            print(f"\n   🔄 {track_name} 処理中...")
            
            try:
                if track_name == "vocals" or track_name == "other":
                    # メロディトラック: basic-pitch使用
                    midi_file = self._convert_melody_track(audio_file, output_path, track_name)
                elif track_name == "drums":
                    # ドラムトラック: librosa使用
                    midi_file = self._convert_drum_track(audio_file, output_path, track_name)
                elif track_name == "bass":
                    # ベーストラック: librosa使用
                    midi_file = self._convert_bass_track(audio_file, output_path, track_name)
                else:
                    # その他: basic-pitch使用
                    midi_file = self._convert_melody_track(audio_file, output_path, track_name)
                
                if midi_file:
                    midi_files[track_name] = midi_file
                    print(f"      ✅ 変換完了: {Path(midi_file).name}")
                else:
                    print(f"      ❌ 変換失敗")
                    
            except Exception as e:
                print(f"      ❌ エラー: {e}")
        
        print(f"\n✅ MIDI変換完了: {len(midi_files)}個のファイル生成")
        return midi_files
    
    def _convert_melody_track(self, audio_file: str, output_dir: Path, track_name: str) -> Optional[str]:
        """メロディトラックをMIDIに変換（basic-pitch使用）"""
        if not BASIC_PITCH_AVAILABLE:
            return self._convert_with_librosa_melody(audio_file, output_dir, track_name)
        
        try:
            output_file = output_dir / f"{track_name}.mid"
            
            # basic-pitchで変換
            predict_and_save(
                [audio_file],
                str(output_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH
            )
            
            # basic-pitchの出力ファイル名を変更
            generated_file = output_dir / f"{Path(audio_file).stem}_basic_pitch.mid"
            if generated_file.exists():
                generated_file.rename(output_file)
                return str(output_file)
            
            return None
            
        except Exception as e:
            print(f"         basic-pitch エラー: {e}")
            return self._convert_with_librosa_melody(audio_file, output_dir, track_name)
    
    def _convert_with_librosa_melody(self, audio_file: str, output_dir: Path, track_name: str) -> Optional[str]:
        """librosaを使ったメロディ変換（basic-pitchの代替）"""
        try:
            # 音声読み込み
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # ピッチ検出（YIN algorithm）
            f0 = librosa.yin(y, fmin=80, fmax=800, sr=sr)
            
            # 時間軸作成
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            
            # MIDIファイル作成
            midi_data = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)  # Piano
            
            # 音符生成
            note_threshold = 100  # Hz以上を有効な音程とする
            min_duration = 0.1   # 最小音符長
            
            current_note = None
            note_start = None
            
            for i, (time, freq) in enumerate(zip(times, f0)):
                if freq > note_threshold and not np.isnan(freq):
                    # 周波数をMIDIノート番号に変換
                    midi_note = int(librosa.hz_to_midi(freq))
                    
                    if current_note is None or abs(midi_note - current_note) > 1:
                        # 新しい音符開始
                        if current_note is not None and note_start is not None:
                            # 前の音符を追加
                            if time - note_start >= min_duration:
                                note = pretty_midi.Note(
                                    velocity=80,
                                    pitch=current_note,
                                    start=note_start,
                                    end=time
                                )
                                instrument.notes.append(note)
                        
                        current_note = midi_note
                        note_start = time
                else:
                    # 音符終了
                    if current_note is not None and note_start is not None:
                        if time - note_start >= min_duration:
                            note = pretty_midi.Note(
                                velocity=80,
                                pitch=current_note,
                                start=note_start,
                                end=time
                            )
                            instrument.notes.append(note)
                        current_note = None
                        note_start = None
            
            # 最後の音符処理
            if current_note is not None and note_start is not None:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=current_note,
                    start=note_start,
                    end=times[-1]
                )
                instrument.notes.append(note)
            
            midi_data.instruments.append(instrument)
            
            # ファイル保存
            output_file = output_dir / f"{track_name}.mid"
            midi_data.write(str(output_file))
            
            print(f"         librosa変換: {len(instrument.notes)}個の音符")
            return str(output_file)
            
        except Exception as e:
            print(f"         librosa変換エラー: {e}")
            return None
    
    def _convert_drum_track(self, audio_file: str, output_dir: Path, track_name: str) -> Optional[str]:
        """ドラムトラックをMIDIに変換（librosa使用）"""
        try:
            # 音声読み込み
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # オンセット検出（ドラムヒット検出）
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr, units='frames',
                hop_length=512,
                backtrack=True,
                pre_max=3,
                post_max=3,
                pre_avg=3,
                post_avg=5,
                delta=0.2,
                wait=10
            )
            
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # MIDIファイル作成
            midi_data = pretty_midi.PrettyMIDI()
            drum_instrument = pretty_midi.Instrument(program=0, is_drum=True)
            
            # ドラム音の分類（簡易版）
            for onset_time in onset_times:
                # 周波数特性に基づく簡易分類
                frame_idx = librosa.time_to_frames(onset_time, sr=sr)
                if frame_idx < len(y) - 1024:
                    # 短い窓でスペクトル分析
                    window = y[frame_idx:frame_idx+1024]
                    if len(window) == 1024:
                        spectrum = np.abs(np.fft.fft(window))
                        
                        # 低域エネルギー（キック判定）
                        low_energy = np.sum(spectrum[0:50])
                        # 高域エネルギー（ハイハット判定）
                        high_energy = np.sum(spectrum[200:512])
                        # 中域エネルギー（スネア判定）
                        mid_energy = np.sum(spectrum[50:200])
                        
                        # ドラム音種類判定
                        if low_energy > high_energy and low_energy > mid_energy:
                            # キックドラム
                            drum_note = 36  # C2
                        elif high_energy > low_energy and high_energy > mid_energy:
                            # ハイハット
                            drum_note = 42  # F#2
                        else:
                            # スネアドラム
                            drum_note = 38  # D2
                        
                        # MIDI音符追加
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=drum_note,
                            start=onset_time,
                            end=onset_time + 0.1  # 短い持続時間
                        )
                        drum_instrument.notes.append(note)
            
            midi_data.instruments.append(drum_instrument)
            
            # ファイル保存
            output_file = output_dir / f"{track_name}.mid"
            midi_data.write(str(output_file))
            
            print(f"         ドラム変換: {len(drum_instrument.notes)}個のヒット")
            return str(output_file)
            
        except Exception as e:
            print(f"         ドラム変換エラー: {e}")
            return None
    
    def _convert_bass_track(self, audio_file: str, output_dir: Path, track_name: str) -> Optional[str]:
        """ベーストラックをMIDIに変換（librosa使用）"""
        try:
            # 音声読み込み
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # 低域フィルタ適用（ベース音域強調）
            y_filtered = librosa.effects.preemphasis(y, coef=0.0)  # プリエンファシス無効
            
            # ピッチ検出（低域用設定）
            f0 = librosa.yin(y_filtered, fmin=40, fmax=300, sr=sr)
            
            # 時間軸作成
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
            
            # MIDIファイル作成
            midi_data = pretty_midi.PrettyMIDI()
            bass_instrument = pretty_midi.Instrument(program=33)  # Electric Bass
            
            # 音符生成
            note_threshold = 50   # Hz以上を有効な音程とする
            min_duration = 0.2    # ベースは少し長めの最小音符長
            
            current_note = None
            note_start = None
            
            for i, (time, freq) in enumerate(zip(times, f0)):
                if freq > note_threshold and not np.isnan(freq):
                    # 周波数をMIDIノート番号に変換
                    midi_note = int(librosa.hz_to_midi(freq))
                    
                    # ベース音域に制限（E1-E4: 28-64）
                    midi_note = max(28, min(64, midi_note))
                    
                    if current_note is None or abs(midi_note - current_note) > 2:
                        # 新しい音符開始
                        if current_note is not None and note_start is not None:
                            # 前の音符を追加
                            if time - note_start >= min_duration:
                                note = pretty_midi.Note(
                                    velocity=90,
                                    pitch=current_note,
                                    start=note_start,
                                    end=time
                                )
                                bass_instrument.notes.append(note)
                        
                        current_note = midi_note
                        note_start = time
                else:
                    # 音符終了
                    if current_note is not None and note_start is not None:
                        if time - note_start >= min_duration:
                            note = pretty_midi.Note(
                                velocity=90,
                                pitch=current_note,
                                start=note_start,
                                end=time
                            )
                            bass_instrument.notes.append(note)
                        current_note = None
                        note_start = None
            
            # 最後の音符処理
            if current_note is not None and note_start is not None:
                note = pretty_midi.Note(
                    velocity=90,
                    pitch=current_note,
                    start=note_start,
                    end=times[-1]
                )
                bass_instrument.notes.append(note)
            
            midi_data.instruments.append(bass_instrument)
            
            # ファイル保存
            output_file = output_dir / f"{track_name}.mid"
            midi_data.write(str(output_file))
            
            print(f"         ベース変換: {len(bass_instrument.notes)}個の音符")
            return str(output_file)
            
        except Exception as e:
            print(f"         ベース変換エラー: {e}")
            return None
    
    def analyze_midi_files(self, midi_files: Dict[str, str]) -> Dict[str, Dict]:
        """生成されたMIDIファイルの分析"""
        print(f"\n📊 MIDI分析")
        
        analysis = {}
        
        for track_name, midi_file in midi_files.items():
            if not os.path.exists(midi_file):
                continue
            
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_file)
                
                total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
                duration = midi_data.get_end_time()
                
                # 楽器情報
                instruments_info = []
                for inst in midi_data.instruments:
                    inst_info = {
                        "program": inst.program,
                        "is_drum": inst.is_drum,
                        "notes": len(inst.notes)
                    }
                    instruments_info.append(inst_info)
                
                analysis[track_name] = {
                    "total_notes": total_notes,
                    "duration": duration,
                    "instruments": instruments_info,
                    "file_size": os.path.getsize(midi_file) / 1024  # KB
                }
                
                print(f"   {track_name}:")
                print(f"     音符数: {total_notes}")
                print(f"     長さ: {duration:.2f}秒")
                print(f"     楽器数: {len(instruments_info)}")
                print(f"     ファイルサイズ: {analysis[track_name]['file_size']:.2f}KB")
                
            except Exception as e:
                print(f"   ❌ {track_name} 分析エラー: {e}")
                analysis[track_name] = {"error": str(e)}
        
        return analysis
    
    def merge_midi_tracks(self, midi_files: Dict[str, str], output_file: str = None) -> Optional[str]:
        """複数のMIDIトラックを1つのファイルに統合"""
        if not midi_files:
            return None
        
        if output_file is None:
            first_file = Path(next(iter(midi_files.values())))
            output_file = first_file.parent / "merged_composition.mid"
        
        print(f"\n🎵 MIDIトラック統合")
        print(f"   出力: {output_file}")
        
        try:
            # 新しいMIDIファイル作成
            merged_midi = pretty_midi.PrettyMIDI()
            
            for track_name, midi_file in midi_files.items():
                if not os.path.exists(midi_file):
                    continue
                
                try:
                    track_midi = pretty_midi.PrettyMIDI(midi_file)
                    
                    # 各楽器を統合MIDIに追加
                    for instrument in track_midi.instruments:
                        # 楽器名を設定
                        instrument.name = f"{track_name}_{instrument.name}" if instrument.name else track_name
                        merged_midi.instruments.append(instrument)
                    
                    print(f"   ✅ {track_name}: {len(track_midi.instruments)}楽器追加")
                    
                except Exception as e:
                    print(f"   ❌ {track_name}: 読み込みエラー - {e}")
            
            # 統合ファイル保存
            merged_midi.write(str(output_file))
            
            # 統計情報
            total_instruments = len(merged_midi.instruments)
            total_notes = sum(len(inst.notes) for inst in merged_midi.instruments)
            duration = merged_midi.get_end_time()
            
            print(f"   📊 統合結果:")
            print(f"     楽器数: {total_instruments}")
            print(f"     総音符数: {total_notes}")
            print(f"     演奏時間: {duration:.2f}秒")
            
            return str(output_file)
            
        except Exception as e:
            print(f"   ❌ 統合エラー: {e}")
            return None


def main():
    """メイン実行関数"""
    print("🎼 音楽変換パイプライン Step 2: MIDI変換")
    print("=" * 50)
    
    # 作業ディレクトリ設定
    work_dir = Path.home() / "ドキュメント" / "conversion_music"
    
    if not work_dir.exists():
        print(f"❌ 作業ディレクトリが見つかりません: {work_dir}")
        print("   Step 1を先に実行してください")
        return
    
    print(f"📁 作業ディレクトリ: {work_dir}")
    
    # 分離ファイル検索
    separated_dirs = list(work_dir.glob("input_separated"))  # 正確なディレクトリ名
    
    if not separated_dirs:
        print("❌ 分離済みファイルが見つかりません")
        print("   Step 1の音源分離を先に実行してください")
        return
    
    print(f"🎵 見つかった分離ディレクトリ: {len(separated_dirs)}個")
    for i, dir_path in enumerate(separated_dirs):
        print(f"   {i+1}. {dir_path.name}")
    
    # 最初の分離ディレクトリで実行
    separated_dir = separated_dirs[0]
    print(f"   処理対象: {separated_dir.name}")
    
    # htdemucsの出力を検索
    htdemucs_dir = separated_dir / "htdemucs"
    if not htdemucs_dir.exists():
        print("❌ htdemucsディレクトリが見つかりません")
        print(f"   期待パス: {htdemucs_dir}")
        return
    
    # inputディレクトリを検索
    input_dirs = list(htdemucs_dir.glob("input"))
    if not input_dirs:
        print("❌ inputディレクトリが見つかりません")
        print(f"   期待パス: {htdemucs_dir}/input/")
        # 利用可能なディレクトリを表示
        available_dirs = [d.name for d in htdemucs_dir.iterdir() if d.is_dir()]
        print(f"   利用可能なディレクトリ: {available_dirs}")
        return
    
    track_dir = input_dirs[0]
    print(f"   音声ファイルディレクトリ: {track_dir}")
    
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
    
    # MIDI変換実行
    converter = MIDIConverter()
    
    try:
        print(f"\n🔄 MIDI変換開始")
        start_time = time.time()
        
        midi_files = converter.convert_separated_tracks(separated_files)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if midi_files:
            print(f"\n✅ MIDI変換完了!")
            print(f"   処理時間: {processing_time:.2f}秒")
            print(f"   生成ファイル数: {len(midi_files)}")
            
            # MIDI分析
            analysis = converter.analyze_midi_files(midi_files)
            
            # MIDIファイル統合
            merged_file = converter.merge_midi_tracks(midi_files)
            
            if merged_file:
                print(f"\n🎵 統合MIDIファイル生成完了")
                print(f"   ファイル: {Path(merged_file).name}")
                print(f"   次のステップ: 音色変換 (Step 3)")
            
        else:
            print(f"❌ MIDIファイルが生成されませんでした")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(f"\n🔧 トラブルシューティング:")
        print(f"   1. basic-pitch確認: pip list | grep basic-pitch")
        print(f"   2. librosa確認: python -c 'import librosa; print(librosa.__version__)'")
        print(f"   3. pretty-midi確認: python -c 'import pretty_midi; print(\"OK\")'")


if __name__ == "__main__":
    main()