# 高品質音楽分離・MIDI変換システム

## 🎯 概要
元音源からhtdemucs_ftによる高品質4トラック分離を行い、元音源のBPMを基準とした32分音符精度の量子化MIDI変換を実行するシステムです。

## ✨ 主な機能
- **htdemucs_ft高品質音源分離** - 4トラック（vocals, drums, bass, other）への精密分離
- **元音源統一BPM検出** - 分離前の音源から正確なBPM検出
- **32分音符量子化** - 最高精度でのリズム補正・アタック調整  
- **ボーカル+その他マージ** - 指定トラックのみのMIDI統合
- **前処理パイプライン** - 正規化・DC除去・周波数補強

## 🚀 環境セットアップ
```bash
# 環境有効化
source music_conversion_env/bin/activate

# または
source activate_env.sh
```

## 📁 ファイル準備
1. 元音源を `input.m4a` として配置
2. バンド楽曲（3-5分程度）を推奨

## 🎵 Step 1: 高品質音源分離
```bash
python enhanced_music_separation.py --model htdemucs_ft --quality high --input input.m4a
```

**出力:**
- `input_separated_enhanced/htdemucs_ft/input_preprocessed/`
  - vocals.wav, drums.wav, bass.wav, other.wav

## 🎼 Step 2: 32分音符量子化MIDI変換
```bash
python enhanced_midi_conversion.py --model htdemucs_ft --quantize --grid thirtysecond --strength 0.8
```

**出力:**
- `input_separated_enhanced/htdemucs_ft/midi_tracks_quantized/`
  - **`merged_composition_vocals_other.mid`** - メインファイル（ボーカル+その他）
  - `merged_composition.mid` - 全楽器統合
  - 個別MIDI: vocals.mid, drums.mid, bass.mid, other.mid

## ⚙️ システム構成

### メインシステム
- `enhanced_music_separation.py` - htdemucs_ft高品質分離
- `enhanced_midi_conversion.py` - 統一MIDI変換システム
- `midi_quantizer.py` - 32分音符量子化エンジン

### 設定ファイル  
- `quantization_config.json` - 量子化設定
- `separation_config.json` - 分離品質設定

### 将来用ツール
- `midi_tuning.py` - MIDI変換精度最適化（将来用）
- `setup.sh` - 環境構築スクリプト

## 📊 量子化設定

### デフォルト設定（32分音符）
```json
{
  "quantization": {
    "grid_resolution": "thirtysecond",
    "strength": 0.8,
    "auto_detect_bpm": true
  }
}
```

### トラック別設定
- **vocals**: 強度0.6（自然さ重視）
- **drums**: 強度1.0（完全量子化）  
- **bass**: 強度0.9（リズム重視）
- **other**: 強度0.8（バランス重視）

## 🎛️ コマンドラインオプション

### 音源分離オプション
```bash
--model htdemucs_ft          # 使用モデル
--quality high               # 品質設定
--input input.m4a           # 入力ファイル
```

### MIDI変換オプション
```bash
--quantize                  # 量子化有効化
--grid thirtysecond         # グリッド解像度（quarter/eighth/sixteenth/thirtysecond）
--strength 0.8              # 量子化強度（0.0-1.0）
--bpm 120                   # 手動BPM指定（省略時は自動検出）
```

## 🔧 技術仕様

### BPM検出
- **ソース**: 元音源（input.m4a）からの統一検出
- **エンジン**: librosa.beat.beat_track
- **精度**: 全トラック統一BPMによる正確な量子化

### 量子化精度
- **32分音符**: 0.125倍間隔（4分音符基準）
- **平均シフト**: 25-30ms程度
- **最大シフト**: 55ms程度

### 対応形式
- **入力**: .m4a, .mp3, .wav等
- **出力**: .mid（MIDI Format 1）

## ⚠️ 注意事項
- 初回実行時は大容量モデル（htdemucs_ft）のダウンロードが発生
- GPU使用時は6GB以上のVRAMを推奨（RTX 2070以上）
- 長時間楽曲では処理時間が大幅に増加

## 🔍 システム要件
- Python 3.11
- PyTorch 2.7.1+cu118（CUDA対応）
- Demucs 4.0.1
- basic-pitch（Spotify製）
- librosa 0.11.0
- pretty_midi

## 📞 トラブルシューティング

### 環境関連
1. 仮想環境が有効化されているか確認
2. CUDA環境の確認（GPU使用時）

### メモリ不足対策
1. セグメント長を削減（--segment 6）
2. 楽曲を短く分割して処理
3. CPUモードで実行（--device cpu）

### BPM検出問題
1. 手動BPM指定（--bpm オプション）
2. 元音源の品質確認

## 🎵 使用例

### 基本的な変換フロー
```bash
# 1. 高品質分離
python enhanced_music_separation.py --model htdemucs_ft --quality high --input input.m4a

# 2. 32分音符量子化MIDI変換
python enhanced_midi_conversion.py --model htdemucs_ft --quantize --grid thirtysecond --strength 0.8

# 結果確認
ls input_separated_enhanced/htdemucs_ft/midi_tracks_quantized/
```

### カスタマイズ例
```bash
# 16分音符量子化・低強度
python enhanced_midi_conversion.py --quantize --grid sixteenth --strength 0.5

# 手動BPM指定
python enhanced_midi_conversion.py --quantize --bpm 128 --strength 0.9
```

## 📈 成果物
最終的に `merged_composition_vocals_other.mid` ファイルが生成され、元音源のBPMに基づく32分音符精度の量子化が適用されたボーカル+その他楽器のMIDIファイルが得られます。