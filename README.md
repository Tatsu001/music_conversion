# 音楽変換パイプライン使用方法

## 🚀 環境有効化
```bash
source music_conversion_env/bin/activate
```

## 📁 ファイル準備
1. テスト用楽曲を `input_song.mp3` として配置
2. 3-5分程度のバンド楽曲を推奨

## 🎵 Step 1: 音源分離
```bash
python music_separation.py
```
- 出力: `separated_audio/` ディレクトリ
- vocals.wav, drums.wav, bass.wav, other.wav

## 🎼 Step 2: MIDI変換
```bash
python midi_conversion_madmom.py
```
- 出力: `midi_output_madmom/` ディレクトリ
- melody.mid, drums.mid, bass.mid

## ⚡ Step 3: 音色変換
```bash
python sound_conversion.py
```
- 8bit風・オルゴール風音源に変換

## 🔍 環境確認
```bash
python diagnose.py
```

## ⚠️ 注意事項
- 初回実行時は大容量モデルのダウンロードが発生
- GPU使用時は6GB以上のVRAMを推奨
- 長時間楽曲では処理時間が増加

## 📞 トラブルシューティング
1. 仮想環境が有効化されているか確認
2. `python diagnose.py` で環境診断
3. GPU メモリ不足の場合は楽曲を短くする
