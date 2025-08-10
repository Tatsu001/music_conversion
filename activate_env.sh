#!/bin/bash
# 音楽変換パイプライン環境有効化
echo "🎵 音楽変換パイプライン環境を有効化..."
source music_conversion_env/bin/activate
echo "✓ 環境有効化完了"
echo ""
echo "利用可能コマンド:"
echo "  python music_separation.py        # Step 1: 音源分離"
echo "  python midi_conversion_madmom.py  # Step 2: MIDI変換"
echo "  python sound_conversion.py        # Step 3: 音色変換"
echo "  python diagnose.py               # 環境診断"
