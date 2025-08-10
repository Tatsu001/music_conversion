#!/bin/bash

# 音楽変換パイプライン Step 1 テスト実行スクリプト
echo "🎵 音楽変換パイプライン Step 1: 音源分離テスト"
echo "=================================================="

# 作業ディレクトリ移動
WORK_DIR="$HOME/ドキュメント/conversion_music"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "📁 作業ディレクトリ: $(pwd)"

# 仮想環境確認・有効化
if [ -d "music_conversion_env" ]; then
    echo "🐍 仮想環境有効化中..."
    source music_conversion_env/bin/activate
    echo "   Python: $(python --version)"
    echo "   pip: $(pip --version)"
else
    echo "❌ 仮想環境が見つかりません"
    echo "   setup.shを実行してください"
    exit 1
fi

# GPU環境確認
echo ""
echo "🔧 システム環境確認"
python -c "
import torch
import sys
print(f'   Python: {sys.version.split()[0]}')
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA利用可能: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA Version: {torch.version.cuda}')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

# 必要ライブラリ確認
echo ""
echo "📦 ライブラリ確認"
python -c "
try:
    import demucs; print(f'   ✅ Demucs: {demucs.__version__}')
except: print('   ❌ Demucs: インストールされていません')

try:
    import librosa; print(f'   ✅ LibROSA: {librosa.__version__}')
except: print('   ❌ LibROSA: インストールされていません')

try:
    import soundfile; print(f'   ✅ SoundFile: {soundfile.__version__}')
except: print('   ❌ SoundFile: インストールされていません')

try:
    import torch; print(f'   ✅ PyTorch: {torch.__version__}')
except: print('   ❌ PyTorch: インストールされていません')
"

# テスト音声ファイル確認
echo ""
echo "🎵 音声ファイル確認"
audio_files_found=false

# 各拡張子を個別にチェック
for ext in mp3 wav m4a flac aac; do
    if ls *.$ext 2>/dev/null | head -1 >/dev/null; then
        audio_files_found=true
        break
    fi
done

if [ "$audio_files_found" = true ]; then
    echo "   見つかった音声ファイル:"
    for ext in mp3 wav m4a flac aac; do
        if ls *.$ext 2>/dev/null >/dev/null; then
            echo "   📄 $ext ファイル:"
            ls -la *.$ext 2>/dev/null | head -3 | while read line; do
                echo "      $line"
            done
        fi
    done
else
    echo "   ⚠️ 音声ファイルが見つかりません"
    echo ""
    echo "📝 テスト用音声ファイル生成（オプション）"
    echo "   以下のコマンドでサンプル音声を生成できます："
    echo ""
    echo "   # 10秒の440Hz正弦波"
    echo "   ffmpeg -f lavfi -i 'sine=frequency=440:duration=10' test_sine.wav"
    echo ""
    echo "   # 10秒のホワイトノイズ"
    echo "   ffmpeg -f lavfi -i 'anoisesrc=duration=10:color=white' test_noise.wav"
    echo ""
    echo "   # 複数周波数の合成音（楽器らしい音）"
    echo "   ffmpeg -f lavfi -i 'sine=f=261.63:d=3,sine=f=329.63:d=3,sine=f=392.00:d=3' -filter_complex amix=inputs=3 test_chord.wav"
    echo ""
fi

# 音源分離スクリプト存在確認
if [ -f "music_separation.py" ]; then
    echo ""
    echo "✅ music_separation.py が見つかりました"
    
    # テスト音声ファイルがある場合は実行
    if [ "$audio_files_found" = true ]; then
        echo ""
        echo "🚀 音源分離テスト実行開始"
        echo "   （RTX 2070での処理時間を測定します）"
        echo ""
        
        # メモリ使用量監視を背景で開始
        echo "💾 GPU使用量監視開始"
        nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv --loop=5 > gpu_usage.log &
        NVIDIA_PID=$!
        
        # 音源分離実行
        python music_separation.py
        
        # GPU監視停止
        kill $NVIDIA_PID 2>/dev/null
        
        echo ""
        echo "📊 GPU使用量ログ (最後の10行):"
        tail -10 gpu_usage.log 2>/dev/null || echo "   ログファイルなし"
        
    else
        echo ""
        echo "⚠️ 音声ファイルがないため、テスト実行をスキップします"
        echo "   音声ファイルを配置後、以下で実行："
        echo "   python music_separation.py"
    fi
else
    echo ""
    echo "❌ music_separation.py が見つかりません"
    echo "   スクリプトを作成してください"
fi

# 次のステップ案内
echo ""
echo "🎯 次のステップ"
echo "   1. 音源分離結果の確認"
echo "   2. Step 2: MIDI変換実装"
echo "   3. 各楽器トラック→MIDIファイル変換"
echo ""
echo "📁 出力ディレクトリ構造："
echo "   [入力ファイル名]_separated/"
echo "   └── htdemucs/"
echo "       └── [ファイル名]/"
echo "           ├── vocals.wav"
echo "           ├── drums.wav"
echo "           ├── bass.wav"
echo "           └── other.wav"
echo ""

deactivate 2>/dev/null || true
echo "✅ テスト完了"