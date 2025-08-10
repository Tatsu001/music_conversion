#!/bin/bash
# 音楽変換パイプライン完全セットアップ（Python3.11版）
# madmom対応 + RTX 2070最適化

echo "🎵 音楽変換パイプライン完全セットアップ（Python3.11版）"
echo "=" * 60

# 環境情報表示
echo "📋 環境情報:"
echo "  OS: $(lsb_release -d | cut -f2)"
echo "  アーキテクチャ: $(uname -m)"
echo "  GPU: RTX 2070 対応"
echo ""

# ステップ1: システム依存関係確認・インストール
echo "ステップ1: システム依存関係インストール..."
echo "管理者権限が必要です。パスワードを入力してください。"

sudo apt update

# Python3.11とツール群
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    build-essential \
    git \
    curl \
    wget

# 音声処理システム依存関係
sudo apt install -y \
    libsndfile1-dev \
    libfftw3-dev \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev

echo "✓ システム依存関係インストール完了"

# ステップ2: Python3.11確認
echo ""
echo "ステップ2: Python3.11環境確認..."
python3.11 --version

if [ $? -ne 0 ]; then
    echo "❌ Python3.11が正しくインストールされていません"
    echo "手動でインストールしてください:"
    echo "sudo apt install python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

echo "✓ Python3.11確認完了"

# ステップ3: 仮想環境作成
echo ""
echo "ステップ3: Python3.11仮想環境作成..."

# 既存環境があれば削除
if [ -d "music_conversion_env" ]; then
    echo "既存の仮想環境を削除中..."
    rm -rf music_conversion_env
fi

# 新しい仮想環境作成
python3.11 -m venv music_conversion_env

if [ $? -ne 0 ]; then
    echo "❌ 仮想環境作成に失敗しました"
    echo "以下を実行してください:"
    echo "sudo apt install python3.11-venv"
    exit 1
fi

echo "✓ 仮想環境作成完了"

# ステップ4: 仮想環境有効化
echo ""
echo "ステップ4: 仮想環境有効化..."
source music_conversion_env/bin/activate

# Python確認
echo "仮想環境Python: $(python --version)"
echo "Python実行パス: $(which python)"

# ステップ5: pip更新
echo ""
echo "ステップ5: pip & 基本ツール更新..."
python -m pip install --upgrade pip
pip install wheel setuptools

echo "✓ pip更新完了"

# ステップ6: 基本数値計算ライブラリ
echo ""
echo "ステップ6: 基本数値計算ライブラリインストール..."
pip install numpy scipy
echo "✓ NumPy & SciPy インストール完了"

# ステップ7: Cython（madmom用）
echo ""
echo "ステップ7: Cython インストール..."
pip install Cython
echo "✓ Cython インストール完了"

# ステップ8: 音声処理ライブラリ
echo ""
echo "ステップ8: 音声処理ライブラリインストール..."
pip install librosa soundfile audioread
echo "✓ 音声処理ライブラリインストール完了"

# ステップ9: PyTorch（CUDA対応）
echo ""
echo "ステップ9: PyTorch（CUDA対応）インストール..."
echo "これには数分かかります..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if [ $? -ne 0 ]; then
    echo "⚠️  CUDA版失敗。CPU版をインストール..."
    pip install torch torchvision torchaudio
fi

echo "✓ PyTorch インストール完了"

# ステップ10: Demucs（音源分離）
echo ""
echo "ステップ10: Demucs インストール..."
pip install demucs
echo "✓ Demucs インストール完了"

# ステップ11: madmom（ドラム検出）
echo ""
echo "ステップ11: madmom インストール..."
echo "これは時間がかかる場合があります..."

pip install madmom

if [ $? -eq 0 ]; then
    echo "✅ madmom インストール成功！"
else
    echo "⚠️  標準インストール失敗。代替方法を試行..."
    
    # 代替方法1: 環境変数設定
    export NUMPY_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())")
    pip install madmom --no-build-isolation
    
    if [ $? -ne 0 ]; then
        echo "⚠️  代替方法も失敗。GitHubから最新版を試行..."
        pip install git+https://github.com/CPJKU/madmom.git
    fi
fi

# ステップ12: MIDI処理ライブラリ
echo ""
echo "ステップ12: MIDI処理ライブラリインストール..."
pip install basic-pitch pretty_midi mido
echo "✓ MIDI処理ライブラリインストール完了"

# ステップ13: 可視化・その他ライブラリ
echo ""
echo "ステップ13: 可視化・その他ライブラリインストール..."
pip install matplotlib seaborn pandas tqdm
echo "✓ 可視化ライブラリインストール完了"

# ステップ14: インストール確認
echo ""
echo "=" * 60
echo "📊 インストール確認テスト"
echo "=" * 60

python -c "
import sys
print(f'Python: {sys.version}')
print(f'実行パス: {sys.executable}')
print('')

packages = [
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('librosa', 'LibROSA'),
    ('torch', 'PyTorch'),
    ('demucs', 'Demucs'),
    ('madmom', 'madmom'),
    ('basic_pitch', 'Basic Pitch'),
    ('pretty_midi', 'Pretty MIDI'),
    ('mido', 'Mido'),
    ('matplotlib', 'Matplotlib')
]

success_count = 0
failed_packages = []

for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'インストール済み')
        print(f'✓ {name}: {version}')
        success_count += 1
    except ImportError as e:
        print(f'❌ {name}: インポートエラー')
        failed_packages.append(name)

print('')
print(f'成功: {success_count}/{len(packages)} パッケージ')

# PyTorch GPU確認
try:
    import torch
    print('')
    print('🚀 GPU環境確認:')
    print(f'  CUDA利用可能: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA バージョン: {torch.version.cuda}')
    else:
        print('  CPU実行モード')
except:
    print('❌ PyTorch GPU確認失敗')

# madmom特別確認
try:
    import madmom
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.downbeats import DBNDownBeatTracker
    from madmom.features.onsets import OnsetPeakPickingProcessor
    
    print('')
    print('🥁 madmom機能確認:')
    print('  ✓ RNNBeatProcessor: 利用可能')
    print('  ✓ DBNDownBeatTracker: 利用可能')
    print('  ✓ OnsetPeakPickingProcessor: 利用可能')
    print('  🎉 madmom完全対応!')
    
except ImportError as e:
    print(f'')
    print(f'⚠️  madmom機能制限: {e}')
    failed_packages.append('madmom機能')

print('')
if len(failed_packages) == 0:
    print('🎉 全ライブラリが正常にインストールされました！')
else:
    print(f'⚠️  {len(failed_packages)}個のパッケージに問題があります: {failed_packages}')
"

# ステップ15: 使用方法ガイド作成
echo ""
echo "=" * 60
echo "📖 使用方法ガイド作成中..."

cat > README_USAGE.md << 'EOF'
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
EOF

# ステップ16: 便利スクリプト作成
echo ""
echo "📝 便利スクリプト作成中..."

# 環境有効化スクリプト
cat > activate_env.sh << 'EOF'
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
EOF

chmod +x activate_env.sh

# クイックテストスクリプト
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""クイックテストスクリプト"""

def quick_test():
    print("🧪 クイックテスト実行中...")
    
    # 基本ライブラリテスト
    try:
        import numpy as np
        import torch
        import librosa
        import demucs
        import madmom
        import basic_pitch
        import pretty_midi
        
        print("✓ 全ライブラリインポート成功")
        
        # GPU確認
        if torch.cuda.is_available():
            print(f"✓ GPU利用可能: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CPU実行モード")
        
        # madmom機能テスト
        from madmom.features.beats import RNNBeatProcessor
        processor = RNNBeatProcessor()
        print("✓ madmom機能確認完了")
        
        print("\n🎉 全システム正常動作中！")
        return True
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        return False

if __name__ == "__main__":
    quick_test()
EOF

chmod +x quick_test.py

# 完了メッセージ
echo ""
echo "=" * 60
echo "🎉 セットアップ完了！"
echo "=" * 60
echo ""
echo "📁 作成されたファイル:"
echo "  music_conversion_env/     # Python3.11仮想環境"
echo "  README_USAGE.md          # 使用方法ガイド"
echo "  activate_env.sh          # 環境有効化スクリプト"
echo "  quick_test.py            # クイックテスト"
echo ""
echo "🚀 次のステップ:"
echo "1. 環境有効化:"
echo "   source music_conversion_env/bin/activate"
echo "   # または ./activate_env.sh"
echo ""
echo "2. テスト実行:"
echo "   python quick_test.py"
echo ""
echo "3. 楽曲変換開始:"
echo "   # input_song.mp3 を配置してから"
echo "   python music_separation.py"
echo ""
echo "📖 詳細な使用方法: README_USAGE.md を参照"
echo ""
echo "🔧 問題が発生した場合:"
echo "   python diagnose.py"
echo ""
echo "💡 このターミナルで仮想環境を有効化するには:"
echo "   source music_conversion_env/bin/activate"