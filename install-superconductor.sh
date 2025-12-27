#!/bin/bash
set -e

# SuperConductor Installation Script
# Installs SuperConductor AppImage to ~/.local/bin and creates desktop shortcut

INSTALL_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons"
APP_NAME="SuperConductor"
VERSION="0.11.3"
DOWNLOAD_URL="https://github.com/SuperFlyTV/SuperConductor/releases/download/v${VERSION}/SuperConductor-${VERSION}-Linux-Executable.AppImage"

echo "==================================="
echo "SuperConductor Installation Script"
echo "==================================="
echo ""

# Create directories if they don't exist
echo "Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$DESKTOP_DIR"
mkdir -p "$ICON_DIR"

# Download SuperConductor
echo "Downloading SuperConductor v${VERSION}..."
cd "$INSTALL_DIR"

if [ -f "superconductor" ]; then
    echo "Backing up existing installation..."
    mv superconductor "superconductor.backup.$(date +%Y%m%d-%H%M%S)"
fi

wget -O superconductor "$DOWNLOAD_URL"

# Make executable
echo "Making executable..."
chmod +x superconductor

# Extract icon from AppImage (if possible)
echo "Extracting icon..."
./superconductor --appimage-extract >/dev/null 2>&1 || true
if [ -d "squashfs-root" ]; then
    # Look for icon files
    find squashfs-root -name "*.png" -o -name "*.svg" | head -1 | while read icon; do
        if [ -n "$icon" ]; then
            cp "$icon" "$ICON_DIR/superconductor.png"
        fi
    done
    rm -rf squashfs-root
fi

# If no icon was extracted, create a placeholder or skip
if [ ! -f "$ICON_DIR/superconductor.png" ]; then
    echo "No icon extracted, desktop entry will use default icon"
    ICON_PATH="applications-multimedia"
else
    ICON_PATH="superconductor"
fi

# Create desktop entry
echo "Creating desktop shortcut..."
cat > "$DESKTOP_DIR/superconductor.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=SuperConductor
Comment=Broadcast Playout Control for CasparCG, ATEM, OBS, vMix and more
Exec=$INSTALL_DIR/superconductor
Icon=$ICON_PATH
Terminal=false
Categories=AudioVideo;Video;AudioVideoEditing;
Keywords=broadcast;playout;casparcg;atem;obs;vmix;
StartupWMClass=SuperConductor
EOF

chmod +x "$DESKTOP_DIR/superconductor.desktop"

# Create desktop icon symlink
if [ -d "$HOME/Desktop" ]; then
    echo "Creating desktop icon..."
    ln -sf "$DESKTOP_DIR/superconductor.desktop" "$HOME/Desktop/superconductor.desktop"
fi

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    echo "Updating desktop database..."
    update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
fi

# Add to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "NOTE: $HOME/.local/bin is not in your PATH"
    echo "Add this line to your ~/.bashrc or ~/.profile:"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
    echo ""
fi

echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo ""
echo "SuperConductor installed to: $INSTALL_DIR/superconductor"
echo "Desktop shortcut created"
echo ""
echo "You can now:"
echo "  - Launch from your application menu"
echo "  - Run 'superconductor' from terminal (if ~/.local/bin is in PATH)"
echo "  - Run directly: $INSTALL_DIR/superconductor"
echo ""
echo "To uninstall, run:"
echo "  rm $INSTALL_DIR/superconductor"
echo "  rm $DESKTOP_DIR/superconductor.desktop"
echo ""
