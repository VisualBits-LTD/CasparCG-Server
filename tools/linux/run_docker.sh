#!/bin/bash

# Start Xvfb for headless OpenGL (CasparCG's EGL code requires a display)
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset > /tmp/xvfb.log 2>&1 &
XVFB_PID=$!
export DISPLAY=:99

# Give Xvfb time to start
sleep 2

# Check if Xvfb is running
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "ERROR: Xvfb failed to start"
    cat /tmp/xvfb.log
    exit 1
fi

echo "Xvfb started on DISPLAY=$DISPLAY"

# Use Mesa EGL for compatibility with Xvfb
unset __EGL_VENDOR_LIBRARY_FILENAMES
export LIBGL_ALWAYS_SOFTWARE=0

# Look for NVIDIA Vulkan ICD
echo "Checking for NVIDIA Vulkan support..."

# Check if nvidia libs are available
if ls /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so* > /dev/null 2>&1; then
    echo "Found NVIDIA libraries, checking for Vulkan ICD..."
    
    # Look for the ICD file
    if [ ! -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
        echo "Creating NVIDIA Vulkan ICD configuration..."
        
        # Create the directory if it doesn't exist
        mkdir -p /etc/vulkan/icd.d
        
        # Create a minimal NVIDIA ICD configuration
        cat > /etc/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.0"
    }
}
EOF
        echo "Created /etc/vulkan/icd.d/nvidia_icd.json"
    fi
    
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    echo "✓ NVIDIA Vulkan ICD configured"
else
    echo "✗ NVIDIA libraries not found, CEF will use software rendering"
    echo "  Set <enable-gpu>false</enable-gpu> in casparcg.config"
fi

echo ""
echo "Environment:"
echo "  DISPLAY=$DISPLAY"
echo "  VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo ""

# Start media-scanner in the background
if [ -d "/opt/casparcg/scanner" ] && [ -x "/opt/casparcg/scanner/scanner" ]; then
    echo "Starting media-scanner on port 8000..."
    cd /opt/casparcg
    ./scanner/scanner --port 8000 --media-path /opt/casparcg/media > /tmp/media-scanner.log 2>&1 &
    SCANNER_PID=$!
    
    # Give scanner time to start
    sleep 2
    
    if kill -0 $SCANNER_PID 2>/dev/null; then
        echo "✓ Media-scanner started (PID: $SCANNER_PID)"
    else
        echo "✗ Media-scanner failed to start, check /tmp/media-scanner.log"
    fi
    cd /opt/casparcg
else
    echo "✗ Media-scanner not found in /opt/casparcg/scanner"
    echo "  CLS, TLS, and THUMBNAIL commands will not work"
fi
echo ""

sh ./run.sh