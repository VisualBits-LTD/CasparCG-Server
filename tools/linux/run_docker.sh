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

# Look for NVIDIA libraries mounted by container runtime
echo "Checking for NVIDIA Vulkan support..."
ls -la /usr/lib/x86_64-linux-gnu/libnvidia* 2>/dev/null | head -5
ls -la /usr/share/vulkan/icd.d/ 2>/dev/null
ls -la /etc/vulkan/icd.d/ 2>/dev/null

# The nvidia-container-runtime should mount an ICD, look for it
if [ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    echo "Found NVIDIA Vulkan ICD: /etc/vulkan/icd.d/nvidia_icd.json"
elif [ -f "/usr/share/vulkan/icd.d/nvidia_icd.json" ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    echo "Found NVIDIA Vulkan ICD: /usr/share/vulkan/icd.d/nvidia_icd.json"
else
    # Use software Vulkan as fallback
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
    echo "NVIDIA Vulkan ICD not found, using software Vulkan (lvp)"
fi

# Debug: show environment
echo "Environment:"
echo "  DISPLAY=$DISPLAY"
echo "  VK_ICD_FILENAMES=$VK_ICD_FILENAMES"

sh ./run.sh