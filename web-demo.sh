#!/bin/bash

# Web-based VLM Inference Demo Launcher
# This script launches the web-based demo interface

echo "🌐 Starting Web-based VLM Inference Demo..."
echo "================================================"

# Check if images directory exists
if [ ! -d "./images" ]; then
    echo "❌ Error: 'images' directory not found!"
    echo "Please create an 'images' directory and add some JPG images."
    exit 1
fi

# Check if there are any JPG images
image_count=$(find ./images -name "*.jpg" -type f | wc -l)
if [ $image_count -eq 0 ]; then
    echo "⚠️  Warning: No JPG images found in 'images' directory"
    echo "Please add some JPG images to the 'images' directory."
    echo ""
fi

# Check if VLM server is running
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ VLM server is running on localhost:8080"
else
    echo "⚠️  Warning: VLM server may not be running on localhost:8080"
    echo "Please start your VLM server before running inference."
    echo ""
fi

echo "📸 Found $image_count JPG images in 'images' directory"
echo ""
echo "🚀 Starting web server..."
echo "📱 Web interface will open in your browser automatically"
echo "🔗 If it doesn't open, navigate to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"

# Run the web demo
python web-demo.py "$@"