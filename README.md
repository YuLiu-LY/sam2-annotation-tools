# SAM2 Video Annotation Tool

A powerful GUI application for video segmentation using Meta's [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) model. This tool allows users to annotate video frames with points and bounding boxes, then automatically generate segmentation masks across the entire video sequence.

## Features

### Core Functionality
- **Video Frame Loading**: Load image sequences from directories containing video frames
- **Interactive Annotation**: Add positive/negative points and bounding boxes to guide segmentation
- **Multi-Object Support**: Annotate multiple objects with different Object IDs
- **Real-time Segmentation**: Generate masks for individual frames or entire video sequences
- **Animation Preview**: Play through annotated frames to visualize results
- **Mask Export**: Save segmentation masks as numpy arrays and visualization images

### User Interface Components

#### File Path Settings
- **Input Directory**: Select folder containing video frame images (JPG, PNG, JPEG)
- **Save Subfolder**: Specify output directory for saved masks (default: "masks")

#### Frame Navigation
- **Previous/Next Frame**: Navigate through video frames
- **Frame Jump**: Directly jump to specific frame numbers
- **Animation Controls**: 
  - Play/Stop animation through frames
  - Adjust interval to show every Nth frame

#### Annotation Tools
- **Object ID**: Set ID for different objects (starts from 0)
- **Point Modes**: 
  - Positive Points (green): Indicate object regions
  - Negative Points (red): Indicate background regions
- **Add Point**: Click on image to add annotation points
- **Add Bounding Box**: Click twice to define rectangular regions
- **Clear Annotations**: Remove all annotations from current frame
- **Reset State**: Clear all annotations and segmentation results

#### Segmentation
- **Segment Current Frame**: Generate mask for current frame only
- **Segment All Frames**: Propagate segmentation across entire video

#### Save Results
- **Save Masks**: Export segmentation results as numpy arrays and visualization images

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- PyQt5
- SAM2 model checkpoint

### Setup
1. Install required dependencies:
```bash
cd segment-anything-2
pip install -e .
pip install PyQt5
pip install matplotlib
pip install Pillow
pip install scikit-image
```

2. Download SAM2 model checkpoint:
   - Place the checkpoint file in `./segment-anything-2/checkpoints/sam2.1_hiera_large.pt`
   - Or the application will prompt you to select the checkpoint file location

3. Ensure SAM2 library is properly installed in the `segment-anything-2/` directory

## Usage

### Starting the Application
```bash
python segment_ui.py
```

### Basic Workflow

1. **Load Video Frames**
   - Click "Browse..." to select directory containing video frame images
   - Supported formats: JPG, PNG, JPEG
   - Frames should be named in sequential order

2. **Navigate Frames**
   - Use "Previous Frame" / "Next Frame" buttons
   - Or enter frame number and click "Jump"
   - Use animation mode to preview video

3. **Add Annotations**
   - Set Object ID for the object you want to segment
   - Choose point mode (Positive/Negative)
   - Click "Add Point" then click on image to add annotation points
   - Or click "Add Bounding Box" and define rectangular region

4. **Generate Segmentation**
   - Click "Segment Current Frame" to process current frame
   - Click "Segment All Frames" to propagate across entire video
   - View results overlaid on frames

5. **Save Results**
   - Click "Save Masks" to export segmentation masks
   - Masks are saved as numpy arrays (.npy files)
   - Visualization images are also generated

### Advanced Features

#### Multi-Object Annotation
- Change Object ID to annotate different objects
- Each object maintains separate annotations and masks
- Objects are color-coded in visualization

#### Animation Mode
- Click "Animation" to start frame-by-frame playback
- Adjust "Interval" to control playback speed
- Useful for reviewing segmentation results

#### Frame Caching
- Application automatically caches frames for smooth navigation
- Cache size is limited to prevent memory issues

## Output Format

### Saved Files
- **Masks**: `{frame_name}.npy` - Binary masks as numpy arrays [K, 1, H, W]
- **Visualizations**: `{frame_name}.png` - Overlaid masks on original frames

### Directory Structure
```
output_directory/
├── masks/           # Numpy mask files
│   ├── frame_001.npy
│   ├── frame_002.npy
│   └── ...
└── masks_vis/       # Visualization images
    ├── frame_001.png
    ├── frame_002.png
    └── ...
```

## Technical Details

### Model Configuration
- Uses SAM2.1 Hiera Large model by default
- Supports CUDA, MPS, and CPU inference
- Automatic mixed precision for CUDA devices

### Performance Optimizations
- Frame caching for smooth navigation
- Optimized matplotlib rendering
- Background frame preloading
- Memory management for large video sequences

### Supported Platforms
- Linux (tested on Ubuntu)

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure SAM2 checkpoint file is in correct location
   - Check file permissions and path

2. **Memory Issues**
   - Reduce frame cache size in code
   - Close other applications to free memory

3. **No Frames Loaded**
   - Check image file formats (JPG, PNG, JPEG)
   - Ensure directory contains image files
   - Verify file naming convention



## License

This tool is built on top of Meta's Segment Anything 2 model. Please refer to the original SAM2 license for model usage terms.

## Contributing

Feel free to submit issues and enhancement requests. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Meta AI for the Segment Anything 2 model
- Open source contributors to the supporting libraries
