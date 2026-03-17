from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
from enhancement import LowLightEnhancer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the enhancer
enhancer = LowLightEnhancer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance_image', methods=['POST'])
def enhance_image():
    try:
        # Get image data from request
        data = request.json
        image_data = data['image']
        method = data.get('method', 'clahe')
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Enhance the image
        enhanced_image = enhancer.enhance_image(image_cv, method)
        
        # Convert back to PIL for encoding
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        
        # Encode to base64
        buffer = io.BytesIO()
        enhanced_pil.save(buffer, format='JPEG')
        enhanced_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'enhanced_image': f'data:image/jpeg;base64,{enhanced_b64}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Process the file
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image = cv2.imread(filename)
            enhanced = enhancer.enhance_image(image)
            
            # Save enhanced image
            enhanced_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced_' + file.filename)
            cv2.imwrite(enhanced_filename, enhanced)
            
            return jsonify({
                'success': True,
                'original_url': f'/uploads/{file.filename}',
                'enhanced_url': f'/uploads/enhanced_{file.filename}'
            })
        
        return jsonify({'success': False, 'error': 'Unsupported file type'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    print("Starting Low-Light Image Enhancement System...")
    print("🚀 Now with REAL trained weights from LOL dataset!")
    app.run(host='127.0.0.1', port=5002, debug=True)
