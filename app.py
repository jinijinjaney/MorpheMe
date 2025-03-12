from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
import dlib
import os
from datetime import datetime
import sys

app = Flask(__name__)

cap = cv2.VideoCapture(0)

# Paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
TEMPLATE_FOLDER = 'templates'  # Folder where you store predefined images

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

# Load Haar Cascade Classifiers for Face & Eye Detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_file(filename)

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    # Returns facial landmarks as (x,y) coordinates
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    # Overlays the landmark points on the image itself
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation.

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                  c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])

def read_im_and_landmarks(image_path):
    if isinstance(image_path, str):
        im = cv2.imread(image_path)
    else:
        im = image_path  # Assume it's already an image array
        
    if im is None:
        raise Exception(f"Failed to load image: {image_path}")
        
    im = cv2.resize(im, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))

def swappy(user_image_path, template_image_path, output_path=None):
    """
    Performs face swapping between user's image and template image.
    Places user's face onto the template body.
    
    Args:
        user_image_path: Path to the user's image (or image array) - SOURCE FACE
        template_image_path: Path to the template image (or image array) - TARGET BODY
        output_path: Path where to save the output image
        
    Returns:
        Path to the swapped face image
    """
    try:
        # Read images and detect landmarks
        user_im, user_landmarks = read_im_and_landmarks(user_image_path)
        template_im, template_landmarks = read_im_and_landmarks(template_image_path)

        # Calculate transformation matrix to align user's face to template's face
        # Note: We're transforming FROM the template landmarks TO the user landmarks
        M = transformation_from_points(template_landmarks[ALIGN_POINTS],
                                      user_landmarks[ALIGN_POINTS])

        # Create a mask of user's face
        mask = get_face_mask(user_im, user_landmarks)
        
        # Warp the mask to fit the template face
        warped_mask = warp_im(mask, M, template_im.shape)
        
        # Combine the mask with the template's face mask
        combined_mask = np.max([get_face_mask(template_im, template_landmarks), warped_mask],
                              axis=0)

        # Warp the user's face to fit the template
        warped_user_im = warp_im(user_im, M, template_im.shape)
        
        # Color correct the warped face to match the template's skin tone
        warped_corrected_user_im = correct_colours(template_im, warped_user_im, template_landmarks)

        # Combine the template and the user's face
        output_im = template_im * (1.0 - combined_mask) + warped_corrected_user_im * combined_mask
        
        # If no output path provided, generate one
        if output_path is None:
            output_path = os.path.join(RESULT_FOLDER, f"uploaded_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            
        cv2.imwrite(output_path, output_im)
        return output_path
    
    except (TooManyFaces, NoFaces) as e:
        print(f"Face detection error: {str(e)}")
        return None
    except Exception as e:
        print(f"Error in face swapping: {str(e)}")
        return None

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Detect eyes within the detected face region
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route for streaming live video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_image():
    try:
        print("📸 Capture request received with template:", request.form.get("template"))
        success, frame = cap.read()
        if not success or frame is None:
            print("❌ Error: Failed to read frame from webcam.")
            return jsonify({"error": "Failed to read frame from webcam"}), 500

        # Save captured image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        user_image_path = f"{UPLOAD_FOLDER}/{timestamp}.jpg"
        cv2.imwrite(user_image_path, frame)
        print(f"✅ Image saved at {user_image_path}")

        template_name = request.form.get("template", "trump").replace(".jpg", "")
        template_path = os.path.join(TEMPLATE_FOLDER, f"{template_name}.jpg")
        
        if not os.path.exists(template_path):
            print(f"❌ Template image not found: {template_path}")
            return jsonify({"error": "Template image not found"}), 400

        output_path = f"{RESULT_FOLDER}/{timestamp}.jpg"
        
        # Call swappy with user image first (source face), then template (target body)
        swapped_face_path = swappy(user_image_path, template_path, output_path)

        if not swapped_face_path:
            print("❌ Face swap failed!")
            return jsonify({"error": "Face detection failed"}), 400

        print(f"✅ Face swap completed! Returning image: {swapped_face_path}")
        return send_file(swapped_face_path, mimetype="image/jpeg")
    except Exception as e:
        print("🔥 Capture error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    print("📤 Received upload request!")

    if "image" not in request.files or "template" not in request.form:
        print("❌ Error: Missing image or template")
        return jsonify({"error": "Missing image or template selection"}), 400

    image_file = request.files["image"]
    template_name = request.form["template"].replace(".jpg", "")  # Remove .jpg

    print(f"📸 Processing image with template: {template_name}")

    if template_name not in ["aiai", "swift", "elon", "gloria", "marcos", "pacman", "sarah", "tekla"]:
        print("❌ Invalid template selected!")
        return jsonify({"error": "Invalid template selection"}), 400

    template_path = os.path.join(TEMPLATE_FOLDER, f"{template_name}.jpg")
    if not os.path.exists(template_path):
        print(f"❌ Template image not found: {template_path}")
        return jsonify({"error": "Template image not found"}), 400

    user_image_path = os.path.join(UPLOAD_FOLDER, f"uploaded_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    image_file.save(user_image_path)
    print(f"✅ User image saved at {user_image_path}")

    output_path = os.path.join(RESULT_FOLDER, f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
    
    # Call swappy with user image first (source face), then template (target body)
    swapped_face_path = swappy(user_image_path, template_path, output_path)

    if not swapped_face_path:
        print("❌ Face swap failed!")
        return jsonify({"error": "Face detection failed"}), 400

    print(f"✅ Face swap completed! Returning image: {swapped_face_path}")
    return send_file(swapped_face_path, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(debug=True)