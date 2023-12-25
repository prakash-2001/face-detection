from flask import Flask, render_template

web = Flask(__name__, static_url_path="/static")

@web.route("/")
def index():
    return render_template('index.html')    

@web.route("/run_prg", methods=['POST'])
def prg_run():
    try:
        import cv2
        from mtcnn.mtcnn import MTCNN
        from concurrent.futures import ThreadPoolExecutor

        cv2_version = cv2.__version__.split('.')[0]

        mtcnn_detector = MTCNN(scale_factor=0.7, min_face_size=30)

        cap = cv2.VideoCapture(0)

        def detect_faces(frame):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = mtcnn_detector.detect_faces(rgb_frame)

            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            return frame

        with ThreadPoolExecutor() as executor:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error reading frame from webcam")
                    break

                frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)

                future = executor.submit(detect_faces, frame)
                result_frame = future.result()

                cv2.imshow('Face Detection', result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


    except Exception as e: 
        return f'someting went wrong {str(e)}'
    
if __name__ == "__main__":
    web.run(host="0.0.0.0", debug=True)
