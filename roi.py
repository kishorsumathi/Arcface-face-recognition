import cv2, sys

cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Our ROI, defined by two points
p1, p2 = None, None
state = 0

# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2
    
    # Left click
    if event == cv2.EVENT_LBUTTONUP:
        # Select first point
        if state == 0:
            p1 = (x,y)
            state += 1
        # Select second point
        elif state == 1:
            p2 = (x,y)
            state += 1
    # Right click (erase current ROI)
    # if event == cv2.EVENT_RBUTTONUP:
    #     p1, p2 = None, None
    #     state = 0

# Register the mouse callback
cv2.setMouseCallback('Frame', on_mouse)

while cap.isOpened():
    val, frame = cap.read()
    print(p1)
    if p1!=None and p2!=None:
        x,y=p1
        w,z=p2
        crop = frame[y:z,x:w]
        cv2.imwrite("crop.png",crop)
    # If a ROI is selected, draw it
    if state > 1:
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 10)
    # Show image
    cv2.imshow('Frame', frame)
    
    # Let OpenCV manage window events
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # If ESCAPE key pressed, stop

cap.release()
cv2.destroyAllWindows()